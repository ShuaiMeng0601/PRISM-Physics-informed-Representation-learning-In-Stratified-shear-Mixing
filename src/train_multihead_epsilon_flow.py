import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import KHHolmboeDataset
from models import SharedEncoder, TinyUNet
from train_multihead import RegressionHead, compute_losses


class MultiHeadEpsilonFlowModel(nn.Module):
    """Multi-head RM model with buoyancy flow and log-epsilon flow decoders."""

    def __init__(
        self,
        n_heads=5,
        dim=64,
        img_size=(491, 200),
        patch_size=(10, 10),
        mask_prob=0.15,
        dropout=0.2,
        n_vars=3,
        epsilon_index=2,
        use_epsilon_mask_channel=True,
        epsilon_input_mask_prob=0.5,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.epsilon_index = epsilon_index
        self.use_epsilon_mask_channel = use_epsilon_mask_channel
        self.epsilon_input_mask_prob = epsilon_input_mask_prob

        encoder_vars = n_vars + int(use_epsilon_mask_channel)
        self.encoder = SharedEncoder(
            img_size=img_size,
            dim=dim,
            patch_size=patch_size,
            out_channels=64,
            n_vars=encoder_vars,
        )
        self.heads = nn.ModuleList([
            RegressionHead(in_channels=64, dropout=dropout)
            for _ in range(n_heads)
        ])
        self.flow_unet = TinyUNet(feature_channels=64, cond_dim=3)
        self.epsilon_flow_unet = TinyUNet(feature_channels=64, cond_dim=3)

    def epsilon_available(self, x, variable_mask=None):
        batch = x.shape[0]
        device = x.device
        if variable_mask is None:
            return torch.isfinite(x[:, self.epsilon_index]).flatten(1).any(dim=1)
        return variable_mask[:, self.epsilon_index].to(device=device, dtype=torch.bool)

    def prepare_encoder_input(self, x, variable_mask=None, force_mask_epsilon=None):
        if self.epsilon_index < 0 or self.epsilon_index >= x.shape[1]:
            raise ValueError(f"epsilon_index={self.epsilon_index} is outside input with {x.shape[1]} channels")

        x_in = x.clone()
        batch, _, height, width = x_in.shape
        device = x_in.device
        available = self.epsilon_available(x_in, variable_mask=variable_mask)

        if force_mask_epsilon is None:
            if self.training:
                random_mask = torch.rand(batch, device=device) < self.epsilon_input_mask_prob
                mask_epsilon = random_mask | ~available
            else:
                mask_epsilon = ~available
        elif isinstance(force_mask_epsilon, bool):
            mask_epsilon = torch.full((batch,), force_mask_epsilon, dtype=torch.bool, device=device)
            mask_epsilon = mask_epsilon | ~available
        else:
            mask_epsilon = force_mask_epsilon.to(device=device, dtype=torch.bool) | ~available

        x_in[mask_epsilon, self.epsilon_index] = 0.0

        if self.use_epsilon_mask_channel:
            epsilon_visible = (~mask_epsilon).float().view(batch, 1, 1, 1).expand(batch, 1, height, width)
            x_in = torch.cat([x_in, epsilon_visible], dim=1)

        return x_in, mask_epsilon, available

    def encode(self, x, variable_mask=None, force_mask_epsilon=None, return_attention=False):
        x_in, epsilon_masked, epsilon_available = self.prepare_encoder_input(
            x,
            variable_mask=variable_mask,
            force_mask_epsilon=force_mask_epsilon,
        )
        token_mask_prob = self.mask_prob if self.training else 0.0
        feature, attn = self.encoder(x_in, mask_prob=token_mask_prob, return_attention=return_attention)
        return feature, attn, epsilon_masked, epsilon_available

    def forward(self, x, variable_mask=None, force_mask_epsilon=None, return_attention=False):
        feature, attn, epsilon_masked, epsilon_available = self.encode(
            x,
            variable_mask=variable_mask,
            force_mask_epsilon=force_mask_epsilon,
            return_attention=return_attention,
        )
        preds = torch.stack([head(feature) for head in self.heads], dim=1)
        return {
            "preds": preds,
            "feature": feature,
            "attention": attn,
            "epsilon_masked": epsilon_masked,
            "epsilon_available": epsilon_available,
        }

    def forward_flow(self, theta, feature, condition):
        batch = theta.shape[0]
        noise = torch.randn_like(theta)
        flow_time = torch.rand(batch, device=theta.device)
        t = flow_time[:, None, None, None]
        noisy_theta = (1.0 - t) * noise + t * theta
        target_velocity = theta - noise
        pred_velocity = self.flow_unet(noisy_theta, feature, condition, flow_time)
        theta_hat = noisy_theta + (1.0 - t) * pred_velocity
        return pred_velocity, target_velocity, theta_hat

    def forward_epsilon_flow(self, log_epsilon, feature, condition):
        batch = log_epsilon.shape[0]
        noise = torch.randn_like(log_epsilon)
        flow_time = torch.rand(batch, device=log_epsilon.device)
        t = flow_time[:, None, None, None]
        noisy_epsilon = (1.0 - t) * noise + t * log_epsilon
        target_velocity = log_epsilon - noise
        pred_velocity = self.epsilon_flow_unet(noisy_epsilon, feature, condition, flow_time)
        epsilon_hat = noisy_epsilon + (1.0 - t) * pred_velocity
        return pred_velocity, target_velocity, epsilon_hat

    @torch.no_grad()
    def generate_log_epsilon(self, feature, condition, steps=32, noise=None, clamp=None):
        batch, _, height, width = feature.shape
        if noise is None:
            field = torch.randn(batch, 1, height, width, device=feature.device, dtype=feature.dtype)
        else:
            field = noise.to(device=feature.device, dtype=feature.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            flow_time = torch.full(
                (batch,),
                (step + 0.5) / steps,
                device=feature.device,
                dtype=feature.dtype,
            )
            velocity = self.epsilon_flow_unet(field, feature, condition, flow_time)
            field = field + dt * velocity
            if clamp is not None:
                field = field.clamp(min=clamp[0], max=clamp[1])
        return field


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_input_mask_prob", type=float, default=0.5)
    parser.add_argument("--eval_force_mask_epsilon", action="store_true")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--n_heads", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_epsilon_mask_channel", action="store_true")
    parser.add_argument("--no_epsilon_mask_channel", dest="use_epsilon_mask_channel", action="store_false")
    parser.set_defaults(use_epsilon_mask_channel=True)
    parser.add_argument("--input_variables", default=None, help="Comma-separated variables, e.g. buoyancy,reduced_shear,log_epsilon")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default="multihead_epsilon_flow_model.pt")
    parser.add_argument("--metrics_csv", default="multihead_epsilon_flow_loss_history.csv")
    parser.add_argument("--loss_plot", default="multihead_epsilon_flow_loss_curves.png")
    return parser.parse_args()


def default_label_csv(args):
    if args.label_csv is not None:
        return args.label_csv
    for candidate in [
        Path(__file__).with_name("RM_summary_table.csv"),
        Path(__file__).resolve().parent.parent / "RM_summary_table.csv",
        Path(__file__).with_name("test_RM_summary_table.csv"),
        Path(__file__).resolve().parent.parent / "test_data" / "test_RM_summary_table.csv",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def find_epsilon_index(input_variables):
    for name in ["log_epsilon", "epsilon", "eps"]:
        if name in input_variables:
            return input_variables.index(name)
    raise ValueError(f"Could not find log_epsilon/epsilon in input variables: {input_variables}")


def collate_training_batch(items):
    batch = {}
    tensor_keys = ["x", "theta", "condition", "ratio", "has_ratio", "variable_mask"]
    for key in tensor_keys:
        batch[key] = torch.stack([item[key] for item in items])
    batch["metadata"] = [item["metadata"] for item in items]
    return batch


def make_loader(h5_path, split, label_csv, batch_size, shuffle, input_variables=None):
    dataset = KHHolmboeDataset(h5_path, split=split, label_csv=label_csv, input_variables=input_variables)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_training_batch,
    )


def run_one_epoch(model, loader, optimizer, device, args):
    is_train = optimizer is not None
    model.train(is_train)
    force_mask = None if is_train else args.eval_force_mask_epsilon

    sums = {
        "total": 0.0,
        "head_mse": 0.0,
        "mean_mse": 0.0,
        "mean_mae": 0.0,
        "mean_pred_std": 0.0,
        "recon": 0.0,
        "flow": 0.0,
        "theta_mse": 0.0,
        "epsilon": 0.0,
        "epsilon_flow": 0.0,
        "epsilon_mse": 0.0,
        "epsilon_available_frac": 0.0,
        "epsilon_masked_frac": 0.0,
    }
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        theta = batch["theta"].to(device)
        condition = batch["condition"].to(device)
        ratio = batch["ratio"].to(device)
        has_ratio = batch["has_ratio"].to(device)
        variable_mask = batch["variable_mask"].to(device)
        log_epsilon = x[:, model.epsilon_index:model.epsilon_index + 1]

        if is_train:
            optimizer.zero_grad()

        out = model(x, variable_mask=variable_mask, force_mask_epsilon=force_mask)
        preds = out["preds"]
        feature = out["feature"]
        epsilon_available = out["epsilon_available"]
        epsilon_masked = out["epsilon_masked"]

        head_mse, mean_mse, mean_mae, mean_std = compute_losses(preds, ratio, has_ratio)

        pred_v, target_v, theta_hat = model.forward_flow(theta, feature, condition)
        flow_loss = F.mse_loss(pred_v, target_v)
        theta_mse = F.mse_loss(theta_hat, theta)
        recon_loss = flow_loss + theta_mse

        if epsilon_available.any():
            eps_pred_v, eps_target_v, eps_hat = model.forward_epsilon_flow(log_epsilon, feature, condition)
            epsilon_flow_loss = F.mse_loss(eps_pred_v[epsilon_available], eps_target_v[epsilon_available])
            epsilon_mse = F.mse_loss(eps_hat[epsilon_available], log_epsilon[epsilon_available])
            epsilon_loss = epsilon_flow_loss + epsilon_mse
        else:
            epsilon_flow_loss = torch.zeros((), device=device)
            epsilon_mse = torch.zeros((), device=device)
            epsilon_loss = torch.zeros((), device=device)

        total_loss = head_mse + args.lambda_recon * recon_loss + args.lambda_epsilon * epsilon_loss

        if is_train:
            total_loss.backward()
            optimizer.step()

        sums["total"] += float(total_loss.detach())
        sums["head_mse"] += float(head_mse.detach())
        sums["mean_mse"] += float(mean_mse.detach())
        sums["mean_mae"] += float(mean_mae.detach())
        sums["mean_pred_std"] += float(mean_std.detach())
        sums["recon"] += float(recon_loss.detach())
        sums["flow"] += float(flow_loss.detach())
        sums["theta_mse"] += float(theta_mse.detach())
        sums["epsilon"] += float(epsilon_loss.detach())
        sums["epsilon_flow"] += float(epsilon_flow_loss.detach())
        sums["epsilon_mse"] += float(epsilon_mse.detach())
        sums["epsilon_available_frac"] += float(epsilon_available.float().mean().detach())
        sums["epsilon_masked_frac"] += float(epsilon_masked.float().mean().detach())
        n_batches += 1

    return {name: value / max(n_batches, 1) for name, value in sums.items()}


def init_metrics_csv(path):
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_total",
        "train_head_mse",
        "train_mean_mse",
        "train_mean_mae",
        "train_mean_pred_std",
        "train_recon",
        "train_flow",
        "train_theta_mse",
        "train_epsilon",
        "train_epsilon_flow",
        "train_epsilon_mse",
        "train_epsilon_available_frac",
        "train_epsilon_masked_frac",
        "val_total",
        "val_head_mse",
        "val_mean_mse",
        "val_mean_mae",
        "val_mean_pred_std",
        "val_recon",
        "val_flow",
        "val_theta_mse",
        "val_epsilon",
        "val_epsilon_flow",
        "val_epsilon_mse",
        "val_epsilon_available_frac",
        "val_epsilon_masked_frac",
        "best_val_mean_mse",
        "checkpoint_saved",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return path, fieldnames


def append_metrics_row(path, fieldnames, row):
    with Path(path).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def plot_loss_curves(metrics_csv, output_path):
    if output_path is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping loss plot.", flush=True)
        return

    rows = []
    with Path(metrics_csv).open("r", newline="") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        return

    output_path = Path(output_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in rows]
    terms = ["total", "mean_mse", "mean_mae", "recon", "epsilon", "epsilon_mse"]
    fig, axes = plt.subplots(len(terms), 1, figsize=(8, 15), sharex=True)
    for ax, term in zip(axes, terms):
        ax.plot(epochs, [float(row[f"train_{term}"]) for row in rows], label="train")
        ax.plot(epochs, [float(row[f"val_{term}"]) for row in rows], label="val")
        ax.set_ylabel(term)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.label_csv = default_label_csv(args)
    device = torch.device(args.device)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    metrics_csv, metric_fields = init_metrics_csv(args.metrics_csv)

    train_loader = make_loader(args.h5, "train", args.label_csv, args.batch_size, True, args.input_variables)
    val_loader = make_loader(args.h5, "val", args.label_csv, args.batch_size, False, args.input_variables)
    input_variables = train_loader.dataset.input_variables
    img_size = tuple(train_loader.dataset.x_data.shape[-2:])
    epsilon_index = find_epsilon_index(input_variables)

    print(f"Using label CSV: {args.label_csv}", flush=True)
    print(f"Using input variables: {input_variables}", flush=True)
    print(f"Using epsilon index: {epsilon_index}", flush=True)
    print(f"Using image size: {img_size}", flush=True)

    model = MultiHeadEpsilonFlowModel(
        n_heads=args.n_heads,
        dropout=args.dropout,
        mask_prob=args.mask_prob,
        n_vars=len(input_variables),
        img_size=img_size,
        epsilon_index=epsilon_index,
        use_epsilon_mask_channel=args.use_epsilon_mask_channel,
        epsilon_input_mask_prob=args.epsilon_input_mask_prob,
    ).to(device)
    if args.init_checkpoint is not None:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Loaded initial checkpoint: {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mean_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, args)
        with torch.no_grad():
            val_metrics = run_one_epoch(model, val_loader, None, device, args)

        checkpoint_saved = False
        if val_metrics["mean_mse"] < best_val_mean_mse:
            best_val_mean_mse = val_metrics["mean_mse"]
            torch.save(model.state_dict(), args.save)
            checkpoint_saved = True

        print(
            f"epoch {epoch:03d} | "
            f"train total {train_metrics['total']:.6f} "
            f"mean_mse {train_metrics['mean_mse']:.6f} "
            f"eps {train_metrics['epsilon']:.6f} "
            f"eps_mse {train_metrics['epsilon_mse']:.6f} "
            f"eps_mask {train_metrics['epsilon_masked_frac']:.3f} | "
            f"val total {val_metrics['total']:.6f} "
            f"mean_mse {val_metrics['mean_mse']:.6f} "
            f"eps {val_metrics['epsilon']:.6f} "
            f"eps_mse {val_metrics['epsilon_mse']:.6f} "
            f"eps_mask {val_metrics['epsilon_masked_frac']:.3f}",
            flush=True,
        )
        if checkpoint_saved:
            print(f"saved {args.save}", flush=True)

        row = {"epoch": epoch, "best_val_mean_mse": best_val_mean_mse, "checkpoint_saved": int(checkpoint_saved)}
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for name, value in metrics.items():
                row[f"{split}_{name}"] = value
        append_metrics_row(metrics_csv, metric_fields, row)
        plot_loss_curves(metrics_csv, args.loss_plot)

    train_loader.dataset.close()
    val_loader.dataset.close()


if __name__ == "__main__":
    main()
