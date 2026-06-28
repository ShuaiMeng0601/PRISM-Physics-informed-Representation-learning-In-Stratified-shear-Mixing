import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import KHHolmboeDataset
from models import SharedEncoder, TinyUNet


class RegressionHead(nn.Module):
    def __init__(self, in_channels=64, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, feature):
        return self.mlp(self.cnn(feature))


class MultiHeadRegressionModel(nn.Module):
    """Full model with shared ViT encoder, multi-head regression, and flow branch."""

    def __init__(
        self,
        n_heads=5,
        dim=64,
        img_size=(501, 100),
        patch_size=(10, 10),
        mask_prob=0.15,
        dropout=0.2,
        n_vars=5,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.encoder = SharedEncoder(img_size=img_size, dim=dim, patch_size=patch_size, out_channels=64, n_vars=n_vars)
        self.heads = nn.ModuleList([
            RegressionHead(in_channels=64, dropout=dropout)
            for _ in range(n_heads)
        ])
        self.flow_unet = TinyUNet(feature_channels=64, cond_dim=3)

    def encode(self, x, return_attention=False):
        mask_prob = self.mask_prob if self.training else 0.0
        return self.encoder(x, mask_prob=mask_prob, return_attention=return_attention)

    def forward(self, x, return_attention=False):
        feature, attn = self.encode(x, return_attention=return_attention)
        preds = torch.stack([head(feature) for head in self.heads], dim=1)
        return preds, feature, attn

    def forward_regression(self, x, return_attention=False):
        return self.forward(x, return_attention=return_attention)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="kh_holmboe_dataset.h5")
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--n_heads", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--input_variables", default=None, help="Comma-separated input variables, e.g. buoyancy,reduced_shear,log_epsilon")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default="multihead_model.pt")
    parser.add_argument("--metrics_csv", default="multihead_loss_history.csv")
    parser.add_argument("--loss_plot", default="multihead_loss_curves.png")
    return parser.parse_args()


def make_loader(h5_path, split, label_csv, batch_size, shuffle, input_variables=None):
    dataset = KHHolmboeDataset(h5_path, split=split, label_csv=label_csv, input_variables=input_variables)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def compute_losses(preds, target, has_ratio):
    if not has_ratio.any():
        zero = torch.zeros((), device=preds.device)
        return zero, zero, zero, zero

    labeled_preds = preds[has_ratio]
    labeled_target = target[has_ratio]
    target_heads = labeled_target[:, None, :].expand_as(labeled_preds)

    head_mse = F.mse_loss(labeled_preds, target_heads)
    pred_mean = labeled_preds.mean(dim=1)
    pred_std = labeled_preds.std(dim=1, unbiased=False)
    mean_mse = F.mse_loss(pred_mean, labeled_target)
    mean_mae = torch.abs(pred_mean - labeled_target).mean()
    mean_std = pred_std.mean()

    return head_mse, mean_mse, mean_mae, mean_std


def run_one_epoch(model, loader, optimizer, device, lambda_recon):
    is_train = optimizer is not None
    model.train(is_train)

    sums = {
        "total": 0.0,
        "head_mse": 0.0,
        "mean_mse": 0.0,
        "mean_mae": 0.0,
        "mean_pred_std": 0.0,
        "recon": 0.0,
        "flow": 0.0,
        "theta_mse": 0.0,
    }
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        theta = batch["theta"].to(device)
        condition = batch["condition"].to(device)
        ratio = batch["ratio"].to(device)
        has_ratio = batch["has_ratio"].to(device)

        if is_train:
            optimizer.zero_grad()

        preds, feature, _ = model(x)
        head_mse, mean_mse, mean_mae, mean_std = compute_losses(preds, ratio, has_ratio)
        pred_v, target_v, theta_hat = model.forward_flow(theta, feature, condition)
        flow_loss = F.mse_loss(pred_v, target_v)
        theta_mse = F.mse_loss(theta_hat, theta)
        recon_loss = flow_loss + theta_mse
        total_loss = head_mse + lambda_recon * recon_loss

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
        "val_total",
        "val_head_mse",
        "val_mean_mse",
        "val_mean_mae",
        "val_mean_pred_std",
        "val_recon",
        "val_flow",
        "val_theta_mse",
        "best_val_mean_mse",
        "checkpoint_saved",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return path, fieldnames


def append_metrics_row(path, fieldnames, row):
    with path.open("a", newline="") as f:
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

    metrics_csv = Path(metrics_csv)
    output_path = Path(output_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with metrics_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    terms = ["total", "head_mse", "mean_mse", "mean_mae", "mean_pred_std", "recon", "flow", "theta_mse"]

    fig, axes = plt.subplots(len(terms), 1, figsize=(8, 18), sharex=True)
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
    device = torch.device(args.device)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    metrics_csv, metric_fields = init_metrics_csv(args.metrics_csv)

    if args.label_csv is None:
        for default_label_csv in [
            Path(__file__).with_name("RM_summary_table.csv"),
            Path(__file__).resolve().parent.parent / "RM_summary_table.csv",
        ]:
            if default_label_csv.exists():
                args.label_csv = str(default_label_csv)
                break
        if args.label_csv is not None:
            print(f"Using label CSV: {args.label_csv}", flush=True)

    train_loader = make_loader(args.h5, "train", args.label_csv, args.batch_size, shuffle=True, input_variables=args.input_variables)
    val_loader = make_loader(args.h5, "val", args.label_csv, args.batch_size, shuffle=False, input_variables=args.input_variables)
    input_variables = train_loader.dataset.input_variables
    img_size = tuple(train_loader.dataset.x_data.shape[-2:])
    if train_loader.dataset.missing_variables:
        print(f"Missing input variables filled with zeros: {train_loader.dataset.missing_variables}", flush=True)
    print(f"Using input variables: {input_variables}", flush=True)
    print(f"Using image size: {img_size}", flush=True)

    model = MultiHeadRegressionModel(
        n_heads=args.n_heads,
        dropout=args.dropout,
        mask_prob=args.mask_prob,
        n_vars=len(input_variables),
        img_size=img_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mean_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, args.lambda_recon)
        with torch.no_grad():
            val_metrics = run_one_epoch(model, val_loader, None, device, args.lambda_recon)

        checkpoint_saved = False
        if val_metrics["mean_mse"] < best_val_mean_mse:
            best_val_mean_mse = val_metrics["mean_mse"]
            torch.save(model.state_dict(), args.save)
            checkpoint_saved = True

        print(
            f"epoch {epoch:03d} | "
            f"train total {train_metrics['total']:.6f} "
            f"head_mse {train_metrics['head_mse']:.6f} "
            f"mean_mse {train_metrics['mean_mse']:.6f} "
            f"mean_mae {train_metrics['mean_mae']:.6f} "
            f"pred_std {train_metrics['mean_pred_std']:.6f} "
            f"recon {train_metrics['recon']:.6f} "
            f"flow {train_metrics['flow']:.6f} "
            f"theta_mse {train_metrics['theta_mse']:.6f} | "
            f"val total {val_metrics['total']:.6f} "
            f"head_mse {val_metrics['head_mse']:.6f} "
            f"mean_mse {val_metrics['mean_mse']:.6f} "
            f"mean_mae {val_metrics['mean_mae']:.6f} "
            f"pred_std {val_metrics['mean_pred_std']:.6f} "
            f"recon {val_metrics['recon']:.6f} "
            f"flow {val_metrics['flow']:.6f} "
            f"theta_mse {val_metrics['theta_mse']:.6f}",
            flush=True,
        )
        if checkpoint_saved:
            print(f"saved {args.save}", flush=True)

        row = {
            "epoch": epoch,
            "best_val_mean_mse": best_val_mean_mse,
            "checkpoint_saved": int(checkpoint_saved),
        }
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for name, value in metrics.items():
                row[f"{split}_{name}"] = value
        append_metrics_row(metrics_csv, metric_fields, row)
        plot_loss_curves(metrics_csv, args.loss_plot)


if __name__ == "__main__":
    main()
