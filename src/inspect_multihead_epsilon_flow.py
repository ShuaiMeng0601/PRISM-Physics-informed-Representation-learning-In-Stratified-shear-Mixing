import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import KHHolmboeDataset
from train_multihead_epsilon_flow import (
    MultiHeadEpsilonFlowModel,
    default_label_csv,
    find_epsilon_index,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--epsilon_input_mask_prob", type=float, default=0.5)
    parser.add_argument("--force_mask_epsilon", action="store_true")
    parser.add_argument("--input_variables", default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--n_feature_maps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="epsilon_flow_visualizations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def tensor_to_image(tensor):
    return tensor.detach().cpu().squeeze().numpy()


def plot_input_variables(x, variables, output_path):
    fig, axes = plt.subplots(1, len(variables), figsize=(3.7 * len(variables), 3.9), constrained_layout=True)
    if len(variables) == 1:
        axes = [axes]
    for idx, (ax, name) in enumerate(zip(axes, variables)):
        field = tensor_to_image(x[0, idx])
        image = ax.imshow(field, cmap="coolwarm", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_latent_representation(feature, output_path, n_feature_maps):
    feature = feature.detach().cpu()[0]
    feature_mean = feature.mean(dim=0)
    feature_norm = torch.linalg.vector_norm(feature, dim=0)
    n_feature_maps = min(n_feature_maps, feature.shape[0])
    panels = [
        ("feature mean", feature_mean.numpy(), "viridis"),
        ("feature L2 norm", feature_norm.numpy(), "magma"),
    ]
    for channel in range(n_feature_maps):
        panels.append((f"feature channel {channel}", feature[channel].numpy(), "coolwarm"))

    n_cols = min(4, len(panels))
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, (title, field, cmap) in zip(axes, panels):
        image = ax.imshow(field, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_three_panel(true_field, pred_field, output_path, true_title, pred_title, cmap="coolwarm"):
    true_np = tensor_to_image(true_field[0, 0])
    pred_np = tensor_to_image(pred_field[0, 0])
    error = pred_np - true_np
    vmin = min(true_np.min(), pred_np.min())
    vmax = max(true_np.max(), pred_np.max())
    err_abs = max(abs(error.min()), abs(error.max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    panels = [
        (true_title, true_np, cmap, vmin, vmax),
        (pred_title, pred_np, cmap, vmin, vmax),
        ("error", error, "seismic", -err_abs, err_abs),
    ]
    for ax, (title, field, cm, lo, hi) in zip(axes, panels):
        image = ax.imshow(field, cmap=cm, aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_generated_log_epsilon(true_eps, generated_eps, available, output_path):
    if bool(available[0].detach().cpu()):
        plot_three_panel(
            true_eps,
            generated_eps,
            output_path,
            "true log_epsilon",
            "generated log_epsilon",
            cmap="magma",
        )
        return

    field = tensor_to_image(generated_eps[0, 0])
    fig, ax = plt.subplots(figsize=(5.2, 4.3), constrained_layout=True)
    image = ax.imshow(field, cmap="magma", aspect="auto")
    ax.set_title("generated log_epsilon")
    ax.set_xlabel("time")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def inspect_sample(model, dataset, sample_idx, device, output_dir, args):
    sample_dir = output_dir / f"{dataset.split}_sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    batch = dataset[sample_idx]
    x = batch["x"].unsqueeze(0).to(device)
    theta = batch["theta"].unsqueeze(0).to(device)
    variable_mask = batch["variable_mask"].unsqueeze(0).to(device)
    condition = batch["condition"].unsqueeze(0).to(device)
    true_eps = x[:, model.epsilon_index:model.epsilon_index + 1]

    with torch.no_grad():
        out = model(x, variable_mask=variable_mask, force_mask_epsilon=args.force_mask_epsilon)
        preds = out["preds"]
        feature = out["feature"]
        pred_v, target_v, theta_hat = model.forward_flow(theta, feature, condition)
        generated_eps = model.generate_log_epsilon(feature, condition, steps=args.steps, clamp=(-1.0, 1.0))

    epsilon_available = variable_mask[:, model.epsilon_index]
    plot_input_variables(x, dataset.input_variables, sample_dir / "input_variables.png")
    plot_latent_representation(feature, sample_dir / "latent_representation.png", args.n_feature_maps)
    plot_three_panel(theta, theta_hat, sample_dir / "buoyancy_reconstruction.png", "true buoyancy", "reconstructed buoyancy")
    plot_generated_log_epsilon(true_eps, generated_eps, epsilon_available, sample_dir / "generated_log_epsilon_vs_true.png")

    torch.save(feature.detach().cpu(), sample_dir / "latent_feature.pt")
    torch.save(theta.detach().cpu(), sample_dir / "buoyancy_true.pt")
    torch.save(theta_hat.detach().cpu(), sample_dir / "buoyancy_hat.pt")
    torch.save(true_eps.detach().cpu(), sample_dir / "log_epsilon_true_or_zero.pt")
    torch.save(generated_eps.detach().cpu(), sample_dir / "log_epsilon_generated.pt")

    summary = {
        "sample": sample_idx,
        "split": dataset.split,
        "metadata": batch["metadata"],
        "input_variables": dataset.input_variables,
        "epsilon_available": bool(epsilon_available[0].detach().cpu()),
        "epsilon_masked_for_encoder": bool(out["epsilon_masked"][0].detach().cpu()),
        "force_mask_epsilon": args.force_mask_epsilon,
        "steps": args.steps,
        "ratio": batch["ratio"].tolist(),
        "has_ratio": bool(batch["has_ratio"]),
        "pred_R_M_mean": float(preds[0, :, 0].mean().detach().cpu()),
        "pred_R_M_std": float(preds[0, :, 0].std(unbiased=False).detach().cpu()),
        "buoyancy_reconstruction_mse": F.mse_loss(theta_hat, theta).item(),
        "buoyancy_flow_mse": F.mse_loss(pred_v, target_v).item(),
    }
    if bool(epsilon_available[0].detach().cpu()):
        summary["generated_log_epsilon_mse"] = F.mse_loss(generated_eps, true_eps).item()
        summary["generated_log_epsilon_mae"] = torch.abs(generated_eps - true_eps).mean().item()
    with (sample_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote epsilon-flow visualizations to {sample_dir}", flush=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.label_csv = default_label_csv(args)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KHHolmboeDataset(args.h5, split=args.split, label_csv=args.label_csv, input_variables=args.input_variables)
    epsilon_index = find_epsilon_index(dataset.input_variables)
    model = MultiHeadEpsilonFlowModel(
        n_heads=args.n_heads,
        dropout=args.dropout,
        mask_prob=args.mask_prob,
        n_vars=len(dataset.input_variables),
        img_size=tuple(dataset.x_data.shape[-2:]),
        epsilon_index=epsilon_index,
        epsilon_input_mask_prob=args.epsilon_input_mask_prob,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    last_sample = min(args.sample + args.n_samples, len(dataset))
    for sample_idx in range(args.sample, last_sample):
        inspect_sample(model, dataset, sample_idx, device, output_dir, args)
    dataset.close()


if __name__ == "__main__":
    main()
