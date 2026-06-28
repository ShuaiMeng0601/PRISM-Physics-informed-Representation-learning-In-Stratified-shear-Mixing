import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import KHHolmboeDataset
from train_multihead import MultiHeadRegressionModel


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
    parser.add_argument("--input_variables", default=None, help="Comma-separated variables, e.g. buoyancy,reduced_shear,log_epsilon")
    parser.add_argument("--flow_time", type=float, default=None, help="Fixed t in [0, 1]. If omitted, sample random t.")
    parser.add_argument("--n_feature_maps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="flow_input_visualizations")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def plot_flow_state(theta, noise, noisy_theta, target_velocity, pred_velocity, output_path):
    panels = [
        ("true buoyancy x1", theta[0, 0], "coolwarm"),
        ("random noise x0", noise[0, 0], "coolwarm"),
        ("flow input x_t", noisy_theta[0, 0], "coolwarm"),
        ("target velocity x1 - x0", target_velocity[0, 0], "seismic"),
        ("predicted velocity", pred_velocity[0, 0], "seismic"),
        ("velocity error", pred_velocity[0, 0] - target_velocity[0, 0], "seismic"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.3), constrained_layout=True)
    axes = axes.ravel()

    for ax, (title, tensor, cmap) in zip(axes, panels):
        field = tensor_to_image(tensor)
        if cmap == "seismic":
            vmax = max(abs(field.min()), abs(field.max()))
            vmin = -vmax
        else:
            vmin = field.min()
            vmax = field.max()
        image = ax.imshow(field, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_flow_unet_inputs(noisy_theta, feature, condition, flow_time, output_path, n_feature_maps):
    feature = feature.detach().cpu()[0]
    feature_mean = feature.mean(dim=0)
    feature_norm = torch.linalg.vector_norm(feature, dim=0)
    n_feature_maps = min(n_feature_maps, feature.shape[0])

    panels = [
        ("x_t / noisy buoyancy", tensor_to_image(noisy_theta[0, 0]), "coolwarm"),
        ("feature mean", feature_mean.numpy(), "viridis"),
        ("feature L2 norm", feature_norm.numpy(), "magma"),
    ]
    for idx in range(n_feature_maps):
        panels.append((f"feature channel {idx}", feature[idx].numpy(), "coolwarm"))

    n_cols = 3
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.8 * n_rows), constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (title, field, cmap) in zip(axes, panels):
        image = ax.imshow(field, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(panels):]:
        ax.axis("off")

    cond_values = [float(value) for value in condition[0].detach().cpu()]
    t_value = float(flow_time[0].detach().cpu())
    fig.suptitle(f"Flow UNet input summaries | condition=[Ri={cond_values[0]:.3g}, a={cond_values[1]:.3g}, Re/1000={cond_values[2]:.3g}], t={t_value:.3f}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_condition_and_time_maps(condition, flow_time, shape, output_path):
    height, width = shape
    cond_values = [float(value) for value in condition[0].detach().cpu()]
    t_value = float(flow_time[0].detach().cpu())
    maps = [
        ("Ri map", torch.full((height, width), cond_values[0])),
        ("a map", torch.full((height, width), cond_values[1])),
        ("Re/1000 map", torch.full((height, width), cond_values[2])),
        ("flow time map", torch.full((height, width), t_value)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    for ax, (title, field) in zip(axes, maps):
        image = ax.imshow(field.numpy(), cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visualize_sample(model, dataset, sample_idx, device, output_dir, args):
    sample_dir = output_dir / f"{dataset.split}_sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    batch = dataset[sample_idx]
    x = batch["x"].unsqueeze(0).to(device)
    theta = batch["theta"].unsqueeze(0).to(device)
    condition = batch["condition"].unsqueeze(0).to(device)

    with torch.no_grad():
        preds, feature, _ = model.forward_regression(x, return_attention=False)
        noise = torch.randn_like(theta)
        if args.flow_time is None:
            flow_time = torch.rand(theta.shape[0], device=device)
        else:
            flow_time = torch.full((theta.shape[0],), args.flow_time, device=device)

        t = flow_time[:, None, None, None]
        noisy_theta = (1.0 - t) * noise + t * theta
        target_velocity = theta - noise
        pred_velocity = model.flow_unet(noisy_theta, feature, condition, flow_time)
        theta_hat_one_step = noisy_theta + (1.0 - t) * pred_velocity

    plot_input_variables(x, dataset.input_variables, sample_dir / "encoder_input_variables.png")
    plot_flow_state(theta, noise, noisy_theta, target_velocity, pred_velocity, sample_dir / "flow_matching_state.png")
    plot_flow_unet_inputs(noisy_theta, feature, condition, flow_time, sample_dir / "flow_unet_inputs.png", args.n_feature_maps)
    plot_condition_and_time_maps(condition, flow_time, theta.shape[-2:], sample_dir / "condition_and_time_maps.png")

    torch.save(x.detach().cpu(), sample_dir / "encoder_input_x.pt")
    torch.save(feature.detach().cpu(), sample_dir / "latent_feature.pt")
    torch.save(theta.detach().cpu(), sample_dir / "true_buoyancy.pt")
    torch.save(noise.detach().cpu(), sample_dir / "random_noise.pt")
    torch.save(noisy_theta.detach().cpu(), sample_dir / "flow_input_noisy_theta.pt")
    torch.save(target_velocity.detach().cpu(), sample_dir / "target_velocity.pt")
    torch.save(pred_velocity.detach().cpu(), sample_dir / "pred_velocity.pt")
    torch.save(theta_hat_one_step.detach().cpu(), sample_dir / "theta_hat_one_step.pt")

    pred_mean = float(preds[0, :, 0].mean().detach().cpu())
    pred_std = float(preds[0, :, 0].std(unbiased=False).detach().cpu())
    summary = {
        "sample": sample_idx,
        "split": dataset.split,
        "metadata": batch["metadata"],
        "input_variables": dataset.input_variables,
        "condition": batch["condition"].tolist(),
        "flow_time": float(flow_time[0].detach().cpu()),
        "ratio": batch["ratio"].tolist(),
        "has_ratio": bool(batch["has_ratio"]),
        "pred_R_M_mean": pred_mean,
        "pred_R_M_std": pred_std,
        "flow_unet_channel_layout": [
            "noisy_theta / x_t: 1 channel",
            "latent feature: 64 channels",
            "condition maps: 3 channels [Ri, a, Re/1000]",
            "flow time map: 1 channel",
        ],
        "flow_unet_total_input_channels": 69,
        "velocity_mse": F.mse_loss(pred_velocity, target_velocity).item(),
        "one_step_theta_mse": F.mse_loss(theta_hat_one_step, theta).item(),
    }
    with (sample_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote flow input visualizations to {sample_dir}", flush=True)


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
    model = MultiHeadRegressionModel(
        n_heads=args.n_heads,
        dropout=args.dropout,
        mask_prob=args.mask_prob,
        n_vars=len(dataset.input_variables),
        img_size=tuple(dataset.x_data.shape[-2:]),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    last_sample = min(args.sample + args.n_samples, len(dataset))
    for sample_idx in range(args.sample, last_sample):
        visualize_sample(model, dataset, sample_idx, device, output_dir, args)

    dataset.close()


if __name__ == "__main__":
    main()
