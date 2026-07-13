import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import KHHolmboeDataset
from train_shared_fno_rm_flow import (
    DEFAULT_VARIABLES,
    SharedFNOFlowModel,
    compute_rm_saliency,
    default_label_csv,
)


def parse_pair(value):
    if isinstance(value, tuple):
        return value
    parts = [int(item.strip()) for item in str(value).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated integers, e.g. 10,10")
    return tuple(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/kh_holmboe_dataset_keep_epsilon.h5")
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/shared_fno_rm_flow_model.pt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--input_variables", default=DEFAULT_VARIABLES)
    parser.add_argument("--output_dir", default="outputs/shared_fno_attention")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Fallback model args used only if the checkpoint does not contain saved args.
    parser.add_argument("--patch_size", type=parse_pair, default=(10, 10))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--vit_depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--fno_modes", type=parse_pair, default=(12, 8))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rm_dropout", type=float, default=0.1)
    parser.add_argument("--flow_base_channels", type=int, default=64)
    return parser.parse_args()


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def normalize_map(array, eps=1e-8):
    array = np.asarray(array, dtype=np.float32)
    return (array - array.min()) / (array.max() - array.min() + eps)


def make_model_from_checkpoint(checkpoint, dataset, args, device):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        saved_args = checkpoint.get("args", {})
        state_dict = checkpoint["model_state"]
        img_size = tuple(checkpoint.get("img_size", tuple(dataset.x_data.shape[-2:])))
    else:
        saved_args = {}
        state_dict = checkpoint
        img_size = tuple(dataset.x_data.shape[-2:])

    def get_arg(name, default):
        return saved_args.get(name, default)

    model = SharedFNOFlowModel(
        img_size=img_size,
        patch_size=tuple(get_arg("patch_size", args.patch_size)),
        n_vars=len(dataset.input_variables),
        dim=int(get_arg("dim", args.dim)),
        vit_depth=int(get_arg("vit_depth", args.vit_depth)),
        heads=int(get_arg("heads", args.heads)),
        latent_channels=int(get_arg("latent_channels", args.latent_channels)),
        fno_modes=tuple(get_arg("fno_modes", args.fno_modes)),
        dropout=float(get_arg("dropout", args.dropout)),
        rm_dropout=float(get_arg("rm_dropout", args.rm_dropout)),
        flow_base_channels=int(get_arg("flow_base_channels", args.flow_base_channels)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def decompose_attention(attention, n_vars, grid_size):
    """
    Convert cross-attention from (heads, V*P, V*P) to interpretable maps.

    Returns:
        var_matrix: V x V, query variable to key variable attention mass.
        pair_heatmaps: V x V x Hy x Wt, spatial key attention for each query/key pair.
        key_heatmaps: V x Hy x Wt, attention received by each key variable/patch.
        spatial_heatmap: Hy x Wt, key attention summed over variables.
    """
    grid_y, grid_t = grid_size
    n_patches = grid_y * grid_t
    attn = attention.mean(dim=0)
    attn = attn.reshape(n_vars, n_patches, n_vars, n_patches)

    var_matrix = attn.sum(dim=3).mean(dim=1)
    pair_heatmaps = attn.mean(dim=1).reshape(n_vars, n_vars, grid_y, grid_t)
    key_heatmaps = attn.mean(dim=(0, 1)).reshape(n_vars, grid_y, grid_t)
    spatial_heatmap = key_heatmaps.sum(dim=0)
    return var_matrix, pair_heatmaps, key_heatmaps, spatial_heatmap


def upsample_maps(maps, size):
    tensor = torch.as_tensor(maps, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
    if tensor.ndim == 3:
        tensor = tensor[None]
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)[0].numpy()
    raise ValueError(f"expected 2D or 3D maps, got shape {tuple(tensor.shape)}")


def plot_input_fields(x, variables, output_path):
    fields = tensor_to_numpy(x[0])
    fig, axes = plt.subplots(1, len(variables), figsize=(4.2 * len(variables), 4), constrained_layout=True)
    if len(variables) == 1:
        axes = [axes]
    for idx, (ax, name) in enumerate(zip(axes, variables)):
        image = ax.imshow(fields[idx], cmap="coolwarm", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_variable_matrix(var_matrix, variables, output_path):
    matrix = tensor_to_numpy(var_matrix)
    fig, ax = plt.subplots(figsize=(5.4, 4.7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(matrix.max(), 1e-8))
    ax.set_xticks(np.arange(len(variables)))
    ax.set_yticks(np.arange(len(variables)))
    ax.set_xticklabels(variables, rotation=35, ha="right")
    ax.set_yticklabels(variables)
    ax.set_xlabel("key / attended variable")
    ax.set_ylabel("query variable")
    ax.set_title("Cross-variable attention mass")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_patch_heatmaps(key_heatmaps, spatial_heatmap, variables, output_path):
    key_maps = tensor_to_numpy(key_heatmaps)
    spatial = tensor_to_numpy(spatial_heatmap)
    panels = list(zip(variables, key_maps)) + [("sum over variables", spatial)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.8), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, heatmap) in zip(axes, panels):
        image = ax.imshow(normalize_map(heatmap), cmap="inferno", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("patch time")
        ax.set_ylabel("patch y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_query_key_heatmaps(pair_heatmaps, var_matrix, variables, output_path):
    pair_maps = tensor_to_numpy(pair_heatmaps)
    matrix = tensor_to_numpy(var_matrix)
    n_vars = len(variables)
    fig, axes = plt.subplots(
        n_vars,
        n_vars,
        figsize=(3.3 * n_vars, 3.0 * n_vars),
        constrained_layout=True,
        squeeze=False,
    )
    for query_idx, query_name in enumerate(variables):
        for key_idx, key_name in enumerate(variables):
            ax = axes[query_idx, key_idx]
            heatmap = normalize_map(pair_maps[query_idx, key_idx])
            image = ax.imshow(heatmap, cmap="inferno", aspect="auto", vmin=0.0, vmax=1.0)
            ax.set_title(f"{query_name} -> {key_name}\nmass={matrix[query_idx, key_idx]:.2f}")
            if query_idx == n_vars - 1:
                ax.set_xlabel("key patch time")
            else:
                ax.set_xticks([])
            if key_idx == 0:
                ax.set_ylabel("key patch y")
            else:
                ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_attention_overlays(x, key_heatmaps, spatial_heatmap, variables, output_path):
    fields = tensor_to_numpy(x[0])
    height, width = fields.shape[-2:]
    upsampled = upsample_maps(tensor_to_numpy(key_heatmaps), size=(height, width))
    upsampled_spatial = upsample_maps(tensor_to_numpy(spatial_heatmap), size=(height, width))

    panels = []
    for idx, name in enumerate(variables):
        panels.append((name, fields[idx], normalize_map(upsampled[idx])))
    panels.append(("sum over variables", fields[0], normalize_map(upsampled_spatial)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.1), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, field, heatmap) in zip(axes, panels):
        ax.imshow(field, cmap="gray", aspect="auto")
        overlay = ax.imshow(heatmap, cmap="inferno", alpha=0.48, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_saliency(saliency, x, output_path):
    saliency_np = tensor_to_numpy(saliency[0])
    field = tensor_to_numpy(x[0, 0])
    height, width = field.shape
    saliency_up = upsample_maps(saliency_np, size=(height, width))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    image0 = axes[0].imshow(normalize_map(saliency_np), cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    axes[0].set_title("RM saliency on compact latent")
    axes[0].set_xlabel("patch time")
    axes[0].set_ylabel("patch y")
    fig.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(field, cmap="gray", aspect="auto")
    image1 = axes[1].imshow(normalize_map(saliency_up), cmap="magma", alpha=0.48, aspect="auto", vmin=0.0, vmax=1.0)
    axes[1].set_title("RM saliency overlay on buoyancy")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("y")
    fig.colorbar(image1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.label_csv = default_label_csv(args.label_csv)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / f"{args.split}_sample_{args.sample:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KHHolmboeDataset(
        args.h5,
        split=args.split,
        label_csv=args.label_csv,
        input_variables=args.input_variables,
    )
    sample = dataset[args.sample]
    x = sample["x"].unsqueeze(0).to(device)
    var_mask = sample["variable_mask"].unsqueeze(0).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = make_model_from_checkpoint(checkpoint, dataset, args, device)

    with torch.no_grad():
        out = model(x, var_mask=var_mask, return_attention=True)

    attention = out["attention"]["cross_attention"][0].detach().cpu()
    var_matrix, pair_heatmaps, key_heatmaps, spatial_heatmap = decompose_attention(
        attention,
        n_vars=len(dataset.input_variables),
        grid_size=model.encoder.grid_size,
    )

    saliency, saliency_rm = compute_rm_saliency(model, x, var_mask=var_mask)
    rm_pred = float(out["rm"][0, 0].detach().cpu())
    rm_saliency_pred = float(saliency_rm[0, 0].detach().cpu())

    plot_input_fields(x, dataset.input_variables, output_dir / "input_fields.png")
    plot_variable_matrix(var_matrix, dataset.input_variables, output_dir / "variable_attention_matrix.png")
    plot_query_key_heatmaps(
        pair_heatmaps,
        var_matrix,
        dataset.input_variables,
        output_dir / "query_key_attention_heatmaps.png",
    )
    plot_patch_heatmaps(
        key_heatmaps,
        spatial_heatmap,
        dataset.input_variables,
        output_dir / "patch_attention_heatmaps.png",
    )
    plot_attention_overlays(
        x,
        key_heatmaps,
        spatial_heatmap,
        dataset.input_variables,
        output_dir / "attention_overlays.png",
    )
    plot_saliency(saliency, x, output_dir / "rm_saliency.png")

    np.savez(
        output_dir / "attention_maps.npz",
        variable_attention=tensor_to_numpy(var_matrix),
        query_key_attention_heatmaps=tensor_to_numpy(pair_heatmaps),
        key_attention_heatmaps=tensor_to_numpy(key_heatmaps),
        spatial_attention=tensor_to_numpy(spatial_heatmap),
        rm_saliency=tensor_to_numpy(saliency[0]),
    )

    summary = {
        "split": args.split,
        "sample": args.sample,
        "metadata": sample["metadata"],
        "input_variables": dataset.input_variables,
        "variable_mask": sample["variable_mask"].tolist(),
        "rm_true": sample["ratio"].tolist(),
        "has_rm": bool(sample["has_ratio"]),
        "rm_pred": rm_pred,
        "rm_pred_from_saliency_pass": rm_saliency_pred,
        "attention_shape": list(attention.shape),
        "grid_size": list(model.encoder.grid_size),
        "outputs": [
            "input_fields.png",
            "variable_attention_matrix.png",
            "query_key_attention_heatmaps.png",
            "patch_attention_heatmaps.png",
            "attention_overlays.png",
            "rm_saliency.png",
            "attention_maps.npz",
        ],
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote attention visualizations to {output_dir}", flush=True)
    print(f"RM pred: {rm_pred:.6f}", flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
