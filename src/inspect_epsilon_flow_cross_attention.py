import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    parser.add_argument("--show_epsilon_visible_key", action="store_true")
    parser.add_argument("--show_epsilon_visible_query", action="store_true")
    parser.add_argument("--include_self_attention", action="store_true")
    parser.add_argument("--display_percentile", type=float, default=95.0)
    parser.add_argument(
        "--mass_vmax",
        default=None,
        help="Comma-separated colorbar maxima for mass maps in row-major order.",
    )
    parser.add_argument("--output_dir", default="epsilon_flow_cross_attention")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_matrix(matrix, query_names, key_names, output_path, title):
    fig, ax = plt.subplots(
        figsize=(1.8 + 0.95 * len(key_names), 1.7 + 0.8 * len(query_names)),
        constrained_layout=True,
    )
    vmax = max(0.1, float(matrix.max()))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(key_names)))
    ax.set_yticks(range(len(query_names)))
    ax.set_xticklabels(key_names, rotation=35, ha="right")
    ax.set_yticklabels(query_names)
    ax.set_xlabel("key / attended variable")
    ax.set_ylabel("query variable")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def transform_panel(field, mode, percentile=95.0):
    if mode == "mass":
        return field, "attention mass"
    if mode == "max":
        panel_max = np.nanmax(field)
        if panel_max > 0:
            return field / panel_max, "max-normalized attention"
        return field, "max-normalized attention"
    if mode == "percentile":
        positive = field[field > 0]
        if positive.size:
            hi = np.nanpercentile(positive, percentile)
            if hi > 0:
                return np.clip(field / hi, 0.0, 1.0), f"{percentile:g}th-percentile clipped attention"
        return field, f"{percentile:g}th-percentile clipped attention"
    if mode == "log":
        panel_max = np.nanmax(field)
        if panel_max > 0:
            return np.log1p(1000.0 * field / panel_max), "log-scaled attention"
        return field, "log-scaled attention"
    raise ValueError(f"Unknown spatial map mode: {mode}")


def save_spatial_maps(maps, query_names, key_names, grid_h, grid_w, output_path, mode="mass", percentile=95.0):
    n_queries = len(query_names)
    n_keys = len(key_names)
    fig, axes = plt.subplots(
        n_queries,
        n_keys,
        figsize=(3.05 * n_keys, 2.65 * n_queries),
        constrained_layout=True,
    )
    if n_queries == 1 and n_keys == 1:
        axes = np.array([[axes]])
    elif n_queries == 1:
        axes = axes[None, :]
    elif n_keys == 1:
        axes = axes[:, None]
    colorbar_label = None
    for query_idx, query_name in enumerate(query_names):
        for key_idx, key_name in enumerate(key_names):
            ax = axes[query_idx, key_idx]
            raw_field = maps[query_idx, key_idx].reshape(grid_h, grid_w)
            field, colorbar_label = transform_panel(raw_field, mode, percentile=percentile)
            panel_max = float(np.nanmax(field)) if field.size else 1.0
            if panel_max <= 0:
                panel_max = 1.0
            image = ax.imshow(field, cmap="magma", aspect="auto", vmin=0.0, vmax=panel_max)
            if query_idx == 0:
                ax.set_title(f"key: {key_name}", fontsize=9)
            if key_idx == 0:
                ax.set_ylabel(f"query:\n{query_name}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=7)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pairwise_matrix(matrix, query_names, row_key_names, output_path, title):
    n_queries = len(query_names)
    n_key_rows = max(len(keys) for keys in row_key_names)
    fig, ax = plt.subplots(
        figsize=(2.2 + 1.35 * n_queries, 1.7 + 0.8 * n_key_rows),
        constrained_layout=True,
    )
    plot_matrix = np.full((n_key_rows, n_queries), np.nan, dtype=float)
    for query_idx, keys in enumerate(row_key_names):
        for key_row in range(len(keys)):
            plot_matrix[key_row, query_idx] = matrix[query_idx, key_row]
    vmax = max(0.1, float(np.nanmax(matrix)))
    image = ax.imshow(plot_matrix, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n_queries))
    ax.set_yticks(range(n_key_rows))
    ax.set_xticklabels(query_names, rotation=25, ha="right")
    ax.set_yticklabels([f"key {idx + 1}" for idx in range(n_key_rows)])
    ax.set_xlabel("query variable")
    ax.set_ylabel("attended variable")
    ax.set_title(title)
    for query_idx, keys in enumerate(row_key_names):
        for key_row, key_name in enumerate(keys):
            label = f"{key_name}\n{matrix[query_idx, key_row]:.2f}"
            ax.text(query_idx, key_row, label, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_vmax_values(value, expected_count):
    if value is None:
        return None
    vmax_values = [float(item) for item in value.split(",")]
    if len(vmax_values) != expected_count:
        raise ValueError(f"Expected {expected_count} --mass_vmax values, got {len(vmax_values)}")
    return vmax_values


def save_pairwise_spatial_maps(
    maps,
    query_names,
    row_key_names,
    grid_h,
    grid_w,
    output_path,
    mode="mass",
    percentile=95.0,
    vmax_values=None,
):
    n_queries = len(query_names)
    n_key_rows = max(len(keys) for keys in row_key_names)
    fig, axes = plt.subplots(
        n_key_rows,
        n_queries,
        figsize=(3.15 * n_queries, 2.65 * n_key_rows),
        constrained_layout=True,
    )
    if n_key_rows == 1 and n_queries == 1:
        axes = np.array([[axes]])
    elif n_key_rows == 1:
        axes = axes[None, :]
    elif n_queries == 1:
        axes = axes[:, None]

    for query_idx, query_name in enumerate(query_names):
        for key_row, key_name in enumerate(row_key_names[query_idx]):
            ax = axes[key_row, query_idx]
            raw_field = maps[query_idx, key_row].reshape(grid_h, grid_w)
            field, colorbar_label = transform_panel(raw_field, mode, percentile=percentile)
            panel_max = float(np.nanmax(field)) if field.size else 1.0
            flat_panel_idx = key_row * n_queries + query_idx
            if vmax_values is not None:
                panel_max = vmax_values[flat_panel_idx]
            if panel_max <= 0:
                panel_max = 1.0
            image = ax.imshow(field, cmap="magma", aspect="auto", vmin=0.0, vmax=panel_max)
            if key_row == 0:
                ax.set_title(f"query: {query_name}", fontsize=9)
            if query_idx == 0:
                ax.set_ylabel(f"key:\n{key_name}", fontsize=9)
            else:
                ax.set_ylabel(f"key:\n{key_name}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(colorbar_label, fontsize=7)
        for key_row in range(len(row_key_names[query_idx]), n_key_rows):
            axes[key_row, query_idx].axis("off")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_attention(attn, n_vars, n_patches, query_indices, row_key_indices):
    # attn shape: batch, heads, query_tokens, key_tokens.
    attn = attn.detach().cpu()[0]
    n_heads = attn.shape[0]
    attn = attn.reshape(n_heads, n_vars, n_patches, n_vars, n_patches)

    # Average over heads/query patches, then sum over key patches. This gives
    # query-variable -> key-variable mass, with each row summing to about 1.
    attention_by_query_key = attn.mean(dim=(0, 2))
    full_variable_matrix = attention_by_query_key.sum(dim=-1)
    matrix_rows = []
    spatial_rows = []
    for query_idx, keys in zip(query_indices, row_key_indices):
        matrix_rows.append(full_variable_matrix[query_idx, keys])
        spatial_rows.append(attention_by_query_key[query_idx, keys, :])
    variable_matrix = torch.stack(matrix_rows, dim=0).numpy()

    # For each query variable and key variable, show where in the key variable tokens
    # attention lands, averaged over heads and query patches.
    spatial_maps = torch.stack(spatial_rows, dim=0).numpy()
    return variable_matrix, spatial_maps


def inspect_sample(model, dataset, sample_idx, device, output_dir, args):
    sample_dir = output_dir / f"{dataset.split}_sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    batch = dataset[sample_idx]
    x = batch["x"].unsqueeze(0).to(device)
    variable_mask = batch["variable_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            x,
            variable_mask=variable_mask,
            force_mask_epsilon=args.force_mask_epsilon,
            return_attention=True,
        )

    attn = out["attention"]
    if attn is None:
        raise RuntimeError("Model did not return attention weights.")

    names = list(dataset.input_variables)
    if model.use_epsilon_mask_channel:
        names.append("epsilon_visible")

    query_indices = list(range(len(names)))
    if "epsilon_visible" in names and not args.show_epsilon_visible_query:
        query_indices.remove(names.index("epsilon_visible"))
    query_names = [names[idx] for idx in query_indices]

    key_pool = list(range(len(names)))
    if "epsilon_visible" in names and not args.show_epsilon_visible_key:
        key_pool.remove(names.index("epsilon_visible"))

    row_key_indices = []
    row_key_names = []
    for query_idx in query_indices:
        keys = list(key_pool)
        if not args.include_self_attention and query_idx in keys:
            keys.remove(query_idx)
        row_key_indices.append(keys)
        row_key_names.append([names[idx] for idx in keys])

    grid_h = model.encoder.grid_h
    grid_w = model.encoder.grid_w
    n_vars = len(names)
    n_patches = grid_h * grid_w
    variable_matrix, spatial_maps = summarize_attention(attn, n_vars, n_patches, query_indices, row_key_indices)
    n_mass_panels = len(query_names) * max(len(keys) for keys in row_key_names)
    mass_vmax_values = parse_vmax_values(args.mass_vmax, n_mass_panels)

    save_pairwise_matrix(
        variable_matrix,
        query_names,
        row_key_names,
        sample_dir / "cross_variable_attention_matrix.png",
        "Cross-variable attention",
    )
    save_pairwise_spatial_maps(
        spatial_maps,
        query_names,
        row_key_names,
        grid_h,
        grid_w,
        sample_dir / "cross_attention_spatial_maps_mass.png",
        mode="mass",
        vmax_values=mass_vmax_values,
    )
    save_pairwise_spatial_maps(
        spatial_maps,
        query_names,
        row_key_names,
        grid_h,
        grid_w,
        sample_dir / "cross_attention_spatial_maps_display.png",
        mode="percentile",
        percentile=args.display_percentile,
    )
    save_pairwise_spatial_maps(
        spatial_maps,
        query_names,
        row_key_names,
        grid_h,
        grid_w,
        sample_dir / "cross_attention_spatial_maps_max_normalized.png",
        mode="max",
    )
    save_pairwise_spatial_maps(
        spatial_maps,
        query_names,
        row_key_names,
        grid_h,
        grid_w,
        sample_dir / "cross_attention_spatial_maps_log.png",
        mode="log",
    )

    np.save(sample_dir / "cross_variable_attention_matrix.npy", variable_matrix)
    np.save(sample_dir / "cross_attention_spatial_maps.npy", spatial_maps)

    summary = {
        "sample": sample_idx,
        "split": dataset.split,
        "metadata": batch["metadata"],
        "input_variables": dataset.input_variables,
        "attention_query_variables": query_names,
        "attention_key_variables_by_query": row_key_names,
        "attention_shape": list(attn.shape),
        "patch_grid": [grid_h, grid_w],
        "epsilon_masked_for_encoder": bool(out["epsilon_masked"][0].detach().cpu()),
        "epsilon_available": bool(out["epsilon_available"][0].detach().cpu()),
        "force_mask_epsilon": args.force_mask_epsilon,
    }
    with (sample_dir / "cross_attention_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote cross-attention visualizations to {sample_dir}", flush=True)


def main():
    args = parse_args()
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
