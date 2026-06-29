import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import KHHolmboeDataset
from test_multihead_epsilon_flow import resolve_variable_indices, spatial_downsample_upsample


DEFAULT_CASES = "1:0,1:0.05,4:0,8:0,8:0.1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize representative test-time input perturbations used in the "
            "downsample/noise robustness sweep."
        )
    )
    parser.add_argument("--h5", default="data/test_dataset_keep_epsilon.h5")
    parser.add_argument("--label_csv", default="data/test_RM_summary_table.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--input_variables", default="buoyancy,reduced_shear,log_epsilon")
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
        help=(
            "Comma-separated downsample:noise settings. "
            "Default is '1:0,1:0.05,4:0,8:0,8:0.1'."
        ),
    )
    parser.add_argument("--noise_variables", default="all")
    parser.add_argument("--sample_indices", default=None, help="Comma-separated dataset indices to visualize.")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="outputs/test_input_perturbation_visualizations")
    return parser.parse_args()


def parse_cases(value):
    cases = []
    for item in value.split(","):
        raw = item.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Expected case format downsample:noise, got {raw!r}")
        downsample, noise = raw.split(":", 1)
        cases.append((float(downsample), float(noise)))
    if not cases:
        raise ValueError("--cases must contain at least one downsample:noise setting")
    return cases


def parse_sample_indices(value):
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def slug_float(value):
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def choose_representative_indices(dataset, n_samples):
    """Pick samples across case IDs, then fill with evenly spaced profiles."""
    selected = []
    seen_cases = set()
    for idx in range(len(dataset)):
        case_id = dataset.metadata[idx].get("case_id", "")
        if case_id not in seen_cases:
            selected.append(idx)
            seen_cases.add(case_id)
        if len(selected) >= n_samples:
            return selected

    if len(selected) < n_samples:
        evenly_spaced = np.linspace(0, len(dataset) - 1, n_samples, dtype=int).tolist()
        for idx in evenly_spaced:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_samples:
                break

    return selected[:n_samples]


def apply_input_perturbation(x, variable_mask, downsample_factor, noise_std, noise_indices, seed):
    perturbed = spatial_downsample_upsample(x, downsample_factor)
    if noise_std <= 0 or not noise_indices:
        return perturbed

    generator = torch.Generator(device=perturbed.device)
    generator.manual_seed(seed)
    perturbed = perturbed.clone()
    for channel_index in noise_indices:
        if not bool(variable_mask[0, channel_index]):
            continue
        noise = torch.randn(
            perturbed[:, channel_index].shape,
            dtype=perturbed.dtype,
            device=perturbed.device,
            generator=generator,
        )
        perturbed[:, channel_index] = perturbed[:, channel_index] + noise * noise_std
    return perturbed


def tensor_to_array(tensor):
    return tensor.detach().cpu().squeeze().numpy()


def robust_limits(fields):
    stacked = np.concatenate([field.reshape(-1) for field in fields])
    vmin, vmax = np.nanpercentile(stacked, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin = float(np.nanmin(stacked))
        vmax = float(np.nanmax(stacked))
    if abs(vmax - vmin) < 1e-12:
        pad = max(1e-6, abs(vmax) * 0.01)
        vmin -= pad
        vmax += pad
    return float(vmin), float(vmax)


def variable_cmap(name):
    lower = name.lower()
    if "epsilon" in lower:
        return "magma"
    if "shear" in lower or lower in {"u", "v", "w"}:
        return "RdBu_r"
    if "buoy" in lower or lower in {"th", "theta"}:
        return "coolwarm"
    return "viridis"


def plot_sample_grid(sample, sample_idx, dataset, cases, noise_indices, output_dir, seed):
    x = sample["x"].unsqueeze(0)
    variable_mask = sample["variable_mask"].unsqueeze(0)
    variables = dataset.input_variables

    perturbed_inputs = []
    for case_idx, (downsample_factor, noise_std) in enumerate(cases):
        case_seed = seed + sample_idx * 1009 + case_idx * 9176
        perturbed = apply_input_perturbation(
            x,
            variable_mask,
            downsample_factor,
            noise_std,
            noise_indices,
            case_seed,
        )
        perturbed_inputs.append(perturbed)

    n_rows = len(cases)
    n_cols = len(variables)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for col, variable in enumerate(variables):
        fields = [tensor_to_array(perturbed[0, col]) for perturbed in perturbed_inputs]
        vmin, vmax = robust_limits(fields)
        for row, ((downsample_factor, noise_std), field) in enumerate(zip(cases, fields)):
            ax = axes[row, col]
            image = ax.imshow(
                field,
                cmap=variable_cmap(variable),
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            if row == 0:
                ax.set_title(variable)
            if col == 0:
                ax.set_ylabel(f"d={downsample_factor:g}\nnoise={noise_std:g}")
            ax.set_xlabel("x / time")
            if col == n_cols - 1:
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    metadata = sample["metadata"]
    fig.suptitle(
        (
            f"Test input perturbations | sample={sample_idx}, "
            f"case={metadata.get('case_id')}, plane={metadata.get('plane')}, "
            f"axis={metadata.get('axis_index')}"
        ),
        fontsize=12,
    )
    output_path = output_dir / f"sample_{sample_idx:04d}_perturbation_grid.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_summary(output_dir, rows):
    json_path = output_dir / "visualization_summary.json"
    csv_path = output_dir / "visualization_summary.csv"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = parse_cases(args.cases)
    explicit_indices = parse_sample_indices(args.sample_indices)

    label_csv = args.label_csv
    if label_csv is not None and str(label_csv).strip() and not Path(label_csv).exists():
        print(f"label CSV not found, continuing without labels: {label_csv}", flush=True)
        label_csv = None

    dataset = KHHolmboeDataset(
        args.h5,
        split=args.split,
        label_csv=label_csv,
        input_variables=args.input_variables,
    )
    noise_indices = resolve_variable_indices(dataset.input_variables, args.noise_variables)
    sample_indices = explicit_indices or choose_representative_indices(dataset, args.n_samples)

    rows = []
    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= len(dataset):
            raise IndexError(f"sample index {sample_idx} is outside dataset length {len(dataset)}")
        sample = dataset[sample_idx]
        output_path = plot_sample_grid(sample, sample_idx, dataset, cases, noise_indices, output_dir, args.seed)
        row = {
            "sample_index": sample_idx,
            "case_id": sample["metadata"].get("case_id"),
            "plane": sample["metadata"].get("plane"),
            "axis_index": sample["metadata"].get("axis_index"),
            "input_variables": ",".join(dataset.input_variables),
            "noise_variables": args.noise_variables,
            "perturbation_cases": ",".join(
                f"d{slug_float(downsample)}_noise{slug_float(noise)}"
                for downsample, noise in cases
            ),
            "figure": str(output_path),
        }
        rows.append(row)
        print(f"wrote {output_path}", flush=True)

    write_summary(output_dir, rows)
    dataset.close()
    print(f"\nWrote perturbation visualizations to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
