import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--epsilon_input_mask_prob", type=float, default=0.5)
    parser.add_argument("--force_mask_epsilon", action="store_true")
    parser.add_argument("--input_variables", default=None)
    parser.add_argument("--output_dir", default="epsilon_flow_test_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class IndexedRangeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, start_index=0, end_index=None):
        self.dataset = dataset
        if end_index is None:
            end_index = len(dataset)
        self.indices = list(range(start_index, min(end_index, len(dataset))))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_index = self.indices[idx]
        return {"dataset_index": dataset_index, "sample": self.dataset[dataset_index]}


def collate_batch(items):
    return {
        "x": torch.stack([item["sample"]["x"] for item in items]),
        "variable_mask": torch.stack([item["sample"]["variable_mask"] for item in items]),
        "ratio": torch.stack([item["sample"]["ratio"] for item in items]),
        "has_ratio": torch.stack([item["sample"]["has_ratio"] for item in items]),
        "metadata": [item["sample"]["metadata"] for item in items],
        "dataset_index": torch.tensor([item["dataset_index"] for item in items], dtype=torch.long),
    }


def write_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def compute_metrics(rows):
    labeled = [row for row in rows if row["has_label"]]
    if not labeled:
        return {"n_labeled": 0, "mae": None, "rmse": None, "bias": None, "mse": None}
    errors = [row["error"] for row in labeled]
    squared = [error ** 2 for error in errors]
    return {
        "n_labeled": len(labeled),
        "mae": mean([abs(error) for error in errors]),
        "rmse": math.sqrt(mean(squared)),
        "bias": mean(errors),
        "mse": mean(squared),
    }


def aggregate_by_case(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)

    case_rows = []
    for case_id, case_samples in sorted(grouped.items()):
        preds = [row["pred_mean"] for row in case_samples]
        stds = [row["pred_std"] for row in case_samples]
        labeled = [row for row in case_samples if row["has_label"]]
        target = labeled[0]["target_R_M"] if labeled else None
        case_pred = mean(preds)
        error = case_pred - target if target is not None else None
        case_rows.append({
            "case_id": case_id,
            "n_samples": len(case_samples),
            "target_R_M": target,
            "pred_mean": case_pred,
            "profile_prediction_std": std(preds),
            "mean_head_uncertainty": mean(stds),
            "error": error,
            "abs_error": abs(error) if error is not None else None,
        })
    return case_rows


def plot_results(sample_rows, case_rows, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping plots.", flush=True)
        return

    output_dir = Path(output_dir)
    labeled_samples = [row for row in sample_rows if row["has_label"]]
    labeled_cases = [row for row in case_rows if row["target_R_M"] is not None]

    if labeled_samples:
        targets = [row["target_R_M"] for row in labeled_samples]
        preds = [row["pred_mean"] for row in labeled_samples]
        stds = [row["pred_std"] for row in labeled_samples]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(targets, preds, yerr=stds, fmt="o", alpha=0.75, capsize=2)
        lo = min(targets + preds)
        hi = max(targets + preds)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("target R_M")
        ax.set_ylabel("predicted R_M")
        ax.set_title("Profile-level epsilon-flow model")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "epsilon_flow_sample_pred_vs_target.png", dpi=180)
        plt.close(fig)

    if labeled_cases:
        targets = [row["target_R_M"] for row in labeled_cases]
        preds = [row["pred_mean"] for row in labeled_cases]
        stds = [row["profile_prediction_std"] for row in labeled_cases]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(targets, preds, yerr=stds, fmt="o", alpha=0.85, capsize=3)
        lo = min(targets + preds)
        hi = max(targets + preds)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("target R_M")
        ax.set_ylabel("case mean predicted R_M")
        ax.set_title("Case-level epsilon-flow model")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "epsilon_flow_case_pred_vs_target.png", dpi=180)
        plt.close(fig)


@torch.no_grad()
def run_test(args):
    args.label_csv = default_label_csv(args)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = KHHolmboeDataset(args.h5, split=args.split, label_csv=args.label_csv, input_variables=args.input_variables)
    epsilon_index = find_epsilon_index(dataset.input_variables)
    eval_dataset = IndexedRangeDataset(dataset, args.start_index, args.end_index)
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

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

    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        variable_mask = batch["variable_mask"].to(device)
        ratio = batch["ratio"].to(device)
        has_ratio = batch["has_ratio"].to(device)
        out = model(x, variable_mask=variable_mask, force_mask_epsilon=args.force_mask_epsilon)
        preds = out["preds"]
        pred_mean = preds.mean(dim=1)
        pred_std = preds.std(dim=1, unbiased=False)

        for i, metadata in enumerate(batch["metadata"]):
            sample_has_label = bool(has_ratio[i].detach().cpu())
            target = float(ratio[i, 0].detach().cpu()) if sample_has_label else None
            mean_value = float(pred_mean[i, 0].detach().cpu())
            std_value = float(pred_std[i, 0].detach().cpu())
            error = mean_value - target if sample_has_label else None
            row = {
                "dataset_index": int(batch["dataset_index"][i].item()),
                "split": args.split,
                "case_id": metadata.get("case_id"),
                "plane": metadata.get("plane"),
                "axis_index": metadata.get("axis_index"),
                "target_R_M": target,
                "has_label": sample_has_label,
                "pred_mean": mean_value,
                "pred_std": std_value,
                "epsilon_available": bool(out["epsilon_available"][i].detach().cpu()),
                "epsilon_masked_for_encoder": bool(out["epsilon_masked"][i].detach().cpu()),
                "error": error,
                "abs_error": abs(error) if error is not None else None,
            }
            for head_idx, value in enumerate(preds[i, :, 0].detach().cpu().tolist()):
                row[f"head_{head_idx}"] = float(value)
            rows.append(row)

    case_rows = aggregate_by_case(rows)
    metrics = compute_metrics(rows)
    sample_fields = [
        "dataset_index", "split", "case_id", "plane", "axis_index",
        "target_R_M", "has_label", "pred_mean", "pred_std",
        "epsilon_available", "epsilon_masked_for_encoder",
        "error", "abs_error",
    ] + [f"head_{idx}" for idx in range(args.n_heads)]
    case_fields = [
        "case_id", "n_samples", "target_R_M", "pred_mean",
        "profile_prediction_std", "mean_head_uncertainty", "error", "abs_error",
    ]
    write_rows(output_dir / "epsilon_flow_test_predictions.csv", sample_fields, rows)
    write_rows(output_dir / "epsilon_flow_test_case_summary.csv", case_fields, case_rows)
    with (output_dir / "epsilon_flow_test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    plot_results(rows, case_rows, output_dir)
    print(f"wrote epsilon-flow test results to {output_dir}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    dataset.close()


def main():
    args = parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
