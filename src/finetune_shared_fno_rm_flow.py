import argparse
import csv
from pathlib import Path

import torch

from train_shared_fno_rm_flow import (
    DEFAULT_VARIABLES,
    REPO_ROOT,
    SharedFNOFlowModel,
    append_metrics_row,
    first_existing_path,
    make_loader,
    parse_pair,
    plot_loss_curves,
    run_one_epoch,
    save_checkpoint,
    set_seed,
)


DEFAULT_FINETUNE_H5 = first_existing_path([
    REPO_ROOT / "data" / "test_dataset_keep_epsilon.h5",
    REPO_ROOT / "test_data" / "test_dataset_keep_epsilon.h5",
    REPO_ROOT / "experiment" / "test_data" / "test_dataset_keep_epsilon.h5",
])


def default_finetune_label_csv(label_csv):
    if label_csv is not None:
        return label_csv
    for candidate in [
        REPO_ROOT / "data" / "test_RM_summary_table.csv",
        REPO_ROOT / "test_data" / "test_RM_summary_table.csv",
        REPO_ROOT / "experiment" / "test_data" / "test_RM_summary_table.csv",
        REPO_ROOT / "data" / "RM_summary_table.csv",
        REPO_ROOT / "RM_summary_table.csv",
        REPO_ROOT / "experiment" / "RM_summary_table.csv",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


def checkpoint_args(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("args", {})
    return {}


def model_arg(args, saved_args, name):
    return saved_args.get(name, getattr(args, name))


def set_heads_only(model, heads_only):
    for parameter in model.parameters():
        parameter.requires_grad = not heads_only

    if heads_only:
        for parameter in model.rm_head.parameters():
            parameter.requires_grad = True


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def init_finetune_metrics_csv(path):
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "stage",
        "trainable_parameters",
        "train_total",
        "train_rm",
        "train_flow",
        "train_buoyancy_mse",
        "train_rm_mae",
        "train_spatial_mask_frac",
        "train_var_visible_frac",
        "val_total",
        "val_rm",
        "val_flow",
        "val_buoyancy_mse",
        "val_rm_mae",
        "val_spatial_mask_frac",
        "val_var_visible_frac",
        "best_val_rm",
        "checkpoint_saved",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return path, fieldnames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=str(DEFAULT_FINETUNE_H5))
    parser.add_argument("--label_csv", default=None)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--heads_only_epochs", type=int, default=5)
    parser.add_argument("--input_variables", default=DEFAULT_VARIABLES)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_flow", type=float, default=0.2)
    parser.add_argument("--spatial_mask_prob", type=float, default=0.25)
    parser.add_argument("--variable_drop_prob", type=float, default=0.25)
    parser.add_argument("--flow_masked_only", action="store_true")
    parser.add_argument("--flow_all_pixels", dest="flow_masked_only", action="store_false")
    parser.set_defaults(flow_masked_only=True)
    parser.add_argument("--patch_size", type=parse_pair, default=(10, 10))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--vit_depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--fno_modes", type=parse_pair, default=(12, 8))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rm_dropout", type=float, default=0.1)
    parser.add_argument("--flow_base_channels", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default="shared_fno_rm_flow_finetuned.pt")
    parser.add_argument("--metrics_csv", default="shared_fno_rm_flow_finetune_loss_history.csv")
    parser.add_argument("--loss_plot", default="shared_fno_rm_flow_finetune_loss_curves.png")
    return parser.parse_args()


def main():
    args = parse_args()
    args.label_csv = default_finetune_label_csv(args.label_csv)
    set_seed(args.seed)
    device = torch.device(args.device)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    metrics_csv, metric_fields = init_finetune_metrics_csv(args.metrics_csv)

    train_loader = make_loader(
        args.h5,
        "train",
        args.label_csv,
        args.batch_size,
        shuffle=True,
        input_variables=args.input_variables,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        args.h5,
        "val",
        args.label_csv,
        args.batch_size,
        shuffle=False,
        input_variables=args.input_variables,
        num_workers=args.num_workers,
    )

    input_variables = train_loader.dataset.input_variables
    img_size = tuple(train_loader.dataset.x_data.shape[-2:])
    print(f"Fine-tune H5: {args.h5}", flush=True)
    print(f"Fine-tune label CSV: {args.label_csv}", flush=True)
    print(f"Using input variables: {input_variables}", flush=True)
    print(f"Using image size: {img_size}", flush=True)

    checkpoint = torch.load(args.init_checkpoint, map_location=device)
    saved_model_args = checkpoint_args(checkpoint)

    model = SharedFNOFlowModel(
        img_size=img_size,
        patch_size=tuple(model_arg(args, saved_model_args, "patch_size")),
        n_vars=len(input_variables),
        dim=int(model_arg(args, saved_model_args, "dim")),
        vit_depth=int(model_arg(args, saved_model_args, "vit_depth")),
        heads=int(model_arg(args, saved_model_args, "heads")),
        latent_channels=int(model_arg(args, saved_model_args, "latent_channels")),
        fno_modes=tuple(model_arg(args, saved_model_args, "fno_modes")),
        dropout=float(model_arg(args, saved_model_args, "dropout")),
        rm_dropout=float(model_arg(args, saved_model_args, "rm_dropout")),
        flow_base_channels=int(model_arg(args, saved_model_args, "flow_base_channels")),
    ).to(device)
    model.load_state_dict(checkpoint_state_dict(checkpoint))
    print(f"Loaded initial checkpoint: {args.init_checkpoint}", flush=True)
    print(f"Compact latent feature per sample: {model.latent_shape}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rm = float("inf")
    current_heads_only = None
    for epoch in range(1, args.epochs + 1):
        heads_only = epoch <= args.heads_only_epochs
        if heads_only != current_heads_only:
            set_heads_only(model, heads_only=heads_only)
            current_heads_only = heads_only
            stage = "rm_head_only" if heads_only else "all_parameters"
            print(
                f"Fine-tune stage: {stage}; trainable parameters: {count_trainable_parameters(model)}",
                flush=True,
            )

        train_metrics = run_one_epoch(model, train_loader, optimizer, device, args)
        with torch.no_grad():
            val_metrics = run_one_epoch(model, val_loader, None, device, args)

        checkpoint_saved = False
        if val_metrics["rm"] < best_val_rm:
            best_val_rm = val_metrics["rm"]
            save_checkpoint(
                args.save,
                model,
                args,
                input_variables,
                img_size,
                best_val_rm,
                epoch,
            )
            checkpoint_saved = True

        print(
            f"epoch {epoch:03d} | "
            f"train total {train_metrics['total']:.6f} "
            f"rm {train_metrics['rm']:.6f} "
            f"flow {train_metrics['flow']:.6f} "
            f"buoy_mse {train_metrics['buoyancy_mse']:.6f} "
            f"rm_mae {train_metrics['rm_mae']:.6f} | "
            f"val total {val_metrics['total']:.6f} "
            f"rm {val_metrics['rm']:.6f} "
            f"flow {val_metrics['flow']:.6f} "
            f"buoy_mse {val_metrics['buoyancy_mse']:.6f} "
            f"rm_mae {val_metrics['rm_mae']:.6f}",
            flush=True,
        )
        if checkpoint_saved:
            print(f"saved {args.save}", flush=True)

        row = {
            "epoch": epoch,
            "stage": "rm_head_only" if heads_only else "all_parameters",
            "trainable_parameters": count_trainable_parameters(model),
            "best_val_rm": best_val_rm,
            "checkpoint_saved": int(checkpoint_saved),
        }
        for split, metrics in [("train", train_metrics), ("val", val_metrics)]:
            for name, value in metrics.items():
                row[f"{split}_{name}"] = value
        append_metrics_row(metrics_csv, metric_fields, row)
        plot_loss_curves(metrics_csv, args.loss_plot)

    train_loader.dataset.close()
    val_loader.dataset.close()


if __name__ == "__main__":
    main()
