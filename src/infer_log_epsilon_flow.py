import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import KHHolmboeDataset
from train_multihead_epsilon_flow import MultiHeadEpsilonFlowModel, default_label_csv, find_epsilon_index


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
    parser.add_argument("--use_epsilon_mask_channel", action="store_true")
    parser.add_argument("--no_epsilon_mask_channel", dest="use_epsilon_mask_channel", action="store_false")
    parser.set_defaults(use_epsilon_mask_channel=True)
    parser.add_argument("--input_variables", default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="log_epsilon_inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def tensor_to_image(tensor):
    return tensor.detach().cpu().squeeze().numpy()


def plot_generated_log_epsilon(true_eps, generated_eps, available, output_path):
    generated = tensor_to_image(generated_eps[0, 0])
    if bool(available[0].detach().cpu()):
        true = tensor_to_image(true_eps[0, 0])
        error = generated - true
        vmin = min(true.min(), generated.min())
        vmax = max(true.max(), generated.max())
        err_abs = max(abs(error.min()), abs(error.max()))
        panels = [
            ("true log_epsilon", true, "magma", vmin, vmax),
            ("generated log_epsilon", generated, "magma", vmin, vmax),
            ("generated - true", error, "seismic", -err_abs, err_abs),
        ]
    else:
        panels = [("generated log_epsilon", generated, "magma", generated.min(), generated.max())]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.3), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, field, cmap, lo, hi) in zip(axes, panels):
        image = ax.imshow(field, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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


def run_sample(model, dataset, sample_idx, device, output_dir, args):
    sample_dir = output_dir / f"{dataset.split}_sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    batch = dataset[sample_idx]
    x = batch["x"].unsqueeze(0).to(device)
    variable_mask = batch["variable_mask"].unsqueeze(0).to(device)
    condition = batch["condition"].unsqueeze(0).to(device)
    true_eps = x[:, model.epsilon_index:model.epsilon_index + 1]

    with torch.no_grad():
        out = model(x, variable_mask=variable_mask, force_mask_epsilon=True)
        generated_eps = model.generate_log_epsilon(out["feature"], condition, steps=args.steps, clamp=(-1.0, 1.0))

    available = variable_mask[:, model.epsilon_index]
    plot_input_variables(x, dataset.input_variables, sample_dir / "input_variables.png")
    plot_generated_log_epsilon(true_eps, generated_eps, available, sample_dir / "generated_log_epsilon_vs_true.png")
    torch.save(out["feature"].detach().cpu(), sample_dir / "latent_feature.pt")
    torch.save(generated_eps.detach().cpu(), sample_dir / "log_epsilon_generated.pt")
    torch.save(true_eps.detach().cpu(), sample_dir / "log_epsilon_true_or_zero.pt")

    preds = out["preds"]
    summary = {
        "sample": sample_idx,
        "split": dataset.split,
        "metadata": batch["metadata"],
        "input_variables": dataset.input_variables,
        "epsilon_available": bool(available[0].detach().cpu()),
        "epsilon_was_forced_masked": True,
        "steps": args.steps,
        "ratio": batch["ratio"].tolist(),
        "has_ratio": bool(batch["has_ratio"]),
        "pred_R_M_mean_masked_input": float(preds[0, :, 0].mean().detach().cpu()),
        "pred_R_M_std_masked_input": float(preds[0, :, 0].std(unbiased=False).detach().cpu()),
    }
    if bool(available[0].detach().cpu()):
        summary["generated_log_epsilon_mse"] = F.mse_loss(generated_eps, true_eps).item()
        summary["generated_log_epsilon_mae"] = torch.abs(generated_eps - true_eps).mean().item()
    with (sample_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote log-epsilon inference to {sample_dir}", flush=True)


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
        use_epsilon_mask_channel=args.use_epsilon_mask_channel,
        epsilon_input_mask_prob=args.epsilon_input_mask_prob,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    last_sample = min(args.sample + args.n_samples, len(dataset))
    for sample_idx in range(args.sample, last_sample):
        run_sample(model, dataset, sample_idx, device, output_dir, args)
    dataset.close()


if __name__ == "__main__":
    main()
