import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def slug_float(value):
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def optional_path_arg(cmd, flag, value):
    if value is not None and str(value).strip():
        cmd.extend([flag, str(value)])


def run_command(cmd, cwd, log_path, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        log_path.write_text(printable + "\n")
        return 0

    start = time.time()
    with log_path.open("w") as log_file:
        log_file.write(printable + "\n\n")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        elapsed = time.time() - start
        log_file.write(f"\nreturn_code: {return_code}\nelapsed_seconds: {elapsed:.2f}\n")
    return return_code


def load_metrics(metrics_path):
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r") as f:
        return json.load(f)


def write_summary(summary_csv, rows):
    if not rows:
        return
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "status",
        "failed_stage",
        "train_fraction",
        "train_max_samples",
        "input_noise_std",
        "noise_variables",
        "seed",
        "used_finetune",
        "finetune_train_fraction",
        "finetune_input_noise_std",
        "checkpoint",
        "test_output_dir",
        "n_labeled",
        "mae",
        "rmse",
        "bias",
        "mse",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep train-set downsampling and train-time input noise, then summarize test metrics."
    )
    parser.add_argument("--main_h5", default="data/kh_holmboe_dataset_keep_epsilon.h5")
    parser.add_argument("--main_label_csv", default="data/RM_summary_table.csv")
    parser.add_argument("--test_h5", default="data/test_dataset_keep_epsilon.h5")
    parser.add_argument("--test_label_csv", default="data/test_RM_summary_table.csv")
    parser.add_argument("--input_variables", default="buoyancy,reduced_shear,log_epsilon")
    parser.add_argument("--fractions", default="1.0,0.5,0.25,0.1")
    parser.add_argument("--noise_stds", default="0.0,0.01,0.05,0.1")
    parser.add_argument("--noise_variables", default="all")
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_input_mask_prob", type=float, default=0.5)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_root", default="outputs/downsample_noise_sweep")
    parser.add_argument("--checkpoint_root", default="checkpoints/downsample_noise_sweep")
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--run_finetune", action="store_true")
    parser.add_argument("--finetune_epochs", type=int, default=30)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--finetune_lambda_recon", type=float, default=0.2)
    parser.add_argument("--finetune_train_fraction", type=float, default=1.0)
    parser.add_argument("--finetune_input_noise_std", type=float, default=0.0)
    parser.add_argument("--perturb_finetune_like_main", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_force_mask_epsilon", dest="force_mask_epsilon", action="store_false")
    parser.set_defaults(force_mask_epsilon=True)
    return parser.parse_args()


def build_train_command(args, checkpoint_path, metrics_csv, loss_plot, train_fraction, noise_std, init_checkpoint=None):
    cmd = [
        sys.executable,
        "src/train_multihead_epsilon_flow.py",
        "--h5",
        args.main_h5 if init_checkpoint is None else args.test_h5,
        "--input_variables",
        args.input_variables,
        "--epochs",
        str(args.epochs if init_checkpoint is None else args.finetune_epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr if init_checkpoint is None else args.finetune_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--lambda_recon",
        str(args.lambda_recon if init_checkpoint is None else args.finetune_lambda_recon),
        "--lambda_epsilon",
        str(args.lambda_epsilon),
        "--epsilon_input_mask_prob",
        str(args.epsilon_input_mask_prob),
        "--eval_force_mask_epsilon",
        "--mask_prob",
        str(args.mask_prob),
        "--train_fraction",
        str(train_fraction),
        "--downsample_seed",
        str(args.seed),
        "--input_noise_std",
        str(noise_std),
        "--noise_variables",
        args.noise_variables,
        "--seed",
        str(args.seed),
        "--save",
        str(checkpoint_path),
        "--metrics_csv",
        str(metrics_csv),
        "--loss_plot",
        str(loss_plot),
    ]
    if args.train_max_samples is not None and init_checkpoint is None:
        cmd.extend(["--train_max_samples", str(args.train_max_samples)])
    if init_checkpoint is not None:
        cmd.extend(["--init_checkpoint", str(init_checkpoint)])
        optional_path_arg(cmd, "--label_csv", args.test_label_csv)
    else:
        optional_path_arg(cmd, "--label_csv", args.main_label_csv)
    optional_path_arg(cmd, "--device", args.device)
    return cmd


def build_test_command(args, checkpoint_path, output_dir):
    cmd = [
        sys.executable,
        "src/test_multihead_epsilon_flow.py",
        "--h5",
        args.test_h5,
        "--label_csv",
        args.test_label_csv,
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "test",
        "--batch_size",
        str(args.test_batch_size),
        "--input_variables",
        args.input_variables,
        "--output_dir",
        str(output_dir),
    ]
    if args.force_mask_epsilon:
        cmd.append("--force_mask_epsilon")
    optional_path_arg(cmd, "--device", args.device)
    return cmd


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary.csv"

    rows = []
    fractions = parse_float_list(args.fractions)
    noise_stds = parse_float_list(args.noise_stds)

    for fraction in fractions:
        for noise_std in noise_stds:
            run_id = f"frac_{slug_float(fraction)}__noise_{slug_float(noise_std)}__seed_{args.seed}"
            run_output = output_root / run_id
            run_checkpoints = checkpoint_root / run_id
            train_checkpoint = run_checkpoints / "multihead_epsilon_flow_model.pt"
            train_metrics = run_output / "train_loss_history.csv"
            train_loss_plot = run_output / "train_loss_curves.png"
            train_log = run_output / "train.log"
            test_checkpoint = train_checkpoint
            used_finetune = False
            status = "ok"
            failed_stage = ""
            summary_finetune_fraction = ""
            summary_finetune_noise = ""

            metrics_path = run_output / "test" / "epsilon_flow_test_metrics.json"
            if args.skip_existing and metrics_path.exists():
                print(f"Skipping existing run: {run_id}", flush=True)
                used_finetune = args.run_finetune
                if args.run_finetune:
                    summary_finetune_fraction = fraction if args.perturb_finetune_like_main else args.finetune_train_fraction
                    summary_finetune_noise = noise_std if args.perturb_finetune_like_main else args.finetune_input_noise_std
                    test_checkpoint = run_checkpoints / "multihead_epsilon_flow_finetuned_external.pt"
            else:
                train_cmd = build_train_command(
                    args,
                    train_checkpoint,
                    train_metrics,
                    train_loss_plot,
                    fraction,
                    noise_std,
                )
                return_code = run_command(train_cmd, repo_root, train_log, dry_run=args.dry_run)
                if return_code != 0:
                    status = "failed"
                    failed_stage = "train"

                if status == "ok" and args.run_finetune:
                    used_finetune = True
                    finetune_checkpoint = run_checkpoints / "multihead_epsilon_flow_finetuned_external.pt"
                    finetune_metrics = run_output / "finetune_loss_history.csv"
                    finetune_loss_plot = run_output / "finetune_loss_curves.png"
                    finetune_log = run_output / "finetune.log"
                    finetune_fraction = fraction if args.perturb_finetune_like_main else args.finetune_train_fraction
                    finetune_noise = noise_std if args.perturb_finetune_like_main else args.finetune_input_noise_std
                    summary_finetune_fraction = finetune_fraction
                    summary_finetune_noise = finetune_noise
                    finetune_cmd = build_train_command(
                        args,
                        finetune_checkpoint,
                        finetune_metrics,
                        finetune_loss_plot,
                        finetune_fraction,
                        finetune_noise,
                        init_checkpoint=train_checkpoint,
                    )
                    return_code = run_command(finetune_cmd, repo_root, finetune_log, dry_run=args.dry_run)
                    if return_code != 0:
                        status = "failed"
                        failed_stage = "finetune"
                    else:
                        test_checkpoint = finetune_checkpoint

                if status == "ok":
                    test_output = run_output / "test"
                    test_log = run_output / "test.log"
                    test_cmd = build_test_command(args, test_checkpoint, test_output)
                    return_code = run_command(test_cmd, repo_root, test_log, dry_run=args.dry_run)
                    if return_code != 0:
                        status = "failed"
                        failed_stage = "test"

            if not args.dry_run:
                metrics = load_metrics(metrics_path)
            else:
                metrics = {}

            rows.append({
                "run_id": run_id,
                "status": status,
                "failed_stage": failed_stage,
                "train_fraction": fraction,
                "train_max_samples": args.train_max_samples,
                "input_noise_std": noise_std,
                "noise_variables": args.noise_variables,
                "seed": args.seed,
                "used_finetune": used_finetune,
                "finetune_train_fraction": summary_finetune_fraction,
                "finetune_input_noise_std": summary_finetune_noise,
                "checkpoint": test_checkpoint,
                "test_output_dir": run_output / "test",
                "n_labeled": metrics.get("n_labeled"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "bias": metrics.get("bias"),
                "mse": metrics.get("mse"),
            })
            write_summary(summary_csv, rows)

            if status != "ok" and not args.continue_on_error:
                raise SystemExit(f"Run {run_id} failed during {failed_stage}. See logs in {run_output}.")

    print(f"\nWrote sweep summary to {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
