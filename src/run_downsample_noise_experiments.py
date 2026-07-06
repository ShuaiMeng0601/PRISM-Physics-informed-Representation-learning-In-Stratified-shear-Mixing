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
        "checkpoint",
        "split",
        "input_noise_std",
        "noise_variables",
        "spatial_downsample_factor",
        "perturb_seed",
        "force_mask_epsilon",
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
        description="Sweep test-time input noise and spatial downsampling for a trained checkpoint."
    )
    parser.add_argument("--h5", default="data/test_dataset_keep_epsilon.h5")
    parser.add_argument("--label_csv", default="data/test_RM_summary_table.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input_variables", default="buoyancy,reduced_shear,log_epsilon")
    parser.add_argument("--noise_stds", default="0.0,0.01,0.05,0.1")
    parser.add_argument("--downsample_factors", default="1.0,2.0,4.0,8.0,16.0,32.0")
    parser.add_argument("--noise_variables", default="all")
    parser.add_argument("--perturb_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--epsilon_input_mask_prob", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_root", default="outputs/test_input_robustness_sweep")
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_mask_epsilon", action="store_true")
    return parser.parse_args()


def build_test_command(args, noise_std, downsample_factor, output_dir):
    cmd = [
        sys.executable,
        "src/test_multihead_epsilon_flow.py",
        "--h5",
        args.h5,
        "--label_csv",
        args.label_csv,
        "--checkpoint",
        args.checkpoint,
        "--split",
        args.split,
        "--batch_size",
        str(args.batch_size),
        "--start_index",
        str(args.start_index),
        "--input_variables",
        args.input_variables,
        "--n_heads",
        str(args.n_heads),
        "--dropout",
        str(args.dropout),
        "--mask_prob",
        str(args.mask_prob),
        "--epsilon_input_mask_prob",
        str(args.epsilon_input_mask_prob),
        "--input_noise_std",
        str(noise_std),
        "--noise_variables",
        args.noise_variables,
        "--spatial_downsample_factor",
        str(downsample_factor),
        "--perturb_seed",
        str(args.perturb_seed),
        "--output_dir",
        str(output_dir),
    ]
    if args.end_index is not None:
        cmd.extend(["--end_index", str(args.end_index)])
    if args.force_mask_epsilon:
        cmd.append("--force_mask_epsilon")
    optional_path_arg(cmd, "--device", args.device)
    return cmd


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary.csv"

    rows = []
    noise_stds = parse_float_list(args.noise_stds)
    downsample_factors = parse_float_list(args.downsample_factors)

    for downsample_factor in downsample_factors:
        for noise_std in noise_stds:
            run_id = (
                f"downsample_{slug_float(downsample_factor)}"
                f"__noise_{slug_float(noise_std)}"
                f"__seed_{args.perturb_seed}"
            )
            output_dir = output_root / run_id
            log_path = output_dir / "test.log"
            metrics_path = output_dir / "epsilon_flow_test_metrics.json"
            status = "ok"
            failed_stage = ""

            if args.skip_existing and metrics_path.exists():
                print(f"Skipping existing run: {run_id}", flush=True)
            else:
                test_cmd = build_test_command(args, noise_std, downsample_factor, output_dir)
                return_code = run_command(test_cmd, repo_root, log_path, dry_run=args.dry_run)
                if return_code != 0:
                    status = "failed"
                    failed_stage = "test"

            metrics = {} if args.dry_run else load_metrics(metrics_path)
            rows.append({
                "run_id": run_id,
                "status": status,
                "failed_stage": failed_stage,
                "checkpoint": args.checkpoint,
                "split": args.split,
                "input_noise_std": noise_std,
                "noise_variables": args.noise_variables,
                "spatial_downsample_factor": downsample_factor,
                "perturb_seed": args.perturb_seed,
                "force_mask_epsilon": args.force_mask_epsilon,
                "test_output_dir": output_dir,
                "n_labeled": metrics.get("n_labeled"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "bias": metrics.get("bias"),
                "mse": metrics.get("mse"),
            })
            write_summary(summary_csv, rows)

            if status != "ok" and not args.continue_on_error:
                raise SystemExit(f"Run {run_id} failed during {failed_stage}. See logs in {output_dir}.")

    print(f"\nWrote test-time robustness summary to {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
