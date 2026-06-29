import argparse
import sys
from pathlib import Path


DEFAULT_EXPERIMENT_DIR = "/content/drive/MyDrive/ML_turbulence/experiment"


LINKS = {
    "data/kh_holmboe_dataset_keep_epsilon.h5": "kh_holmboe_dataset_keep_epsilon.h5",
    "data/test_dataset_keep_epsilon.h5": "test_dataset_keep_epsilon.h5",
    "data/RM_summary_table.csv": "RM_summary_table.csv",
    "data/test_RM_summary_table.csv": "test_RM_summary_table.csv",
    "checkpoints/multihead_epsilon_flow_model.pt": "model_1/multihead_epsilon_flow_model.pt",
    "checkpoints/multihead_epsilon_flow_finetuned_external.pt": "model_1/multihead_epsilon_flow_finetuned_external.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Colab symlinks from this repo to ML_turbulence/experiment in Google Drive."
    )
    parser.add_argument(
        "--experiment_dir",
        default=DEFAULT_EXPERIMENT_DIR,
        help="Google Drive experiment directory mounted in Colab.",
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing non-symlink files at destination paths.",
    )
    return parser.parse_args()


def create_link(src, dst, force=False):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        if not force:
            print(f"exists, skipped: {dst}")
            return True
        if dst.is_dir():
            print(f"refusing to replace directory: {dst}", file=sys.stderr)
            return False
        dst.unlink()

    dst.symlink_to(src)
    print(f"linked: {dst} -> {src}")
    return True


def main():
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()

    if not experiment_dir.exists():
        print(f"experiment directory does not exist: {experiment_dir}", file=sys.stderr)
        print("Mount Google Drive first with: from google.colab import drive; drive.mount('/content/drive')", file=sys.stderr)
        return 1

    missing = []
    for rel_dst, rel_src in LINKS.items():
        src = experiment_dir / rel_src
        if not src.exists():
            missing.append(src)

    if missing:
        print("missing expected Drive files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    ok = True
    for rel_dst, rel_src in LINKS.items():
        src = experiment_dir / rel_src
        dst = repo_root / rel_dst
        ok = create_link(src, dst, force=args.force) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
