"""Shared helpers for building KH/Holmboe HDF5 datasets from mean/movie files.

Default changes relative to the original builder:
  - 100 xy profiles + 50 zy profiles = 150 clean samples per time window.
  - Multiple 100-frame windows, with every window starting at or after the
    first local peak of integrated urms.
  - Optional train-only augmentation using downsample/upsample and Gaussian
    noise on normalized fields. By default, augmented samples are about 30%
    of the final train split, so clean samples remain the majority.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


VARIABLES = ["u", "v", "w", "th", "epsilon"]
NY_NEW = 501
NT_NEW = 100
EPSILON_FLOOR = 1e-12
TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_csv_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one number")
    return values


def parse_csv_strings(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an expanded KH/Holmboe HDF5 dataset from raw mean.h5/movie.h5 "
            "files using multiple post-peak windows and train-only augmentation."
        )
    )
    parser.add_argument("--root", default="/Volumes/LaCie/Pr_1_KH_Holm")
    parser.add_argument("--out_file", default=None)
    parser.add_argument("--reference_gyf_file", default=None)
    parser.add_argument("--n_x_profiles", type=int, default=100)
    parser.add_argument("--n_z_profiles", type=int, default=50)
    parser.add_argument(
        "--time_window_offsets",
        default="0,25,50",
        help=(
            "Comma-separated frame offsets after the first local urms peak. "
            "Each value starts a 100-frame window at peak_index + offset."
        ),
    )
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--random_seed", type=int, default=13)
    parser.add_argument("--skip_cases", default="Ri016_a05_Re1000")
    parser.add_argument(
        "--test_cases",
        default="KH:Ri020_a05_Re1000,Holmboe:Ri016_a05_Re2000",
        help="Comma-separated case IDs to hold out entirely as test cases.",
    )
    parser.add_argument(
        "--noise_stds",
        default="0,0.01,0.03",
        help="Train augmentation noise stds applied after normalization.",
    )
    parser.add_argument(
        "--downsample_factors",
        default="1,2,4",
        help="Train augmentation factors. factor=1 means no downsample.",
    )
    parser.add_argument(
        "--augmentation_fraction",
        type=float,
        default=0.3,
        help=(
            "Target fraction of the final train split that should be augmented. "
            "For example, 0.3 gives approximately 70%% clean / 30%% augmented."
        ),
    )
    parser.add_argument("--augmentation_seed", type=int, default=13)
    parser.add_argument("--no_train_augmentation", action="store_true")
    parser.add_argument("--gzip_level", type=int, default=4)
    parser.add_argument(
        "--chunk_dir",
        default=None,
        help=(
            "Directory used by the chunked builder. Defaults to "
            "<out_file stem>_chunks beside --out_file."
        ),
    )
    parser.add_argument(
        "--force_rebuild_chunks",
        action="store_true",
        help="For the chunked builder, rebuild completed case chunks instead of reusing them.",
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="For the chunked builder, skip chunk generation and only merge existing chunks.",
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="For the chunked builder, generate chunks but do not merge them yet.",
    )
    args = parser.parse_args()

    args.root = Path(args.root)
    if args.out_file is None:
        args.out_file = args.root / "kh_holmboe_dataset_augmented.h5"
    else:
        args.out_file = Path(args.out_file)
    if args.chunk_dir is None:
        args.chunk_dir = args.out_file.parent / f"{args.out_file.stem}_chunks"
    else:
        args.chunk_dir = Path(args.chunk_dir)
    if args.reference_gyf_file is None:
        args.reference_gyf_file = args.root / "KH/KH_test/Ri016_a10_Re1000/1/mean.h5"
    else:
        args.reference_gyf_file = Path(args.reference_gyf_file)

    args.time_window_offsets = parse_csv_ints(args.time_window_offsets)
    args.noise_stds = parse_csv_floats(args.noise_stds)
    args.downsample_factors = parse_csv_floats(args.downsample_factors)
    args.skip_cases = parse_csv_strings(args.skip_cases)
    args.test_cases = parse_csv_strings(args.test_cases)

    if args.n_x_profiles <= 0 or args.n_z_profiles <= 0:
        raise ValueError("profile counts must be positive")
    if args.train_fraction <= 0 or args.train_fraction >= 1:
        raise ValueError("--train_fraction must be between 0 and 1")
    if any(offset < 0 for offset in args.time_window_offsets):
        raise ValueError("--time_window_offsets must be >= 0 so windows start at/after peak")
    if any(std < 0 for std in args.noise_stds):
        raise ValueError("--noise_stds must be >= 0")
    if any(factor <= 0 for factor in args.downsample_factors):
        raise ValueError("--downsample_factors must be positive")
    if args.augmentation_fraction < 0 or args.augmentation_fraction >= 1:
        raise ValueError("--augmentation_fraction must be in [0, 1)")
    if args.gzip_level < 0 or args.gzip_level > 9:
        raise ValueError("--gzip_level must be between 0 and 9")
    return args


def number_keys(group):
    return sorted([key for key in group.keys() if key.isdigit()], key=int)


def uniform_indices(n_total: int, n_take: int) -> np.ndarray:
    return np.round(np.linspace(0, n_total - 1, n_take)).astype(int)


def first_local_peak(signal: np.ndarray) -> int:
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            return i
    return int(np.argmax(signal))


def movie_group(var_name: str, plane: str) -> str:
    if var_name == "th":
        return f"th1_{plane}"
    return f"{var_name}_{plane}"


def read_target_y(reference_gyf_file: Path) -> np.ndarray:
    with h5py.File(reference_gyf_file, "r") as f:
        return f["/gyf/0001"][:]


def interp_y(profiles: np.ndarray, old_y: np.ndarray, new_y: np.ndarray) -> np.ndarray:
    if len(old_y) == len(new_y) and np.allclose(old_y, new_y):
        return profiles.astype(np.float32)

    out = np.zeros((profiles.shape[0], len(new_y)), dtype=np.float32)
    for i in range(profiles.shape[0]):
        out[i] = np.interp(new_y, old_y, profiles[i])
    return out


def discover_cases(root: Path, skip_cases: set[str]):
    cases = []
    case_pattern = re.compile(r"^Ri\d+_a\d+_Re\d+$")

    for family in ["KH", "Holmboe"]:
        folder = root / family / f"{family}_test"
        if not folder.exists():
            print(f"Skip missing family folder: {folder}")
            continue

        for case_dir in sorted(folder.iterdir()):
            if not case_dir.is_dir():
                continue
            if not case_pattern.match(case_dir.name):
                continue
            if case_dir.name in skip_cases:
                continue

            run_dirs = []
            for run_dir in sorted(case_dir.iterdir(), key=lambda p: p.name):
                if run_dir.is_dir() and run_dir.name.isdigit():
                    if (run_dir / "mean.h5").exists() and (run_dir / "movie.h5").exists():
                        run_dirs.append(run_dir)

            if run_dirs:
                cases.append((family, case_dir.name, run_dirs))

    return cases


def movie_frame_available(movie_h5: h5py.File, key: str) -> bool:
    try:
        for var_name in VARIABLES:
            for plane in ["xy", "zy"]:
                if f"/{movie_group(var_name, plane)}/{key}" not in movie_h5:
                    return False
    except (KeyError, OSError, RuntimeError):
        return False
    return True


def get_frames(run_dirs: list[Path]):
    frames = []
    urms_int_list = []
    skipped_missing_movie = 0

    for run_dir in run_dirs:
        mean_file = run_dir / "mean.h5"
        movie_file = run_dir / "movie.h5"

        with h5py.File(mean_file, "r") as mean_h5, h5py.File(movie_file, "r") as movie_h5:
            for key in number_keys(mean_h5["/time"]):
                if not movie_frame_available(movie_h5, key):
                    skipped_missing_movie += 1
                    continue

                t = float(mean_h5[f"/time/{key}"][0])
                gyf = mean_h5[f"/gyf/{key}"][:]
                urms = mean_h5[f"/urms/{key}"][:]
                urms_int = TRAPEZOID(urms, gyf)

                frames.append({
                    "time": t,
                    "key": key,
                    "mean_file": mean_file,
                    "movie_file": movie_file,
                })
                urms_int_list.append(urms_int)

    if skipped_missing_movie:
        print(f"  skipped {skipped_missing_movie} frames with incomplete movie variables")

    order = np.argsort([frame["time"] for frame in frames])
    frames = [frames[i] for i in order]
    urms_int_list = np.array([urms_int_list[i] for i in order])
    return frames, urms_int_list


def read_window_data(
    family: str,
    case_name: str,
    frames: list[dict],
    start_i: int,
    peak_i: int,
    window_id: int,
    window_offset: int,
    x_idx: np.ndarray,
    z_idx: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    case_id = f"{family}:{case_name}"
    window_frames = frames[start_i:start_i + NT_NEW]
    times = np.array([frame["time"] for frame in window_frames])

    n_profiles = len(x_idx) + len(z_idx)
    data = np.zeros((n_profiles, len(VARIABLES), NY_NEW, NT_NEW), dtype=np.float32)

    for t_i, frame in enumerate(window_frames):
        with h5py.File(frame["mean_file"], "r") as f:
            old_y = f[f"/gyf/{frame['key']}"][:]

        with h5py.File(frame["movie_file"], "r") as f:
            for v_i, var_name in enumerate(VARIABLES):
                xy = f[f"/{movie_group(var_name, 'xy')}/{frame['key']}"][:]
                zy = f[f"/{movie_group(var_name, 'zy')}/{frame['key']}"][:]

                xy_profiles = xy[:, x_idx].T
                zy_profiles = zy[:, z_idx].T
                profiles = np.concatenate([xy_profiles, zy_profiles], axis=0)
                data[:, v_i, :, t_i] = interp_y(profiles, old_y, target_y)

    metadata = []
    for idx in x_idx:
        metadata.append({
            "case_id": case_id,
            "family": family,
            "case": case_name,
            "plane": "xy",
            "axis_index": int(idx),
            "window_id": int(window_id),
            "window_offset_from_peak": int(window_offset),
            "peak_frame_index": int(peak_i),
            "start_frame_index": int(start_i),
            "t_start": float(times[0]),
            "t_end": float(times[-1]),
            "augmentation": "clean",
            "augmentation_downsample_factor": 1.0,
            "augmentation_noise_std": 0.0,
        })

    for idx in z_idx:
        metadata.append({
            "case_id": case_id,
            "family": family,
            "case": case_name,
            "plane": "zy",
            "axis_index": int(idx),
            "window_id": int(window_id),
            "window_offset_from_peak": int(window_offset),
            "peak_frame_index": int(peak_i),
            "start_frame_index": int(start_i),
            "t_start": float(times[0]),
            "t_end": float(times[-1]),
            "augmentation": "clean",
            "augmentation_downsample_factor": 1.0,
            "augmentation_noise_std": 0.0,
        })

    return data, metadata


def build_case(family: str, case_name: str, run_dirs: list[Path], target_y: np.ndarray, args):
    case_id = f"{family}:{case_name}"
    frames, urms_int = get_frames(run_dirs)
    if len(frames) < NT_NEW:
        print(f"Skip {case_id}: only {len(frames)} complete movie frames available.")
        return None, None

    peak_i = first_local_peak(urms_int)

    with h5py.File(frames[0]["movie_file"], "r") as f:
        sample_xy = f[f"/u_xy/{frames[0]['key']}"][:]
        sample_zy = f[f"/u_zy/{frames[0]['key']}"][:]

    x_idx = uniform_indices(sample_xy.shape[1], args.n_x_profiles)
    z_idx = uniform_indices(sample_zy.shape[1], args.n_z_profiles)

    case_data = []
    case_metadata = []
    for window_id, offset in enumerate(args.time_window_offsets):
        start_i = peak_i + offset
        end_i = start_i + NT_NEW
        if end_i > len(frames):
            print(
                f"Skip window for {case_id}: peak+{offset} needs frames "
                f"[{start_i}, {end_i}), only {len(frames)} available."
            )
            continue

        print(f"  window {window_id}: start=peak+{offset}, frames [{start_i}, {end_i})")
        try:
            data, metadata = read_window_data(
                family,
                case_name,
                frames,
                start_i,
                peak_i,
                window_id,
                offset,
                x_idx,
                z_idx,
                target_y,
            )
        except (KeyError, OSError, RuntimeError) as exc:
            print(f"  skip window {window_id} for {case_id}: failed to read movie data ({exc})")
            continue
        case_data.append(data)
        case_metadata.extend(metadata)

    if not case_data:
        print(f"Skip {case_id}: no valid 100-frame windows at/after peak.")
        return None, None

    return np.concatenate(case_data, axis=0), case_metadata


def train_val_indices(n: int, train_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_train = int(train_fraction * n)
    return order[:n_train], order[n_train:]


def update_min_max(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray):
    temp = data.copy()
    eps_i = VARIABLES.index("epsilon")
    temp[:, eps_i] = np.log(np.maximum(temp[:, eps_i], EPSILON_FLOOR))

    for i in range(len(VARIABLES)):
        mins[i] = min(mins[i], float(np.min(temp[:, i])))
        maxs[i] = max(maxs[i], float(np.max(temp[:, i])))


def normalize(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    temp = data.copy()
    eps_i = VARIABLES.index("epsilon")
    temp[:, eps_i] = np.log(np.maximum(temp[:, eps_i], EPSILON_FLOOR))

    for i in range(len(VARIABLES)):
        scale = maxs[i] - mins[i]
        if scale <= 0:
            temp[:, i] = 0.0
        else:
            temp[:, i] = 2.0 * (temp[:, i] - mins[i]) / scale - 1.0

    return np.clip(temp, -1.0, 1.0).astype(np.float32)


def resize_axis(data: np.ndarray, new_size: int, axis: int) -> np.ndarray:
    old_size = data.shape[axis]
    if old_size == new_size:
        return data.astype(np.float32, copy=True)

    old_grid = np.linspace(0.0, 1.0, old_size)
    new_grid = np.linspace(0.0, 1.0, new_size)
    moved = np.moveaxis(data, axis, -1)
    flat = moved.reshape(-1, old_size)
    out = np.empty((flat.shape[0], new_size), dtype=np.float32)
    for i in range(flat.shape[0]):
        out[i] = np.interp(new_grid, old_grid, flat[i])
    out = out.reshape(moved.shape[:-1] + (new_size,))
    return np.moveaxis(out, -1, axis)


def downsample_upsample(data: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1:
        return data.astype(np.float32, copy=True)

    height, width = data.shape[-2:]
    down_h = max(1, int(round(height / factor)))
    down_w = max(1, int(round(width / factor)))
    low_res = resize_axis(data, down_h, axis=-2)
    low_res = resize_axis(low_res, down_w, axis=-1)
    restored = resize_axis(low_res, height, axis=-2)
    restored = resize_axis(restored, width, axis=-1)
    return restored.astype(np.float32)


def perturbation_specs(noise_stds: list[float], downsample_factors: list[float]):
    specs = []
    seen = set()
    for factor in downsample_factors:
        for noise_std in noise_stds:
            factor = float(factor)
            noise_std = float(noise_std)
            if factor == 1.0 and noise_std == 0.0:
                continue
            key = (factor, noise_std)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
    return specs


def augment_train_data(data: np.ndarray, metadata: list[dict], args):
    if args.no_train_augmentation or args.augmentation_fraction == 0:
        return data, metadata

    rng = np.random.default_rng(args.augmentation_seed)
    specs = perturbation_specs(args.noise_stds, args.downsample_factors)
    if not specs:
        print("  no non-clean augmentation specs; using clean train data only")
        return data, metadata

    n_clean = data.shape[0]
    n_augmented = int(round(n_clean * args.augmentation_fraction / (1.0 - args.augmentation_fraction)))
    if n_augmented <= 0:
        return data, metadata

    augmented_chunks = [data]
    augmented_metadata = list(metadata)
    source_indices = rng.integers(0, n_clean, size=n_augmented)
    spec_indices = rng.integers(0, len(specs), size=n_augmented)

    print(
        "  train augmentation: "
        f"clean={n_clean}, augmented={n_augmented}, "
        f"target_aug_fraction={args.augmentation_fraction:g}"
    )

    for spec_i, (factor, noise_std) in enumerate(specs):
        selected = source_indices[spec_indices == spec_i]
        if selected.size == 0:
            continue

        x = downsample_upsample(data[selected], factor)
        if noise_std > 0:
            x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)

        variant_name = f"downsample_{factor:g}_noise_{noise_std:g}"
        augmented_chunks.append(x.astype(np.float32))
        for source_index in selected:
            item = dict(metadata[int(source_index)])
            item["augmentation"] = variant_name
            item["augmentation_downsample_factor"] = float(factor)
            item["augmentation_noise_std"] = float(noise_std)
            item["augmentation_seed"] = int(args.augmentation_seed)
            item["augmentation_source_index"] = int(source_index)
            augmented_metadata.append(item)

    return np.concatenate(augmented_chunks, axis=0), augmented_metadata
