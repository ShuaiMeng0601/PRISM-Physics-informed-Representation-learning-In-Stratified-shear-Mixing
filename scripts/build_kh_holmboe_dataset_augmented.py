#!/usr/bin/env python3
"""Build the augmented KH/Holmboe dataset as recoverable per-case chunks.

The output is produced in three recoverable steps:

1. Write raw clean HDF5 chunks while collecting normalization statistics.
2. Convert raw chunks into normalized/augmented processed chunks.
3. Merge completed processed chunks into the final dataset.

If the process is interrupted, rerun the same command. Completed chunks are
reused by default; pass --force_rebuild_chunks to regenerate them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np

from dataset_building import (
    NT_NEW,
    NY_NEW,
    VARIABLES,
    augment_train_data,
    build_case,
    discover_cases,
    normalize,
    parse_args,
    read_target_y,
    train_val_indices,
    update_min_max,
)


def safe_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", case_id)


def chunk_path(chunk_dir: Path, case_id: str) -> Path:
    return chunk_dir / f"{safe_name(case_id)}.h5"


def raw_chunk_dir(args) -> Path:
    return args.chunk_dir / "raw"


def processed_chunk_dir(args) -> Path:
    return args.chunk_dir / "processed"


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_manifest(chunk_dir: Path, payload: dict):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default))


def read_normalization(chunk_dir: Path):
    path = chunk_dir / "normalization.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return np.array(payload["min"], dtype=np.float32), np.array(payload["max"], dtype=np.float32)


def write_normalization(chunk_dir: Path, mins: np.ndarray, maxs: np.ndarray):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "normalization.json").write_text(
        json.dumps({"min": mins.tolist(), "max": maxs.tolist()}, indent=2, sort_keys=True)
    )


def processed_chunk_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if f.attrs.get("status", "") != "complete":
                return False
            return all(f"{split}/X" in f and f"{split}/metadata_json" in f for split in ["train", "val", "test"])
    except (OSError, RuntimeError):
        return False


def raw_chunk_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if f.attrs.get("status", "") != "complete":
                return False
            return "X" in f and "metadata_json" in f
    except (OSError, RuntimeError):
        return False


def create_split(group, x: np.ndarray, metadata: list[dict], gzip_level: int):
    compression = "gzip" if gzip_level > 0 else None
    compression_opts = gzip_level if gzip_level > 0 else None
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("X", data=x.astype(np.float32, copy=False), compression=compression, compression_opts=compression_opts)
    group.create_dataset(
        "metadata_json",
        data=[json.dumps(m, sort_keys=True) for m in metadata],
        dtype=string_dtype,
    )


def empty_x() -> np.ndarray:
    return np.empty((0, len(VARIABLES), NY_NEW, NT_NEW), dtype=np.float32)


def write_raw_case_chunk(
    raw_file: Path,
    family: str,
    case_name: str,
    data: np.ndarray,
    metadata: list[dict],
    target_y: np.ndarray,
    args,
):
    case_id = f"{family}:{case_name}"
    tmp_file = raw_file.with_suffix(".tmp.h5")
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    if tmp_file.exists():
        tmp_file.unlink()

    compression = "gzip" if args.gzip_level > 0 else None
    compression_opts = args.gzip_level if args.gzip_level > 0 else None
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(tmp_file, "w") as f:
        f.attrs["status"] = "running"
        f.attrs["case_id"] = case_id
        f.attrs["family"] = family
        f.attrs["case"] = case_name
        f.attrs["variables"] = json.dumps(VARIABLES)
        f.attrs["sample_shape"] = f"({len(VARIABLES)}, {NY_NEW}, {NT_NEW})"
        f.attrs["is_test_case"] = json.dumps(case_id in args.test_cases)
        f.attrs["n_x_profiles"] = int(args.n_x_profiles)
        f.attrs["n_z_profiles"] = int(args.n_z_profiles)
        f.attrs["time_window_offsets_after_peak"] = json.dumps(args.time_window_offsets)
        f.create_dataset("y", data=target_y.astype(np.float32))
        f.create_dataset("frame_index", data=np.arange(NT_NEW).astype(np.int32))
        f.create_dataset(
            "X",
            data=data.astype(np.float32, copy=False),
            compression=compression,
            compression_opts=compression_opts,
        )
        f.create_dataset(
            "metadata_json",
            data=[json.dumps(m, sort_keys=True) for m in metadata],
            dtype=string_dtype,
        )
        f.attrs["status"] = "complete"

    tmp_file.replace(raw_file)
    return {
        "case_id": case_id,
        "path": str(raw_file),
        "samples": int(data.shape[0]),
    }


def read_raw_case_chunk(raw_file: Path):
    with h5py.File(raw_file, "r") as f:
        data = f["X"][:]
        metadata = [
            json.loads(item.decode("utf-8") if isinstance(item, bytes) else str(item))
            for item in f["metadata_json"][:]
        ]
    return data, metadata


def write_case_chunk(
    chunk_file: Path,
    family: str,
    case_name: str,
    data: np.ndarray,
    metadata: list[dict],
    target_y: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    args,
):
    case_id = f"{family}:{case_name}"
    tmp_file = chunk_file.with_suffix(".tmp.h5")
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    if tmp_file.exists():
        tmp_file.unlink()

    normalized = normalize(data, mins, maxs)
    del data

    train_x = empty_x()
    val_x = empty_x()
    test_x = empty_x()
    train_meta: list[dict] = []
    val_meta: list[dict] = []
    test_meta: list[dict] = []

    if case_id in args.test_cases:
        test_x = normalized
        test_meta = metadata
    else:
        train_idx, val_idx = train_val_indices(normalized.shape[0], args.train_fraction, args.random_seed)
        clean_train = normalized[train_idx]
        clean_train_meta = [metadata[i] for i in train_idx]
        train_x, train_meta = augment_train_data(clean_train, clean_train_meta, args)
        val_x = normalized[val_idx]
        val_meta = [metadata[i] for i in val_idx]

    with h5py.File(tmp_file, "w") as f:
        f.attrs["status"] = "running"
        f.attrs["case_id"] = case_id
        f.attrs["family"] = family
        f.attrs["case"] = case_name
        f.attrs["variables"] = json.dumps(VARIABLES)
        f.attrs["sample_shape"] = f"({len(VARIABLES)}, {NY_NEW}, {NT_NEW})"
        f.attrs["is_test_case"] = json.dumps(case_id in args.test_cases)
        f.attrs["n_x_profiles"] = int(args.n_x_profiles)
        f.attrs["n_z_profiles"] = int(args.n_z_profiles)
        f.attrs["time_window_offsets_after_peak"] = json.dumps(args.time_window_offsets)
        f.attrs["augmentation_fraction"] = float(args.augmentation_fraction)
        f.attrs["augmentation_noise_stds"] = json.dumps(args.noise_stds)
        f.attrs["augmentation_downsample_factors"] = json.dumps(args.downsample_factors)

        f.create_dataset("y", data=target_y.astype(np.float32))
        f.create_dataset("frame_index", data=np.arange(NT_NEW).astype(np.int32))
        norm = f.create_group("normalization")
        norm.create_dataset("min", data=mins.astype(np.float32))
        norm.create_dataset("max", data=maxs.astype(np.float32))

        create_split(f.create_group("train"), train_x, train_meta, args.gzip_level)
        create_split(f.create_group("val"), val_x, val_meta, args.gzip_level)
        create_split(f.create_group("test"), test_x, test_meta, args.gzip_level)
        f.attrs["status"] = "complete"

    tmp_file.replace(chunk_file)
    return {
        "case_id": case_id,
        "path": str(chunk_file),
        "train": int(train_x.shape[0]),
        "val": int(val_x.shape[0]),
        "test": int(test_x.shape[0]),
    }


def generate_raw_chunks_and_normalization(cases, target_y: np.ndarray, args):
    raw_dir = raw_chunk_dir(args)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cached = read_normalization(args.chunk_dir)
    use_cached_norm = cached is not None and not args.force_rebuild_chunks
    if use_cached_norm:
        mins, maxs = cached
        print(f"Using cached normalization: {args.chunk_dir / 'normalization.json'}")
    else:
        mins = np.full(len(VARIABLES), np.inf)
        maxs = np.full(len(VARIABLES), -np.inf)

    print("\nStep 1/3: writing raw per-case chunks and collecting normalization")
    results = []
    for family, case_name, run_dirs in cases:
        case_id = f"{family}:{case_name}"
        out = chunk_path(raw_dir, case_id)

        data = None
        if raw_chunk_is_complete(out) and not args.force_rebuild_chunks:
            print(f"\nReuse raw chunk {case_id}: {out}")
            with h5py.File(out, "r") as f:
                result = {"case_id": case_id, "path": str(out), "samples": int(f["X"].shape[0])}
            if not use_cached_norm and case_id not in args.test_cases:
                data, _ = read_raw_case_chunk(out)
        else:
            print(f"\nBuilding raw chunk {case_id}")
            data, metadata = build_case(family, case_name, run_dirs, target_y, args)
            if data is None:
                print(f"  skipped {case_id}; no raw chunk written")
                continue
            result = write_raw_case_chunk(out, family, case_name, data, metadata, target_y, args)
            print(f"  wrote raw samples={result['samples']} -> {out}")

        if not use_cached_norm and case_id not in args.test_cases and data is not None:
            train_idx, _ = train_val_indices(data.shape[0], args.train_fraction, args.random_seed)
            update_min_max(data[train_idx], mins, maxs)
        elif not use_cached_norm and case_id in args.test_cases:
            print("  held-out test case; not used for normalization")

        del data
        results.append(result)
        write_manifest(
            args.chunk_dir,
            {
                "status": "raw_chunking",
                "raw_dir": str(raw_dir),
                "raw_chunks": results,
                "normalization_ready": bool(use_cached_norm),
            },
        )

    if not use_cached_norm:
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            raise RuntimeError("Failed to compute finite normalization statistics.")
        write_normalization(args.chunk_dir, mins, maxs)

    write_manifest(
        args.chunk_dir,
        {
            "status": "raw_chunks_complete",
            "raw_dir": str(raw_dir),
            "raw_chunks": results,
            "normalization_ready": True,
        },
    )
    return mins, maxs


def generate_processed_chunks(cases, target_y: np.ndarray, mins: np.ndarray, maxs: np.ndarray, args):
    raw_dir = raw_chunk_dir(args)
    processed_dir = processed_chunk_dir(args)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("\nStep 2/3: converting raw chunks into normalized/augmented chunks")
    results = []
    for family, case_name, run_dirs in cases:
        case_id = f"{family}:{case_name}"
        raw_file = chunk_path(raw_dir, case_id)
        out = chunk_path(processed_dir, case_id)

        if processed_chunk_is_complete(out) and not args.force_rebuild_chunks:
            print(f"\nReuse processed chunk {case_id}: {out}")
            with h5py.File(out, "r") as f:
                results.append({
                    "case_id": case_id,
                    "path": str(out),
                    "train": int(f["train/X"].shape[0]),
                    "val": int(f["val/X"].shape[0]),
                    "test": int(f["test/X"].shape[0]),
                })
            continue

        if not raw_chunk_is_complete(raw_file):
            print(f"\nSkip {case_id}: missing raw chunk {raw_file}")
            continue

        print(f"\nBuilding processed chunk {case_id}")
        try:
            data, metadata = read_raw_case_chunk(raw_file)
        except (OSError, RuntimeError) as exc:
            print(f"  raw chunk read failed for {case_id}: {exc}")
            print("  rebuilding this raw chunk from source files")
            data, metadata = build_case(family, case_name, run_dirs, target_y, args)
            if data is None:
                print(f"  skipped {case_id}; could not rebuild raw chunk")
                continue
            write_raw_case_chunk(raw_file, family, case_name, data, metadata, target_y, args)
            data, metadata = read_raw_case_chunk(raw_file)

        result = write_case_chunk(out, family, case_name, data, metadata, target_y, mins, maxs, args)
        print(
            "  wrote "
            f"train={result['train']}, val={result['val']}, test={result['test']} -> {out}"
        )
        results.append(result)
        write_manifest(
            args.chunk_dir,
            {
                "status": "processed_chunking",
                "processed_dir": str(processed_dir),
                "processed_chunks": results,
            },
        )

    write_manifest(
        args.chunk_dir,
        {
            "status": "processed_chunks_complete",
            "processed_dir": str(processed_dir),
            "processed_chunks": results,
        },
    )
    return results


def create_final_file(out_file: Path, target_y: np.ndarray, mins: np.ndarray, maxs: np.ndarray, args):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if args.gzip_level > 0 else None
    compression_opts = args.gzip_level if args.gzip_level > 0 else None
    string_dtype = h5py.string_dtype(encoding="utf-8")

    f = h5py.File(out_file, "w")
    f.attrs["variables"] = json.dumps(VARIABLES)
    f.attrs["sample_shape"] = f"({len(VARIABLES)}, {NY_NEW}, {NT_NEW})"
    f.attrs["peak_signal"] = "urms_int = trapz(gyf, urms, 1)"
    f.attrs["n_x_profiles"] = int(args.n_x_profiles)
    f.attrs["n_z_profiles"] = int(args.n_z_profiles)
    f.attrs["time_window_offsets_after_peak"] = json.dumps(args.time_window_offsets)
    f.attrs["train_augmentation_enabled"] = json.dumps(not args.no_train_augmentation)
    f.attrs["augmentation_fraction"] = float(args.augmentation_fraction)
    f.attrs["augmentation_noise_stds"] = json.dumps(args.noise_stds)
    f.attrs["augmentation_downsample_factors"] = json.dumps(args.downsample_factors)
    f.attrs["write_mode"] = "chunked_then_merged"

    f.create_dataset("y", data=target_y.astype(np.float32))
    f.create_dataset("frame_index", data=np.arange(NT_NEW).astype(np.int32))
    norm = f.create_group("normalization")
    norm.create_dataset("min", data=mins.astype(np.float32))
    norm.create_dataset("max", data=maxs.astype(np.float32))

    for split in ["train", "val", "test"]:
        group = f.create_group(split)
        group.create_dataset(
            "X",
            shape=(0, len(VARIABLES), NY_NEW, NT_NEW),
            maxshape=(None, len(VARIABLES), NY_NEW, NT_NEW),
            chunks=(8, len(VARIABLES), NY_NEW, NT_NEW),
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
        )
        group.create_dataset("metadata_json", shape=(0,), maxshape=(None,), dtype=string_dtype, chunks=(1024,))
    return f


def append_from_chunk(final_group, chunk_group, block_size: int = 16):
    source_x = chunk_group["X"]
    source_meta = chunk_group["metadata_json"]
    n = source_x.shape[0]
    if n == 0:
        return

    dest_x = final_group["X"]
    dest_meta = final_group["metadata_json"]
    old_n = dest_x.shape[0]
    dest_x.resize((old_n + n, len(VARIABLES), NY_NEW, NT_NEW))
    dest_meta.resize((old_n + n,))

    write_i = old_n
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        dest_x[write_i:write_i + (end - start)] = source_x[start:end]
        meta = source_meta[start:end]
        dest_meta[write_i:write_i + (end - start)] = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in meta
        ]
        write_i += end - start


def merge_chunks(cases, target_y: np.ndarray, mins: np.ndarray, maxs: np.ndarray, args):
    print("\nStep 3/3: merging processed chunks")
    processed_dir = processed_chunk_dir(args)
    tmp_file = args.out_file.with_suffix(".tmp.h5")
    if tmp_file.exists():
        tmp_file.unlink()

    final = create_final_file(tmp_file, target_y, mins, maxs, args)
    merged = []
    try:
        for family, case_name, _ in cases:
            case_id = f"{family}:{case_name}"
            path = chunk_path(processed_dir, case_id)
            if not processed_chunk_is_complete(path):
                print(f"Skip missing/incomplete chunk during merge: {case_id}")
                continue

            print(f"  merging {case_id}")
            with h5py.File(path, "r") as chunk:
                for split in ["train", "val", "test"]:
                    append_from_chunk(final[split], chunk[split])
            final.flush()
            merged.append(case_id)
            write_manifest(args.chunk_dir, {"status": "merging", "merged_cases": merged})
    except Exception:
        final.close()
        raise

    counts = {split: int(final[f"{split}/X"].shape[0]) for split in ["train", "val", "test"]}
    final.attrs["status"] = "complete"
    final.flush()
    final.close()
    tmp_file.replace(args.out_file)
    write_manifest(args.chunk_dir, {"status": "complete", "merged_cases": merged, "counts": counts})
    print(f"Done. Wrote {args.out_file}")
    print(f"Counts: {counts}")


def main():
    args = parse_args()
    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    target_y = read_target_y(args.reference_gyf_file)
    cases = discover_cases(args.root, args.skip_cases)

    print(f"Found {len(cases)} cases:")
    for family, case_name, _ in cases:
        case_id = f"{family}:{case_name}"
        suffix = "  [test]" if case_id in args.test_cases else ""
        print(f"  {case_id}{suffix}")
    print(f"Chunk directory: {args.chunk_dir}")
    print(f"Raw chunk directory: {raw_chunk_dir(args)}")
    print(f"Processed chunk directory: {processed_chunk_dir(args)}")
    print(f"Final output: {args.out_file}")

    if args.merge_only:
        cached = read_normalization(args.chunk_dir)
        if cached is None:
            raise RuntimeError(f"--merge_only needs {args.chunk_dir / 'normalization.json'}")
        mins, maxs = cached
    else:
        mins, maxs = generate_raw_chunks_and_normalization(cases, target_y, args)
        generate_processed_chunks(cases, target_y, mins, maxs, args)

    if args.no_merge:
        print("\nChunks are ready; --no_merge was set, so final merge was skipped.")
        return
    merge_chunks(cases, target_y, mins, maxs, args)


if __name__ == "__main__":
    main()
