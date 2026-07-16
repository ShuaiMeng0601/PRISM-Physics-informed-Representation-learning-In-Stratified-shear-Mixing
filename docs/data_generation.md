# Data Generation

This repository keeps large HDF5 files out of git. The builder scripts are kept
here so the datasets can be regenerated from the raw KH/Holmboe `mean.h5` and
`movie.h5` files.

## Current Builder

Use one entry point:

```text
scripts/build_kh_holmboe_dataset_augmented.py
```

Shared lower-level helpers live in:

```text
scripts/dataset_building.py
```

The builder writes recoverable per-case chunks before merging the final HDF5:

```text
raw chunks -> normalized/augmented processed chunks -> final merged dataset
```

This makes long dataset builds resumable. Completed chunks are reused by
default. Use `--force_rebuild_chunks` only when a chunk should be regenerated.

## Expanded Dataset Setup

The expanded dataset uses:

```text
variables: u,v,w,th,epsilon
sample shape: (5, 501, 100)
profiles per window: 100 xy + 50 zy = 150 samples
time windows: peak+0, peak+25, peak+50
train augmentation target: 70% clean / 30% augmented
augmentation noise stds: 0,0.01,0.03
augmentation downsample factors: 1,2,4
```

Validation and test splits are clean. Augmentation is train-only.

## Build Command

```bash
python scripts/build_kh_holmboe_dataset_augmented.py \
  --root /Volumes/LaCie/Pr_1_KH_Holm \
  --out_file /Volumes/LaCie/Pr_1_KH_Holm/ml_framework/experiment/kh_holmboe_dataset_augmented.h5 \
  --chunk_dir /Volumes/LaCie/Pr_1_KH_Holm/ml_framework/experiment/kh_holmboe_dataset_augmented_chunks \
  --reference_gyf_file /Volumes/LaCie/Pr_1_KH_Holm/KH/KH_test/Ri016_a10_Re1000/1/mean.h5 \
  --n_x_profiles 100 \
  --n_z_profiles 50 \
  --time_window_offsets 0,25,50 \
  --augmentation_fraction 0.3 \
  --noise_stds 0,0.01,0.03 \
  --downsample_factors 1,2,4 \
  --gzip_level 4
```

The generated file from the July 2026 run was:

```text
/Volumes/LaCie/Pr_1_KH_Holm/ml_framework/experiment/kh_holmboe_dataset_augmented.h5
```

with:

```text
train: (12336, 5, 501, 100)
val:   (2160, 5, 501, 100)
test:  (900, 5, 501, 100)
```

## Old vs New Dataset

| Feature | Original keep-epsilon dataset | Expanded augmented dataset |
|---|---:|---:|
| File | `kh_holmboe_dataset_keep_epsilon.h5` | `kh_holmboe_dataset_augmented.h5` |
| Variables | `buoyancy,reduced_shear,log_epsilon` | `u,v,w,th,epsilon` |
| Sample shape | `(3, 491, 200)` | `(5, 501, 100)` |
| y grid | cropped to 491 | full 501 |
| Time windows | one 200-frame window | three 100-frame windows |
| Window starts | first local peak | `peak+0`, `peak+25`, `peak+50` |
| Clean samples per complete case | 170 | 450 |
| Augmentation | none | train-only noise/downsample |
| Train samples | 4760 | 12336 |
| Val samples | 1190 | 2160 |
| Test samples | 340 | 900 |
| Total samples | 6290 | 15396 |

The expanded build used two held-out test cases:

```text
KH:Ri020_a05_Re1000
Holmboe:Ri016_a05_Re2000
```

Some raw cases are skipped when fewer than 100 complete movie frames are
available after missing/incomplete frame filtering.
