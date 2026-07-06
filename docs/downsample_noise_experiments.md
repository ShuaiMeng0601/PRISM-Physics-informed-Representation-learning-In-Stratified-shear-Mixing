# Test-Time Downsample and Noise Experiments

This experiment measures how a fixed, already-trained checkpoint performs when
the test input fields are degraded.

The model is not retrained. The test labels stay clean. Only the input tensor
`x` is changed immediately before model inference.

## Colab Data Setup

Your Google Drive data live under:

```text
/content/drive/MyDrive/ML_turbulence/experiment/
```

After cloning the repo and mounting Drive, run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/PRISM-Physics-informed-Representation-learning-In-Stratified-shear-Mixing
!python scripts/setup_colab_drive_links.py
!ls -lhL data/test_dataset_keep_epsilon.h5
!ls -lhL data/test_RM_summary_table.csv
!ls -lhL checkpoints/multihead_epsilon_flow_finetuned_external.pt
```

## Perturbations

Two perturbations can be swept:

```text
input_noise_std
spatial_downsample_factor
```

`input_noise_std` adds Gaussian noise to selected test input channels.

`spatial_downsample_factor` resizes every test input field down and then back to
the original model input size. For example, factor `2.0` means:

```text
original H x W -> approximately H/2 x W/2 -> original H x W
```

This simulates a lower-resolution measurement while keeping the trained model
architecture and checkpoint unchanged.

## Quick Smoke Test

Use this first to make sure the checkpoint/data paths are correct:

```bash
python src/run_downsample_noise_experiments.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --downsample_factors 1.0,4.0 \
  --noise_stds 0.0,0.05 \
  --batch_size 16 \
  --force_mask_epsilon \
  --output_root outputs/test_input_robustness_smoke
```

## Main Sweep

```bash
python src/run_downsample_noise_experiments.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --downsample_factors 1.0,2.0,4.0,8.0,16.0,32.0 \
  --noise_stds 0.0,0.01,0.05,0.1 \
  --batch_size 16 \
  --force_mask_epsilon \
  --perturb_seed 0 \
  --skip_existing \
  --output_root outputs/test_input_robustness_sweep
```

With four noise levels and six downsample factors, the main sweep contains
24 runs. If the original `1,2,4,8` sweep already exists, `--skip_existing`
reuses those outputs and computes only the missing `16` and `32` cases.

The summary is written to:

```text
outputs/test_input_robustness_sweep/summary.csv
```

Each row reports:

```text
spatial_downsample_factor,input_noise_std,n_labeled,mae,rmse,bias,mse
```

Lower `mae`, `rmse`, and `mse` are better. `bias` shows whether predictions are
systematically high or low.

## Single Manual Test

You can also run one degraded test condition directly:

```bash
python src/test_multihead_epsilon_flow.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --split test \
  --batch_size 16 \
  --force_mask_epsilon \
  --input_noise_std 0.05 \
  --noise_variables all \
  --spatial_downsample_factor 4.0 \
  --perturb_seed 0 \
  --output_dir outputs/test_input_robustness_single
```

If `--force_mask_epsilon` is used, the model masks the epsilon channel before
encoding. In that case perturbing `log_epsilon` itself will not affect the
encoder input, because that channel is intentionally hidden.
