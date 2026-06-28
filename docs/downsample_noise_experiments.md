# Downsample and Noise Experiments

This experiment measures how final test performance changes when the training
split is randomly downsampled and/or the model sees Gaussian input noise during
training.

The test set is not perturbed. Noise is applied only inside the training loop,
after loading a clean batch. Reconstruction and epsilon-flow targets stay clean,
so the experiment measures robustness to noisy inputs rather than changing the
ground truth.

## Quick Smoke Test

Use this first to make sure paths and dependencies are correct:

```bash
python src/run_downsample_noise_experiments.py \
  --main_h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --main_label_csv data/RM_summary_table.csv \
  --test_h5 data/test_dataset_keep_epsilon.h5 \
  --test_label_csv data/test_RM_summary_table.csv \
  --fractions 1.0,0.25 \
  --noise_stds 0.0,0.05 \
  --epochs 2 \
  --batch_size 4 \
  --test_batch_size 16 \
  --output_root outputs/downsample_noise_smoke \
  --checkpoint_root checkpoints/downsample_noise_smoke
```

## Main Sweep

```bash
python src/run_downsample_noise_experiments.py \
  --main_h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --main_label_csv data/RM_summary_table.csv \
  --test_h5 data/test_dataset_keep_epsilon.h5 \
  --test_label_csv data/test_RM_summary_table.csv \
  --fractions 1.0,0.5,0.25,0.1 \
  --noise_stds 0.0,0.01,0.05,0.1 \
  --epochs 60 \
  --batch_size 4 \
  --test_batch_size 16 \
  --seed 0 \
  --output_root outputs/downsample_noise_sweep \
  --checkpoint_root checkpoints/downsample_noise_sweep
```

The summary is written to:

```text
outputs/downsample_noise_sweep/summary.csv
```

Each row reports:

```text
train_fraction,input_noise_std,n_labeled,mae,rmse,bias,mse
```

Lower `mae`, `rmse`, and `mse` are better. `bias` shows whether predictions are
systematically high or low.

## With External Fine-Tuning

The original notebook workflow trains on the main dataset, fine-tunes on the
external keep-epsilon dataset, then tests on the external test split. To include
that step for every sweep condition:

```bash
python src/run_downsample_noise_experiments.py \
  --run_finetune \
  --main_h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --main_label_csv data/RM_summary_table.csv \
  --test_h5 data/test_dataset_keep_epsilon.h5 \
  --test_label_csv data/test_RM_summary_table.csv \
  --fractions 1.0,0.5,0.25,0.1 \
  --noise_stds 0.0,0.01,0.05,0.1 \
  --epochs 60 \
  --finetune_epochs 30 \
  --batch_size 4 \
  --test_batch_size 16 \
  --output_root outputs/downsample_noise_sweep_finetune \
  --checkpoint_root checkpoints/downsample_noise_sweep_finetune
```

By default, the external fine-tuning step uses the full external training split
with no added noise. This isolates how degraded main training affects the final
pipeline. If you want to perturb the fine-tuning data the same way as the main
training data, add:

```bash
--perturb_finetune_like_main
```

## Single Manual Run

The training script also supports one-off experiments:

```bash
python src/train_multihead_epsilon_flow.py \
  --h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --label_csv data/RM_summary_table.csv \
  --input_variables buoyancy,reduced_shear,log_epsilon \
  --train_fraction 0.25 \
  --input_noise_std 0.05 \
  --noise_variables all \
  --epochs 60 \
  --batch_size 4 \
  --lr 1e-4 \
  --lambda_recon 1.0 \
  --lambda_epsilon 1.0 \
  --epsilon_input_mask_prob 0.5 \
  --eval_force_mask_epsilon \
  --mask_prob 0.15 \
  --save checkpoints/example_downsample_noise.pt \
  --metrics_csv outputs/example_downsample_noise_loss_history.csv \
  --loss_plot outputs/example_downsample_noise_loss_curves.png
```
