# PRISM: ML Turbulence Backbone

This repository is a GitHub-ready backbone extracted from the Drive project
`ML_turbulence` and the Colab notebook `Colab Notebooks/Train_ML.ipynb`.

It keeps the final multihead epsilon-flow workflow and leaves large datasets,
checkpoints, logs, and exploratory result folders out of version control.

## Included Backbone

- `src/dataset.py`: HDF5 dataset loader and label parsing.
- `src/models.py` and `src/models_scale_variable.py`: core neural network modules and supporting architectures.
- `src/train_multihead.py`: shared multihead regression head, baseline model, and loss helpers used by the epsilon-flow workflow.
- `src/train_multihead_epsilon_flow.py`: train and fine-tune the final model.
- `src/test_multihead_epsilon_flow.py`: evaluate the model on a split or external test set.
- `src/infer_log_epsilon_flow.py`: generate log-epsilon field inference.
- `src/inspect_multihead_epsilon_flow.py`: visualize epsilon-flow reconstructions.
- `src/inspect_epsilon_flow_cross_attention.py`: visualize cross-variable attention.
- `src/visualize_flow_inputs.py`: inspect/visualize flow inputs.
- `src/run_downsample_noise_experiments.py`: sweep train-set downsampling and train-time input noise, then summarize test performance.

## Project Layout

```text
.
├── src/
├── configs/
├── data/
├── checkpoints/
├── outputs/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Install PyTorch for your CUDA/CPU environment first, then install the rest:

```bash
pip install -r requirements.txt
```

## Data

Put local copies or symlinks in `data/`:

```text
data/kh_holmboe_dataset_keep_epsilon.h5
data/test_dataset_keep_epsilon.h5
data/RM_summary_table.csv
data/test_RM_summary_table.csv
```

The HDF5 files are intentionally ignored by git.

## Main Training

```bash
python src/train_multihead_epsilon_flow.py \
  --h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --input_variables buoyancy,reduced_shear,log_epsilon \
  --epochs 60 \
  --batch_size 4 \
  --lr 1e-4 \
  --lambda_recon 1.0 \
  --lambda_epsilon 1.0 \
  --epsilon_input_mask_prob 0.5 \
  --eval_force_mask_epsilon \
  --mask_prob 0.15 \
  --save checkpoints/multihead_epsilon_flow_model.pt \
  --metrics_csv outputs/epsilon_flow_loss_history.csv \
  --loss_plot outputs/epsilon_flow_loss_curves.png
```

## External Fine-Tuning

```bash
python src/train_multihead_epsilon_flow.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --init_checkpoint checkpoints/multihead_epsilon_flow_model.pt \
  --input_variables buoyancy,reduced_shear,log_epsilon \
  --epochs 30 \
  --batch_size 4 \
  --lr 2e-5 \
  --lambda_recon 0.2 \
  --lambda_epsilon 1.0 \
  --epsilon_input_mask_prob 0.5 \
  --eval_force_mask_epsilon \
  --mask_prob 0.15 \
  --save checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --metrics_csv outputs/epsilon_flow_finetune_loss_history.csv \
  --loss_plot outputs/epsilon_flow_finetune_loss_curves.png
```

## Test

```bash
python src/test_multihead_epsilon_flow.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --split test \
  --batch_size 16 \
  --force_mask_epsilon \
  --output_dir outputs/epsilon_flow_test_results_external
```

## Downsample + Noise Experiments

To measure how training-set size and train-time input noise affect final test
performance, run a sweep:

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
  --output_root outputs/downsample_noise_sweep \
  --checkpoint_root checkpoints/downsample_noise_sweep
```

The sweep writes per-run logs/results under `outputs/downsample_noise_sweep/`
and a combined summary table at `outputs/downsample_noise_sweep/summary.csv`.

For the full original workflow, add `--run_finetune` so each perturbed main
training run is followed by external fine-tuning before test.

## Visualize

```bash
python src/inspect_multihead_epsilon_flow.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --split test \
  --sample 0 \
  --n_samples 5 \
  --steps 32 \
  --force_mask_epsilon \
  --output_dir outputs/epsilon_flow_inspect_external
```

```bash
python src/infer_log_epsilon_flow.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --split test \
  --sample 0 \
  --n_samples 10 \
  --steps 32 \
  --output_dir outputs/log_epsilon_inference_external
```

## Provenance

The final workflow comes from `Train_ML.ipynb`, especially cells 61-75:
main epsilon-flow training, external fine-tuning, testing, inference, and
attention visualization.
