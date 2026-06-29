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
- `src/run_downsample_noise_experiments.py`: sweep test-time input noise and spatial downsampling for a trained checkpoint.
- `scripts/setup_colab_drive_links.py`: create Colab symlinks to `ML_turbulence/experiment/` in Google Drive.

## Project Layout

```text
.
├── src/
├── configs/
├── data/
├── checkpoints/
├── outputs/
├── docs/
├── scripts/
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

Large data and checkpoint files are intentionally not stored in git. In Google
Drive, this project expects your files under:

```text
/content/drive/MyDrive/ML_turbulence/experiment/
```

After mounting Google Drive in Colab, create repo-local symlinks with:

```bash
python scripts/setup_colab_drive_links.py
```

This links the Drive files into `data/` and `checkpoints/`:

```text
data/kh_holmboe_dataset_keep_epsilon.h5
data/test_dataset_keep_epsilon.h5
data/RM_summary_table.csv
data/test_RM_summary_table.csv
checkpoints/multihead_epsilon_flow_model.pt
checkpoints/multihead_epsilon_flow_finetuned_external.pt
```

If your Drive folder is mounted somewhere else, pass it explicitly:

```bash
python scripts/setup_colab_drive_links.py \
  --experiment_dir "/content/drive/MyDrive/ML_turbulence/experiment"
```

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

## Test-Time Downsample + Noise Experiments

To measure how a trained model responds to degraded test inputs, run a sweep
over test-time Gaussian noise and spatial downsampling. The checkpoint is fixed;
the script does not retrain the model.

```bash
python src/run_downsample_noise_experiments.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --downsample_factors 1.0,2.0,4.0,8.0 \
  --noise_stds 0.0,0.01,0.05,0.1 \
  --batch_size 16 \
  --force_mask_epsilon \
  --output_root outputs/test_input_robustness_sweep
```

The sweep writes per-run logs/results under `outputs/test_input_robustness_sweep/`
and a combined summary table at `outputs/test_input_robustness_sweep/summary.csv`.
`--downsample_factors 2.0` means the input fields are resized to half spatial
resolution and then resized back to the model's expected input size before
inference.

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
