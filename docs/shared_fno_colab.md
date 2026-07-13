# Colab: SharedFNO compact-latent RM + flow training

This workflow keeps the old encoder style, adds an FNO spectral mixer, and
uses a compact latent feature instead of the old full-resolution encoder output.

## 1. Start a GPU runtime

In Colab:

```text
Runtime -> Change runtime type -> T4 GPU/A100 GPU
```

## 2. Clone the repository

```bash
git clone https://github.com/ShuaiMeng0601/PRISM-Physics-informed-Representation-learning-In-Stratified-shear-Mixing.git
cd PRISM-Physics-informed-Representation-learning-In-Stratified-shear-Mixing
```

## 3. Install dependencies

Colab usually already includes PyTorch. Install the repo dependencies:

```bash
pip install -r requirements.txt
```

## 4. Mount Google Drive and link data

The script uses the previous three-variable dataset by default:

```text
kh_holmboe_dataset_keep_epsilon.h5
```

It expects the H5 and RM table under the same Drive experiment folder used by
the existing repository workflow:

```text
/content/drive/MyDrive/ML_turbulence/experiment/
```

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create repo-local links:

```bash
python scripts/setup_colab_drive_links.py
```

After this, the expected paths are:

```text
data/kh_holmboe_dataset_keep_epsilon.h5
data/RM_summary_table.csv
```

If your Drive folder is elsewhere:

```bash
python scripts/setup_colab_drive_links.py \
  --experiment_dir "/content/drive/MyDrive/your_folder/experiment"
```

## 5. Train the SharedFNO model

```bash
python src/train_shared_fno_rm_flow.py \
  --h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --label_csv data/RM_summary_table.csv \
  --input_variables buoyancy,reduced_shear,log_epsilon \
  --epochs 100 \
  --batch_size 4 \
  --device cuda \
  --save checkpoints/shared_fno_rm_flow_model.pt \
  --metrics_csv outputs/shared_fno_rm_flow_loss_history.csv \
  --loss_plot outputs/shared_fno_rm_flow_loss_curves.png
```

If GPU memory is tight:

```bash
python src/train_shared_fno_rm_flow.py \
  --h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --label_csv data/RM_summary_table.csv \
  --input_variables buoyancy,reduced_shear,log_epsilon \
  --epochs 100 \
  --batch_size 2 \
  --dim 48 \
  --latent_channels 48 \
  --flow_base_channels 48 \
  --device cuda \
  --save checkpoints/shared_fno_rm_flow_model_small.pt \
  --metrics_csv outputs/shared_fno_rm_flow_loss_history_small.csv \
  --loss_plot outputs/shared_fno_rm_flow_loss_curves_small.png
```

## 6. Expected shapes

Default settings:

```text
input: (B,3,491,200)
compact latent: (B,64,50,20)
RM prediction: (B,1)
flow velocity/reconstruction: (B,1,491,200)
```

## 7. Fine-tune from the pretrained SharedFNO checkpoint

This mirrors the old fine-tune settings:

```text
epochs = 30
heads_only_epochs = 5
lr = 2e-5
lambda_flow = 0.2
batch_size = 4
```

Run after pretraining has produced `checkpoints/shared_fno_rm_flow_model.pt`:

```bash
python src/finetune_shared_fno_rm_flow.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --init_checkpoint checkpoints/shared_fno_rm_flow_model.pt \
  --input_variables buoyancy,reduced_shear,log_epsilon \
  --epochs 30 \
  --heads_only_epochs 5 \
  --batch_size 4 \
  --lr 2e-5 \
  --lambda_flow 0.2 \
  --device cuda \
  --save checkpoints/shared_fno_rm_flow_finetuned.pt \
  --metrics_csv outputs/shared_fno_rm_flow_finetune_loss_history.csv \
  --loss_plot outputs/shared_fno_rm_flow_finetune_loss_curves.png
```

To save directly to Google Drive, replace the output paths with absolute Drive
paths, for example:

```bash
--save /content/drive/MyDrive/ML_turbulence/experiment/checkpoints/shared_fno_rm_flow_finetuned.pt
```

## 8. Important notes

- The flow branch does not condition on `Pr`, `Ri`, `Re`, or `a`.
- The flow branch only uses noisy buoyancy, flow time, and compact latent.
- Missing variables are handled through `variable_mask` and training-time
  variable dropout.
- Pretraining/from-scratch training uses `src/train_shared_fno_rm_flow.py`.
- Fine-tuning from a pretrained SharedFNO checkpoint uses
  `src/finetune_shared_fno_rm_flow.py`.

## 9. Visualize cross-attention and RM saliency

After training has produced `checkpoints/shared_fno_rm_flow_model.pt`, run:

```bash
python src/inspect_shared_fno_attention.py \
  --h5 data/kh_holmboe_dataset_keep_epsilon.h5 \
  --label_csv data/RM_summary_table.csv \
  --checkpoint checkpoints/shared_fno_rm_flow_model.pt \
  --split val \
  --sample 0 \
  --device cuda \
  --output_dir outputs/shared_fno_attention
```

The script writes a folder like:

```text
outputs/shared_fno_attention/val_sample_0000/
```

Key files:

- `variable_attention_matrix.png`: 3 x 3 cross-variable attention mass.
- `query_key_attention_heatmaps.png`: full query-variable x key-variable
  spatial attention maps; rows are the variable doing the attending, columns
  are the variable being attended to.
- `patch_attention_heatmaps.png`: attention received by each variable on the
  compact `50 x 20` patch grid after averaging over query variables.
- `attention_overlays.png`: attention maps upsampled and overlaid on the input
  fields.
- `rm_saliency.png`: gradient-based RM saliency on the compact latent.
- `attention_maps.npz`: raw arrays for later plotting.
