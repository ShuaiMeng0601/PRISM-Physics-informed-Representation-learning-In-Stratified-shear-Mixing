# Colab Workflow Commands

These are the final backbone commands extracted from `Train_ML.ipynb`, rewritten
with repo-relative paths.

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

## External Test

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

## Inspection

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

## Log-Epsilon Inference

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

## Cross-Attention Visualization

```bash
python src/inspect_epsilon_flow_cross_attention.py \
  --h5 data/test_dataset_keep_epsilon.h5 \
  --label_csv data/test_RM_summary_table.csv \
  --checkpoint checkpoints/multihead_epsilon_flow_finetuned_external.pt \
  --split val \
  --sample 139 \
  --n_samples 1 \
  --force_mask_epsilon \
  --output_dir outputs/epsilon_flow_cross_attention_val_sample_139
```
