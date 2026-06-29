# Checkpoints

Model checkpoints are not tracked by git.

In Colab, mount Google Drive and create checkpoint symlinks with:

```bash
python scripts/setup_colab_drive_links.py
```

The default Drive source for the fine-tuned checkpoint is:

```text
/content/drive/MyDrive/ML_turbulence/experiment/model_1/
```

Expected runtime outputs:

```text
multihead_epsilon_flow_model.pt
multihead_epsilon_flow_finetuned_external.pt
```

The first checkpoint is trained on the main KH/Holmboe dataset. The second is
fine-tuned on the external keep-epsilon test dataset.
