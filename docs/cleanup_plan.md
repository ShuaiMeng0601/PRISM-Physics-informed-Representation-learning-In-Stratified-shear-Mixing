# Drive Cleanup Plan

The new `github_backbone_2026-06-28` folder is intentionally small and
GitHub-oriented. It should be the folder you publish or clone into a repo.

## Keep In The Backbone

- Final model/data/training code from `experiment/model_1`.
- Small docs and config manifests.
- Placeholder README files for data, checkpoints, and outputs.

## Leave Out Of Git

- `*.h5` datasets.
- `*.pt` checkpoints.
- Loss plots, result CSV files, generated visualizations, and test result folders.
- `__pycache__`.
- Early exploratory scripts unless you want a separate `legacy/` folder.

## Archive Candidates In The Original Drive Folder

These are useful historically but should not be in the GitHub backbone:

- `cnn_test_results`, `train_CNN.py`, `test_CNN.py`
- old `test_results*`, `visualizations`, `multihead_visualizations*`
- `zero_shot_*` folders and scripts
- root-level `model.pt`, `loss_history.csv`, `loss_curves.png`, `train.log`
- duplicate large datasets outside `experiment/`

Do not delete these until the backbone repo runs end-to-end from fresh data and
checkpoint paths.
