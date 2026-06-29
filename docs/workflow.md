# Workflow

This is the streamlined workflow extracted from `Train_ML.ipynb`.

## 1. Data

Use the keep-epsilon HDF5 datasets:

```text
kh_holmboe_dataset_keep_epsilon.h5
test_dataset_keep_epsilon.h5
```

The main input variables are:

```text
buoyancy,reduced_shear,log_epsilon
```

## 2. Model

The final model is defined in `src/models_scale_variable.py`.

At a high level, the workflow uses a multi-variable encoder over turbulence
fields, then combines regression/reconstruction objectives with an epsilon-flow
branch. The final notebook workflow masks epsilon during evaluation to test
whether the model can infer missing epsilon structure from the remaining fields.

## 3. Train Main Model

Run `src/train_multihead_epsilon_flow.py` on the main KH/Holmboe dataset and
save `multihead_epsilon_flow_model.pt`.

## 4. Fine-Tune External Model

Run the same training script with `--init_checkpoint` on the external
keep-epsilon dataset and save `multihead_epsilon_flow_finetuned_external.pt`.

## 5. Evaluate

Run `src/test_multihead_epsilon_flow.py` with `--force_mask_epsilon`.

The notebook's external test run reported:

```text
n_labeled: 340
mae: 0.027548348640694338
rmse: 0.03077877824307018
bias: 0.002585204120944528
mse: 0.0009473331901360904
```

## 6. Visualize

Use:

```text
src/inspect_multihead_epsilon_flow.py
src/infer_log_epsilon_flow.py
src/inspect_epsilon_flow_cross_attention.py
src/visualize_flow_inputs.py
```

These cover reconstruction/inference panels, log-epsilon inference, attention
diagnostics, and flow-input sanity checks.

## 7. Test-Time Robustness Sweep

Use `src/run_downsample_noise_experiments.py` to test how a fixed trained
checkpoint performs when test inputs are spatially downsampled and/or perturbed
with Gaussian noise. The runner writes per-run logs and a combined summary CSV
with `mae`, `rmse`, `bias`, and `mse`.

See `docs/downsample_noise_experiments.md` for commands.
