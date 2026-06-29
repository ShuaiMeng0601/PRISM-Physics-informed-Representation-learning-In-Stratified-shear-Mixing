# Data

Do not commit HDF5 datasets to git.

In Colab, mount Google Drive and create these links with:

```bash
python scripts/setup_colab_drive_links.py
```

The default Drive source is:

```text
/content/drive/MyDrive/ML_turbulence/experiment/
```

Expected local files:

```text
kh_holmboe_dataset_keep_epsilon.h5
test_dataset_keep_epsilon.h5
RM_summary_table.csv
test_RM_summary_table.csv
```

The training scripts expect HDF5 splits such as `train`, `val`, and `test`
with field arrays and metadata. The final workflow uses:

```text
buoyancy,reduced_shear,log_epsilon
```

Keep the large Drive copies in `ML_turbulence/experiment/` and copy or symlink
them into this folder when running locally.
