# Prepared input arrays

The experiments use the same prepared train/test arrays for all three generator families within each dataset and seed.

Expected layout:

```text
data/prepared/<dataset>/seed_<seed>/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

Large NumPy arrays are not stored in Git. `prepared_class_counts.csv` records the expected class counts for each dataset, seed, and split. Validate a local copy with:

```bash
python scripts/validate_prepared_data.py \
  --dataset mitbih \
  --prepared_root data/prepared/mitbih
```

and analogously for SVDB.

The repository's reproducibility scope begins from these prepared beat arrays. Raw PhysioNet data are not redistributed here.
