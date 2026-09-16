# Reproducibility protocol

## Experimental configuration

- Seeds: `19, 88, 123`
- Sequence length: `256`
- MIT-BIH classes: `N, S, V, F, Q`
- SVDB classes: `N, S, V`
- TimeGAN: one model per class
- LSTM-VAE: one model per class
- DDPM: class-conditional model
- Synthetic sample count: `2,000` per class
- MMD evaluation sample cap: `500` samples per class when available
- MMD bandwidth: estimated per seed/class from real training beats only and shared across generator families within that seed/class

## Execution order

1. Place prepared arrays under `data/prepared/<dataset>/seed_<seed>/`.
2. Validate shapes and class counts with `scripts/validate_prepared_data.py`.
3. Run `src/train_classwise_timegan_lstmvae.py`.
4. Train DDPM with `src/train_ddpm.py`, or place compatible generated arrays/checkpoints under the paths supplied to the evaluator.
5. Run `src/evaluate_ddpm_trust.py`.
6. Compare seed-level metrics with the CSV files under `results/`.
7. Record the environment with `scripts/capture_environment.py`.

## Distributional discrepancy

For each seed and arrhythmia class, the RBF kernel bandwidth is estimated using real training beats only. The resulting gamma is then held constant when evaluating TimeGAN, LSTM-VAE, and DDPM for that class. Class-wise biased empirical RBF-MMD² values are macro-averaged to obtain the reported dataset-level value.

## TimeGAN minibatches

The TimeGAN moment-matching term computes a within-batch standard deviation. When a minibatch contains a single observation, the implementation uses a finite singleton-safe form of this term rather than requesting an unbiased sample standard deviation, which is undefined for `n=1`.

## Class imbalance

Some MIT-BIH splits contain very few examples for the F class. In particular, seed 123 contains 19 held-out F beats. For this reason, seed-level and class-wise outputs should be retained alongside aggregate means.

## Hardware and environment

The reported TimeGAN/LSTM-VAE runs were executed in Google Colab with an NVIDIA T4 GPU. Exact environment reproduction should use a captured package snapshot from `scripts/capture_environment.py` when new runs are performed.

## Prepared-data requirement

This repository expects prepared beat arrays as inputs and does not redistribute the PhysioNet source data. The expected class counts are listed in `splits/prepared_class_counts.csv`.
