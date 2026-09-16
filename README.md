# Synthetic ECG Multidimensional Evaluation

Reproducibility repository for the study:

**Beyond Fidelity: Multidimensional Evaluation and Admissibility Analysis of Synthetic ECG Generators**

The repository provides the experimental code and reported results for evaluating synthetic ECG generators across complementary dimensions of quality and utility. The evaluated generator families are TimeGAN, LSTM-VAE, and DDPM.

## Evaluation dimensions

The experimental protocol reports:

- **Distributional discrepancy:** macro-average class-wise RBF-MMD² between generated and held-out real beats.
- **Morphological diversity:** mean pairwise Euclidean distance among generated beats.
- **Generation stability:** output variation under the specified perturbation protocol.
- **Diagnostic utility:** train-on-synthetic, test-on-real (TSTR) accuracy and macro-F1.
- **Class-wise recall balance:** `1 - std(class recalls)`.
- **Physiological validity:** fraction of generated beats satisfying the specified amplitude and first-difference screening rules.

## Datasets and classes

- **MIT-BIH Arrhythmia Database:** `N, S, V, F, Q`
- **MIT-BIH Supraventricular Arrhythmia Database (SVDB):** `N, S, V`
- Random seeds: `19, 88, 123`
- Beat length: `256`
- Generated samples per class: `2,000`

Raw PhysioNet data are not redistributed in this repository.

## Reproducibility scope

The code reproduces generator training and evaluation from prepared beat arrays. Expected inputs are four NumPy arrays for each dataset and seed:

```text
data/prepared/<dataset>/seed_<seed>/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

The prepared arrays are not committed because of their size. Their expected shapes and class counts are recorded in `splits/prepared_class_counts.csv` and can be checked with `scripts/validate_prepared_data.py`.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── trust_metrics.py
│   ├── train_classwise_timegan_lstmvae.py
│   ├── ddpm_model.py
│   ├── train_ddpm.py
│   └── evaluate_ddpm_trust.py
├── scripts/
│   ├── validate_prepared_data.py
│   ├── verify_results.py
│   └── capture_environment.py
├── splits/
│   ├── README.md
│   └── prepared_class_counts.csv
├── results/
│   ├── README.md
│   ├── mitbih_seed_results.csv
│   ├── mitbih_summary.csv
│   ├── svdb_seed_results.csv
│   └── svdb_summary.csv
├── docs/
│   └── REPRODUCIBILITY.md
└── .github/workflows/verify-results.yml
```

## Installation

```bash
python -m pip install -r requirements.txt
```

The reported TimeGAN/LSTM-VAE experiments were executed in Google Colab with an NVIDIA T4 GPU. For new runs, record the software and hardware environment with:

```bash
python scripts/capture_environment.py
```

## Validate prepared inputs

MIT-BIH:

```bash
python scripts/validate_prepared_data.py \
  --dataset mitbih \
  --prepared_root data/prepared/mitbih
```

SVDB:

```bash
python scripts/validate_prepared_data.py \
  --dataset svdb \
  --prepared_root data/prepared/svdb
```

## TimeGAN and LSTM-VAE experiments

One TimeGAN and one LSTM-VAE are trained separately for each arrhythmia class.

MIT-BIH:

```bash
python src/train_classwise_timegan_lstmvae.py \
  --dataset mitbih \
  --project_root . \
  --prepared_root data/prepared/mitbih \
  --seeds 19,88,123 \
  --no_proxies
```

SVDB:

```bash
python src/train_classwise_timegan_lstmvae.py \
  --dataset svdb \
  --project_root . \
  --prepared_root data/prepared/svdb \
  --seeds 19,88,123 \
  --no_proxies
```

## DDPM experiments

Example DDPM training command for MIT-BIH seed 19:

```bash
python src/train_ddpm.py \
  --data_dir data/prepared/mitbih/seed_19 \
  --out_dir outputs/ddpm_mitbih_seed19 \
  --seed 19 \
  --num_classes 5 \
  --epochs 120 \
  --batch_size 64 \
  --T 300 \
  --ddim_steps 50 \
  --guidance 1.0 \
  --samples_per_class 2000 \
  --x0_clip 6
```

Evaluate generated DDPM samples and checkpoints:

```bash
python src/evaluate_ddpm_trust.py \
  --dataset mitbih \
  --project_root . \
  --prepared_root data/prepared/mitbih \
  --generated_template 'outputs/ddpm_mitbih_seed{seed}' \
  --seeds 19,88,123
```

Use `svdb`, `data/prepared/svdb`, and `outputs/ddpm_svdb_seed{seed}` for SVDB.

## Reported results

### MIT-BIH mean ± SD

| Model | MMD² ↓ | Diversity ↑ | Stability error ↓ | Accuracy ↑ | Macro-F1 ↑ | Recall balance ↑ | Validity ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDPM | 0.3662 ± 0.0556 | 22.2430 ± 0.1838 | 0.0192 ± 0.0059 | 0.4704 ± 0.0558 | 0.3654 ± 0.0399 | 0.8062 ± 0.1229 | 0.8646 ± 0.0052 |
| LSTM-VAE | 0.6747 ± 0.1088 | 7.2824 ± 0.4480 | 0.0015 ± 0.0021 | 0.5757 ± 0.0066 | 0.2031 ± 0.0514 | 0.6268 ± 0.0255 | 0.9695 ± 0.0311 |
| TimeGAN | 0.3693 ± 0.1128 | 16.9229 ± 0.5315 | 0.0046 ± 0.0041 | 0.4777 ± 0.1135 | 0.3083 ± 0.0503 | 0.7237 ± 0.0836 | 0.8186 ± 0.0526 |

### SVDB mean ± SD

| Model | MMD² ↓ | Diversity ↑ | Stability error ↓ | Accuracy ↑ | Macro-F1 ↑ | Recall balance ↑ | Validity ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DDPM | 0.0896 ± 0.0185 | 22.1333 ± 0.0139 | 0.0230 ± 0.0080 | 0.6399 ± 0.0716 | 0.6180 ± 0.0498 | 0.8737 ± 0.0570 | 0.8471 ± 0.0183 |
| LSTM-VAE | 0.2362 ± 0.0876 | 7.2930 ± 5.2062 | 0.0013 ± 0.0021 | 0.3934 ± 0.1453 | 0.3030 ± 0.0572 | 0.7265 ± 0.1254 | 0.9976 ± 0.0042 |
| TimeGAN | 0.0705 ± 0.0077 | 20.6854 ± 0.5190 | 0.0142 ± 0.0015 | 0.4680 ± 0.0510 | 0.4653 ± 0.0487 | 0.8414 ± 0.0418 | 0.7561 ± 0.0353 |

Seed-level values are provided in `results/mitbih_seed_results.csv` and `results/svdb_seed_results.csv`.

## Metric details

For MMD², each class uses an RBF kernel bandwidth estimated from real training beats only. The same class-specific bandwidth is then used when evaluating all generator families for that seed. The reported dataset-level value is the macro-average over classes.

Physiological validity uses the normalized-beat screening rules `max|x| <= 5` and `max|Δx| <= 4`. These thresholds are empirical screening criteria and are not presented as clinically calibrated acceptance limits.

## Result verification

```bash
python scripts/verify_results.py
```

Additional execution details are provided in `docs/REPRODUCIBILITY.md`.
