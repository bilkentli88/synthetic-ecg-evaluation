from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from trust_metrics import (
    classwise_mmd2,
    classwise_rbf_gammas,
    diversity_score,
    physiological_validity,
    summarize_by_model,
    tstr_metrics,
    write_csv,
)


@dataclass
class Config:
    seeds: Tuple[int, ...] = (19, 88, 123)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Generator hyperparameters used in the reported experiments.
    tg_hidden: int = 24
    tg_layers: int = 3
    tg_batch: int = 64
    tg_lr: float = 0.001
    tg_epochs: int = 150

    vae_hidden: int = 64
    vae_latent: int = 20
    vae_layers: int = 1
    vae_batch: int = 64
    vae_lr: float = 0.001
    vae_epochs: int = 100
    gen_batch: int = 256

    beat_len: int = 256
    latent_delta: float = 0.05
    robust_n_total: int = 200
    diversity_sample: int = 500
    diversity_pairs: int = 2000
    rf_estimators: int = 50
    mmd_sample_per_class: int = 500

    baseline_noise: float = 0.35
    robust_noise: float = 0.08
    amax: float = 5.0
    dmax: float = 4.0

    # Harmonize generated class counts with the DDPM protocol in the repository.
    samples_per_class: int = 2000


cfg = Config()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _shape_for_rnn(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        return X[..., None]
    if X.ndim == 3 and X.shape[-1] == 1:
        return X
    raise ValueError(f"Expected (n, 256) or (n, 256, 1), got {X.shape}")


def load_prepared_seed(prepared_root: Path, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed_dir = prepared_root / f"seed_{seed}"
    required = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    missing = [name for name in required if not (seed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared arrays for seed {seed} in {seed_dir}: {missing}. "
            "Use the same DDPM-prepared arrays that generated the saved DDPM outputs so all models share the exact split."
        )
    X_train = _shape_for_rnn(np.load(seed_dir / "X_train.npy"))
    y_train = np.load(seed_dir / "y_train.npy").astype(np.int64)
    X_test = _shape_for_rnn(np.load(seed_dir / "X_test.npy"))
    y_test = np.load(seed_dir / "y_test.npy").astype(np.int64)
    return X_train, y_train, X_test, y_test


class TimeGAN_Net(nn.Module):
    def __init__(self, d_in, d_out, hidden, layers, output_sig=False):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, layers, batch_first=True)
        self.lin = nn.Linear(hidden, d_out)
        self.output_sig = output_sig
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        o, _ = self.rnn(x)
        o = self.lin(o)
        return self.sigmoid(o) if self.output_sig else o


class TimeGAN:
    def __init__(self):
        self.E = TimeGAN_Net(1, cfg.tg_hidden, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.R = TimeGAN_Net(cfg.tg_hidden, 1, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.G = TimeGAN_Net(1, cfg.tg_hidden, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.S = TimeGAN_Net(cfg.tg_hidden, cfg.tg_hidden, cfg.tg_hidden, cfg.tg_layers).to(cfg.device)
        self.D = TimeGAN_Net(cfg.tg_hidden, 1, cfg.tg_hidden, cfg.tg_layers, output_sig=True).to(cfg.device)
        self.opt_e = optim.Adam(list(self.E.parameters()) + list(self.R.parameters()), lr=cfg.tg_lr)
        self.opt_g = optim.Adam(list(self.G.parameters()) + list(self.S.parameters()), lr=cfg.tg_lr)
        self.opt_d = optim.Adam(self.D.parameters(), lr=cfg.tg_lr)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def train(self, X_train: np.ndarray) -> None:
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=cfg.tg_batch, shuffle=True)
        ae_epoch = int(cfg.tg_epochs * 0.2)
        sup_epoch = int(cfg.tg_epochs * 0.2)

        for _ in range(ae_epoch):
            for (x,) in loader:
                x = x.to(cfg.device)
                loss = self.mse(x, self.R(self.E(x)))
                self.opt_e.zero_grad(); loss.backward(); self.opt_e.step()

        for _ in range(sup_epoch):
            for (x,) in loader:
                x = x.to(cfg.device)
                h = self.E(x).detach()
                loss = self.mse(h[:, 1:], self.S(h)[:, :-1])
                self.opt_g.zero_grad(); loss.backward(); self.opt_g.step()

        for _ in range(cfg.tg_epochs):
            for (x,) in loader:
                x = x.to(cfg.device)
                b_size = x.size(0)
                z = torch.randn(b_size, cfg.beat_len, 1, device=cfg.device)

                e_hat = self.G(z)
                h_hat = self.S(e_hat)
                x_hat = self.R(h_hat)
                y_fake = self.D(h_hat)
                h = self.E(x).detach()
                loss_g_adv = self.bce(y_fake, torch.ones_like(y_fake))
                loss_s = self.mse(h[:, 1:], self.S(h)[:, :-1])
                # Use the sample standard-deviation term for minibatches with n > 1,
                # and a finite singleton-safe form when n = 1. PyTorch's
                # default unbiased std is undefined for n=1; in that case both
                # empirical within-batch standard deviations are zero, so the std
                # difference contributes zero while the mean-matching term remains.
                if b_size > 1:
                    std_hat = torch.std(x_hat, 0)
                    std_real = torch.std(x, 0)
                else:
                    std_hat = torch.zeros_like(x_hat[0])
                    std_real = torch.zeros_like(x[0])
                loss_mom = (
                    torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
                    + torch.mean(torch.abs(std_hat - std_real))
                )
                loss_g = loss_g_adv + 10 * loss_s + 100 * loss_mom
                self.opt_g.zero_grad(); loss_g.backward(); self.opt_g.step()

                y_real = self.D(h)
                h_hat_d = self.S(self.G(z)).detach()
                y_fake_d = self.D(h_hat_d)
                loss_d = (
                    self.bce(y_real, torch.ones_like(y_real))
                    + self.bce(y_fake_d, torch.zeros_like(y_fake_d))
                )
                self.opt_d.zero_grad(); loss_d.backward(); self.opt_d.step()

    @torch.no_grad()
    def generate(self, n: int, batch_size: int = 256) -> np.ndarray:
        self.G.eval(); self.S.eval(); self.R.eval()
        outs = []
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            z = torch.randn(b, cfg.beat_len, 1, device=cfg.device)
            outs.append(self.R(self.S(self.G(z))).cpu().numpy())
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def paired_mse(self, n: int, latent_delta: float) -> np.ndarray:
        self.G.eval(); self.S.eval(); self.R.eval()
        z = torch.randn(n, cfg.beat_len, 1, device=cfg.device)
        d = torch.randn_like(z) * latent_delta
        x1 = self.R(self.S(self.G(z)))
        x2 = self.R(self.S(self.G(z + d)))
        return torch.mean((x1 - x2) ** 2, dim=(1, 2)).cpu().numpy()


class LSTM_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        z = self.reparameterize(mu, logvar)
        d_in = self.decoder_input(z).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder_lstm(d_in)
        return self.final_layer(out), mu, logvar


class VAE_Trainer:
    def __init__(self):
        self.model = LSTM_VAE(1, cfg.vae_hidden, cfg.vae_latent, cfg.vae_layers).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.vae_lr)

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        mse = nn.functional.mse_loss(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld

    def train(self, X_train: np.ndarray) -> None:
        self.model.train()
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_train)), batch_size=cfg.vae_batch, shuffle=True)
        for _ in range(cfg.vae_epochs):
            for (x,) in loader:
                x = x.to(cfg.device)
                recon_x, mu, logvar = self.model(x)
                loss = self.loss_function(recon_x, x, mu, logvar)
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        d_in = self.model.decoder_input(z).unsqueeze(1).repeat(1, cfg.beat_len, 1)
        out, _ = self.model.decoder_lstm(d_in)
        return self.model.final_layer(out)

    @torch.no_grad()
    def generate(self, n: int, batch_size: int | None = None) -> np.ndarray:
        self.model.eval()
        batch_size = batch_size or cfg.gen_batch
        outs = []
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)
            z = torch.randn(b, cfg.vae_latent, device=cfg.device)
            outs.append(self._decode(z).cpu().numpy())
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def paired_mse(self, n: int, latent_delta: float) -> np.ndarray:
        self.model.eval()
        z = torch.randn(n, cfg.vae_latent, device=cfg.device)
        d = torch.randn_like(z) * latent_delta
        x1 = self._decode(z)
        x2 = self._decode(z + d)
        return torch.mean((x1 - x2) ** 2, dim=(1, 2)).cpu().numpy()


def train_classwise_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_labels: Sequence[int],
    seed: int,
) -> Tuple[Dict[int, TimeGAN], Dict[int, VAE_Trainer], List[Dict]]:
    tg_models: Dict[int, TimeGAN] = {}
    vae_models: Dict[int, VAE_Trainer] = {}
    counts = []
    for c in class_labels:
        Xc = X_train[y_train == c]
        if len(Xc) == 0:
            raise RuntimeError(f"No training samples for class {c}")
        counts.append({"Seed": seed, "Class": int(c), "NTrain": int(len(Xc))})

        class_seed = seed * 1000 + int(c) * 37 + 11
        print(f"    class {c}: n_train={len(Xc)} | TimeGAN seed={class_seed}")
        set_seeds(class_seed)
        tg = TimeGAN(); tg.train(Xc)
        tg_models[int(c)] = tg

        class_seed_vae = seed * 1000 + int(c) * 37 + 23
        print(f"    class {c}: n_train={len(Xc)} | LSTM-VAE seed={class_seed_vae}")
        set_seeds(class_seed_vae)
        vae = VAE_Trainer(); vae.train(Xc)
        vae_models[int(c)] = vae

    return tg_models, vae_models, counts


def generate_balanced(models: Dict[int, object], class_labels: Sequence[int], n_per_class: int, seed: int, kind: str):
    Xs, ys = [], []
    for c in class_labels:
        set_seeds(seed * 1000 + int(c) * 41 + (101 if kind == "TimeGAN" else 211))
        Xc = models[int(c)].generate(n_per_class)
        Xs.append(Xc)
        ys.append(np.full(n_per_class, int(c), dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def classwise_stability(models: Dict[int, object], class_labels: Sequence[int], seed: int) -> float:
    base = cfg.robust_n_total // len(class_labels)
    rem = cfg.robust_n_total % len(class_labels)
    vals = []
    for j, c in enumerate(class_labels):
        n = base + (1 if j < rem else 0)
        set_seeds(seed * 1000 + int(c) * 53 + 307)
        vals.extend(models[int(c)].paired_mse(n, cfg.latent_delta).tolist())
    return float(np.mean(vals))


def proxy_generate(X_train: np.ndarray, y_train: np.ndarray, class_labels: Sequence[int], noise: float, n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in class_labels:
        Xc = X_train[y_train == c]
        idx = rng.choice(len(Xc), size=n_per_class, replace=True)
        base = Xc[idx]
        syn = base + noise * rng.normal(size=base.shape).astype(np.float32)
        Xs.append(syn.astype(np.float32))
        ys.append(np.full(n_per_class, int(c), dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys)


def proxy_stability(X_train: np.ndarray, y_train: np.ndarray, class_labels: Sequence[int], noise: float, seed: int) -> float:
    # Perturbation-based proxy generation balanced over classes.
    rng = np.random.default_rng(seed)
    base_n = cfg.robust_n_total // len(class_labels)
    rem = cfg.robust_n_total % len(class_labels)
    per_sample = []
    for j, c in enumerate(class_labels):
        n = base_n + (1 if j < rem else 0)
        Xc = X_train[y_train == c]
        idx = rng.choice(len(Xc), size=n, replace=True)
        Xseed = Xc[idx]
        z = rng.normal(0, 1, Xseed.shape)
        d = rng.normal(0, cfg.latent_delta, Xseed.shape)
        x1 = Xseed + noise * z
        x2 = Xseed + noise * (z + d)
        per_sample.extend(np.mean((x1 - x2) ** 2, axis=(1, 2)).tolist())
    return float(np.mean(per_sample))


def evaluate_generated(
    model_name: str,
    seed: int,
    X_gen: np.ndarray,
    y_gen: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: Sequence[int],
    stability_error: float,
    gamma_by_class,
):
    mmd_macro, mmd_rows = classwise_mmd2(
        X_gen, y_gen, X_test, y_test,
        class_labels=class_labels,
        seed=seed,
        gamma_by_class=gamma_by_class,
        max_samples_each=cfg.mmd_sample_per_class,
    )
    tstr = tstr_metrics(
        X_gen, y_gen, X_test, y_test,
        class_labels=class_labels,
        seed=seed,
        rf_estimators=cfg.rf_estimators,
    )
    row = {
        "Model": model_name,
        "Seed": seed,
        "DistributionalMMD2": mmd_macro,
        "Diversity": diversity_score(X_gen, seed=seed, sample=cfg.diversity_sample, pairs=cfg.diversity_pairs),
        "StabilityError": stability_error,
        "UtilityAccuracy": tstr["UtilityAccuracy"],
        "MacroF1": tstr["MacroF1"],
        "ClassRecallBalance": tstr["ClassRecallBalance"],
        "PhysiologicalValidity": physiological_validity(X_gen, amax=cfg.amax, dmax=cfg.dmax),
    }
    for r in mmd_rows:
        r.update({"Model": model_name, "Seed": seed})
    return row, mmd_rows


def save_generated(out_dir: Path, dataset: str, model_name: str, seed: int, X: np.ndarray, y: np.ndarray) -> None:
    gdir = out_dir / "generated"
    gdir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_{model_name.lower().replace('-', '').replace('_','')}_seed{seed}"
    np.save(gdir / f"{stem}_X.npy", X.astype(np.float32))
    np.save(gdir / f"{stem}_y.npy", y.astype(np.int64))


def parse_args():
    p = argparse.ArgumentParser(description="Class-wise TimeGAN/LSTM-VAE experiment")
    p.add_argument("--dataset", choices=["mitbih", "svdb"], required=True)
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--prepared_root", type=str, default=None,
                   help="Defaults to data/prepared/<dataset>.")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seeds", type=str, default="19,88,123")
    p.add_argument("--samples_per_class", type=int, default=2000)
    p.add_argument("--mmd_sample_per_class", type=int, default=500)
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--no_proxies", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.device = args.device
    cfg.samples_per_class = args.samples_per_class
    cfg.mmd_sample_per_class = args.mmd_sample_per_class
    cfg.seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    if args.dataset == "mitbih":
        class_names = ["N", "S", "V", "F", "Q"]
    else:
        class_names = ["N", "S", "V"]
    class_labels = list(range(len(class_names)))

    root = Path(args.project_root)
    prepared_root = Path(args.prepared_root) if args.prepared_root else root / "data" / "prepared" / args.dataset
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else root / "outputs" / f"{args.dataset}_classwise_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": args.dataset,
        "class_names": class_names,
        "class_labels": class_labels,
        "prepared_root": str(prepared_root),
        "output_dir": str(out_dir),
        "config": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "protocol": {
            "timegan": "one model per arrhythmia class",
            "lstm_vae": "one model per arrhythmia class",
            "generated_samples_per_class": cfg.samples_per_class,
            "distributional_metric": "macro-average class-wise RBF-MMD^2 against held-out real beats",
            "mmd_bandwidth": "per-class gamma estimated from real training beats only and shared across generators within each seed/class",
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    raw_rows: List[Dict] = []
    mmd_rows_all: List[Dict] = []
    class_count_rows: List[Dict] = []

    print("=" * 72)
    print(f"Synthetic ECG experiment | dataset={args.dataset} | device={cfg.device}")
    print(f"prepared_root={prepared_root}")
    print(f"out_dir={out_dir}")
    print("=" * 72)

    for seed in cfg.seeds:
        print(f"\n>>> SEED {seed}")
        set_seeds(seed)
        X_train, y_train, X_test, y_test = load_prepared_seed(prepared_root, seed)
        print("  train:", X_train.shape, {class_names[c]: int(np.sum(y_train == c)) for c in class_labels})
        print("  test :", X_test.shape, {class_names[c]: int(np.sum(y_test == c)) for c in class_labels})

        gamma_by_class = classwise_rbf_gammas(
            X_train, y_train, class_labels, seed=seed,
            max_samples_each=cfg.mmd_sample_per_class,
        )
        print("  fixed MMD gammas from real training data:",
              {class_names[c]: round(gamma_by_class[c]["Gamma"], 8) for c in class_labels})

        tg_models, vae_models, counts = train_classwise_models(X_train, y_train, class_labels, seed)
        for r in counts:
            r["ClassName"] = class_names[r["Class"]]
        class_count_rows.extend(counts)

        X_tg, y_tg = generate_balanced(tg_models, class_labels, cfg.samples_per_class, seed, "TimeGAN")
        X_vae, y_vae = generate_balanced(vae_models, class_labels, cfg.samples_per_class, seed, "LSTM-VAE")
        save_generated(out_dir, args.dataset, "TimeGAN", seed, X_tg, y_tg)
        save_generated(out_dir, args.dataset, "LSTM-VAE", seed, X_vae, y_vae)

        stab_tg = classwise_stability(tg_models, class_labels, seed)
        stab_vae = classwise_stability(vae_models, class_labels, seed)
        row, mmd_rows = evaluate_generated("TimeGAN", seed, X_tg, y_tg, X_test, y_test, class_labels, stab_tg, gamma_by_class)
        raw_rows.append(row); mmd_rows_all.extend(mmd_rows)
        row, mmd_rows = evaluate_generated("LSTM-VAE", seed, X_vae, y_vae, X_test, y_test, class_labels, stab_vae, gamma_by_class)
        raw_rows.append(row); mmd_rows_all.extend(mmd_rows)

        if not args.no_proxies:
            for name, noise, offset in [("Baseline", cfg.baseline_noise, 401), ("Robust", cfg.robust_noise, 503)]:
                Xp, yp = proxy_generate(X_train, y_train, class_labels, noise, cfg.samples_per_class, seed + offset)
                save_generated(out_dir, args.dataset, name, seed, Xp, yp)
                stab = proxy_stability(X_train, y_train, class_labels, noise, seed + offset)
                row, mmd_rows = evaluate_generated(name, seed, Xp, yp, X_test, y_test, class_labels, stab, gamma_by_class)
                raw_rows.append(row); mmd_rows_all.extend(mmd_rows)

        metrics = [
            "DistributionalMMD2", "Diversity", "StabilityError", "UtilityAccuracy",
            "MacroF1", "ClassRecallBalance", "PhysiologicalValidity",
        ]
        write_csv(out_dir / f"{args.dataset}_seed_results.csv", raw_rows)
        write_csv(out_dir / f"{args.dataset}_classwise_mmd.csv", mmd_rows_all)
        write_csv(out_dir / f"{args.dataset}_training_class_counts.csv", class_count_rows)
        summary = summarize_by_model(raw_rows, metrics)
        write_csv(out_dir / f"{args.dataset}_summary.csv", summary)
        print("  saved intermediate results")

        # Free GPU memory before the next seed.
        del tg_models, vae_models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"\nDONE. Results: {out_dir}")
    print(f"ZIP: {out_dir}.zip")


if __name__ == "__main__":
    main()
