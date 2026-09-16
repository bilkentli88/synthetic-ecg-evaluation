from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import pairwise_distances


def _as_2d(X: np.ndarray) -> np.ndarray:
    """Return beats as (n_samples, n_features)."""
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[-1] == 1:
        X = X[..., 0]
    return X.reshape(len(X), -1).astype(np.float32, copy=False)


def physiological_validity(X: np.ndarray, amax: float = 5.0, dmax: float = 4.0) -> float:
    """Fraction of beats satisfying the manuscript's amplitude and first-difference checks."""
    X2 = _as_2d(X)
    amp_ok = np.max(np.abs(X2), axis=1) <= amax
    grad_ok = np.max(np.abs(np.diff(X2, axis=1)), axis=1) <= dmax
    return float(np.mean(amp_ok & grad_ok))


def diversity_score(
    X: np.ndarray,
    seed: int,
    sample: int = 500,
    pairs: int = 2000,
) -> float:
    """Mean pairwise Euclidean distance on a random subset."""
    X2 = _as_2d(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X2), size=min(sample, len(X2)), replace=False)
    sub = X2[idx]
    i1 = rng.integers(0, len(sub), size=pairs)
    i2 = rng.integers(0, len(sub), size=pairs)
    return float(np.mean(np.linalg.norm(sub[i1] - sub[i2], axis=1)))


def _subsample_rows(X: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    X2 = _as_2d(X)
    if len(X2) <= max_n:
        return X2
    idx = rng.choice(len(X2), size=max_n, replace=False)
    return X2[idx]


def estimate_rbf_gamma(
    X: np.ndarray,
    seed: int,
    max_samples: int = 500,
) -> Tuple[float, int]:
    """Estimate a deterministic RBF bandwidth from one reference distribution.

    The bandwidth is based only on real training data so every generator compared
    within a seed/class uses exactly the same kernel scale.
    """
    rng = np.random.default_rng(seed)
    Xs = _subsample_rows(X, max_samples, rng)
    if len(Xs) < 2:
        return 1.0, len(Xs)
    d2 = pairwise_distances(Xs, metric="sqeuclidean")
    nz = d2[d2 > 0]
    if len(nz) == 0:
        return 1.0, len(Xs)
    med2 = float(np.median(nz))
    return 1.0 / max(med2, 1e-12), len(Xs)


def classwise_rbf_gammas(
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    class_labels: Sequence[int],
    seed: int,
    max_samples_each: int = 500,
) -> Dict[int, Dict[str, float]]:
    """Fixed per-class RBF gamma values estimated from real training beats only."""
    out: Dict[int, Dict[str, float]] = {}
    y_reference = np.asarray(y_reference)
    for c in class_labels:
        Xc = X_reference[y_reference == c]
        gamma, n_used = estimate_rbf_gamma(
            Xc,
            seed=seed * 1009 + int(c) * 9176 + 701,
            max_samples=max_samples_each,
        )
        out[int(c)] = {
            "Gamma": float(gamma),
            "NReferenceUsedForGamma": int(n_used),
            "NReferenceAvailableForGamma": int(len(Xc)),
        }
    return out


def rbf_mmd2_biased_fixed_gamma(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
    seed: int,
    max_samples_each: int = 500,
) -> Tuple[float, int, int]:
    """Biased empirical RBF-MMD^2 using a supplied fixed kernel gamma."""
    rng = np.random.default_rng(seed)
    Xs = _subsample_rows(X, max_samples_each, rng)
    Ys = _subsample_rows(Y, max_samples_each, rng)
    if len(Xs) < 1 or len(Ys) < 1:
        return float("nan"), len(Xs), len(Ys)

    dxx = pairwise_distances(Xs, metric="sqeuclidean")
    dyy = pairwise_distances(Ys, metric="sqeuclidean")
    dxy = pairwise_distances(Xs, Ys, metric="sqeuclidean")
    kxx = np.exp(-gamma * dxx)
    kyy = np.exp(-gamma * dyy)
    kxy = np.exp(-gamma * dxy)
    mmd2 = float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
    if mmd2 < 0 and mmd2 > -1e-10:
        mmd2 = 0.0
    return mmd2, len(Xs), len(Ys)


def classwise_mmd2(
    X_gen: np.ndarray,
    y_gen: np.ndarray,
    X_real: np.ndarray,
    y_real: np.ndarray,
    class_labels: Sequence[int],
    seed: int,
    gamma_by_class: Dict[int, Dict[str, float]],
    max_samples_each: int = 500,
) -> Tuple[float, List[Dict[str, float]]]:
    """Macro-average class-wise RBF-MMD^2 against held-out real beats.

    Crucially, each class uses a kernel bandwidth estimated once from real
    training data and then held fixed across TimeGAN, LSTM-VAE, DDPM, and
    proxy generators. This makes MMD values directly comparable across models.
    """
    rows: List[Dict[str, float]] = []
    vals: List[float] = []
    y_gen = np.asarray(y_gen)
    y_real = np.asarray(y_real)
    for c in class_labels:
        Xg = X_gen[y_gen == c]
        Xr = X_real[y_real == c]
        gamma_info = gamma_by_class[int(c)]
        gamma = float(gamma_info["Gamma"])
        mmd2, ng, nr = rbf_mmd2_biased_fixed_gamma(
            Xg,
            Xr,
            gamma=gamma,
            seed=seed * 1009 + int(c) * 9176 + 17,
            max_samples_each=max_samples_each,
        )
        rows.append({
            "Class": int(c),
            "MMD2": mmd2,
            "Gamma": gamma,
            "NReferenceUsedForGamma": int(gamma_info["NReferenceUsedForGamma"]),
            "NReferenceAvailableForGamma": int(gamma_info["NReferenceAvailableForGamma"]),
            "NGeneratedUsed": int(ng),
            "NRealUsed": int(nr),
            "NGeneratedAvailable": int(len(Xg)),
            "NRealAvailable": int(len(Xr)),
        })
        if np.isfinite(mmd2):
            vals.append(float(mmd2))
    return (float(np.mean(vals)) if vals else float("nan")), rows


def tstr_metrics(
    X_gen: np.ndarray,
    y_gen: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: Sequence[int],
    seed: int,
    rf_estimators: int = 50,
) -> Dict[str, float]:
    """Train-on-Synthetic, Test-on-Real diagnostic utility and class-recall balance."""
    Xg = _as_2d(X_gen)
    Xt = _as_2d(X_test)
    clf = RandomForestClassifier(
        n_estimators=rf_estimators,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(Xg, y_gen)
    preds = clf.predict(Xt)
    recalls = recall_score(
        y_test,
        preds,
        average=None,
        labels=list(class_labels),
        zero_division=0,
    )
    return {
        "UtilityAccuracy": float(accuracy_score(y_test, preds)),
        "MacroF1": float(f1_score(
            y_test,
            preds,
            average="macro",
            labels=list(class_labels),
            zero_division=0,
        )),
        "ClassRecallBalance": float(1.0 - np.std(recalls)),
    }


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_by_model(rows: List[Dict], metric_names: Sequence[str]) -> List[Dict]:
    models = []
    for r in rows:
        if r["Model"] not in models:
            models.append(r["Model"])
    out = []
    for model in models:
        mr = [r for r in rows if r["Model"] == model]
        row = {"Model": model, "NSeeds": len(mr)}
        for metric in metric_names:
            vals = np.asarray([float(r[metric]) for r in mr], dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals))
        out.append(row)
    return out
