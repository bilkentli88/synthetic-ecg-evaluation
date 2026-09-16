from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from trust_metrics import (
    classwise_mmd2,
    classwise_rbf_gammas,
    diversity_score,
    physiological_validity,
    summarize_by_model,
    tstr_metrics,
    write_csv,
)

# Local DDPM implementation matching the checkpoint architecture used in the experiments.
from ddpm_model import ECGDiffusion


@torch.no_grad()
def ddim_from_initial(
    gen: ECGDiffusion,
    x_init: torch.Tensor,
    y: torch.Tensor,
    ddim_steps: int = 50,
    guidance: float = 1.0,
    x0_clip: float = 6.0,
    use_ema: bool = True,
) -> torch.Tensor:
    backup = None
    if use_ema and gen.ema_state is not None:
        backup = {k: v.detach().clone() for k, v in gen.model.state_dict().items()}
        gen.model.load_state_dict(gen.ema_state)
    gen.model.eval()
    device = gen.device
    x = x_init.to(device)
    y = y.to(device)
    B = x.shape[0]
    step_idx = torch.linspace(gen.T - 1, 0, ddim_steps, device=device).long()
    for i, t_cur_tensor in enumerate(step_idx):
        t_cur = int(t_cur_tensor.item())
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
        ac_t = gen.acp[t_cur]
        if i + 1 < len(step_idx):
            t_next = int(step_idx[i + 1].item())
            ac_next = gen.acp[t_next] if t_next > 0 else torch.tensor(1.0, device=device)
        else:
            ac_next = torch.tensor(1.0, device=device)
        eps_c = gen.model(x, t_batch, y)
        if guidance is not None and guidance != 1.0:
            y_null = torch.full_like(y, gen.model.null_idx)
            eps_u = gen.model(x, t_batch, y_null)
            eps = eps_u + guidance * (eps_c - eps_u)
        else:
            eps = eps_c
        x0 = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
        if x0_clip is not None and x0_clip > 0:
            x0 = x0.clamp(-x0_clip, x0_clip)
        x = ac_next.sqrt() * x0 + (1 - ac_next).sqrt() * eps
    if backup is not None:
        gen.model.load_state_dict(backup)
    return x


def load_ddpm_from_checkpoint(ckpt_path: Path, device: str, default_num_classes: int) -> ECGDiffusion:
    ckpt = torch.load(ckpt_path, map_location=device)
    extra = ckpt.get("extra", {}) or {}
    seq_len = int(ckpt.get("seq_len", extra.get("seq_len", 256)))
    num_classes = int(ckpt.get("num_classes", extra.get("num_classes", default_num_classes)))
    T = int(ckpt.get("T", extra.get("T", 300)))
    p_uncond = float(ckpt.get("p_uncond", extra.get("p_uncond", 0.1)))
    base_ch = int(extra.get("base_ch", 64))
    seed = int(extra.get("seed", 0))
    gen = ECGDiffusion(
        seq_len=seq_len,
        num_classes=num_classes,
        T=T,
        base_ch=base_ch,
        p_uncond=p_uncond,
        device=device,
        seed=seed,
    )
    gen.model.load_state_dict(ckpt["model_state"])
    gen.ema_state = ckpt.get("ema_state", None)
    return gen


def ddpm_stability(
    ckpt_path: Path,
    seed: int,
    device: str,
    num_classes: int,
    n: int = 200,
    latent_delta: float = 0.05,
    ddim_steps: int = 50,
    guidance: float = 1.0,
    x0_clip: float = 6.0,
    batch: int = 50,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = load_ddpm_from_checkpoint(ckpt_path, device=device, default_num_classes=num_classes)
    mses = []
    done = 0
    while done < n:
        b = min(batch, n - done)
        labels = torch.tensor([(done + i) % gen.num_classes for i in range(b)], device=gen.device, dtype=torch.long)
        z = torch.randn(b, gen.seq_len, device=gen.device)
        delta = torch.randn_like(z) * latent_delta
        x1 = ddim_from_initial(gen, z, labels, ddim_steps=ddim_steps, guidance=guidance, x0_clip=x0_clip)
        x2 = ddim_from_initial(gen, z + delta, labels, ddim_steps=ddim_steps, guidance=guidance, x0_clip=x0_clip)
        mses.extend(torch.mean((x1 - x2) ** 2, dim=1).detach().cpu().numpy().tolist())
        done += b
    return float(np.mean(mses))


def parse_args():
    p = argparse.ArgumentParser(description="Re-evaluate existing DDPM samples with MMD-based distributional fidelity")
    p.add_argument("--dataset", choices=["mitbih", "svdb"], required=True)
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--prepared_root", type=str, default=None)
    p.add_argument("--generated_template", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--seeds", type=str, default="19,88,123")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mmd_sample_per_class", type=int, default=500)
    p.add_argument("--diversity_sample", type=int, default=500)
    p.add_argument("--diversity_pairs", type=int, default=2000)
    p.add_argument("--rf_estimators", type=int, default=50)
    p.add_argument("--amax", type=float, default=5.0)
    p.add_argument("--dmax", type=float, default=4.0)
    p.add_argument("--robust_n", type=int, default=200)
    p.add_argument("--robust_batch", type=int, default=50)
    p.add_argument("--latent_delta", type=float, default=0.05)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--x0_clip", type=float, default=6.0)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.project_root)
    if args.dataset == "mitbih":
        class_names = ["N", "S", "V", "F", "Q"]
        default_gen = root / "outputs" / "ddpm_mitbih_seed{seed}"
    else:
        class_names = ["N", "S", "V"]
        default_gen = root / "outputs" / "ddpm_svdb_seed{seed}"
    labels = list(range(len(class_names)))
    prepared_root = Path(args.prepared_root) if args.prepared_root else root / "data" / "prepared" / args.dataset
    generated_template = args.generated_template or str(default_gen)
    out_dir = Path(args.out_dir) if args.out_dir else root / "outputs" / f"ddpm_{args.dataset}_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    mmd_rows_all: List[Dict] = []
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        prepared = prepared_root / f"seed_{seed}"
        gen_dir = Path(generated_template.format(seed=seed))
        X_train = np.load(prepared / "X_train.npy").astype(np.float32)
        y_train = np.load(prepared / "y_train.npy").astype(np.int64)
        X_test = np.load(prepared / "X_test.npy").astype(np.float32)
        y_test = np.load(prepared / "y_test.npy").astype(np.int64)
        X_gen = np.load(gen_dir / f"ddpm_X_syn_seed{seed}.npy").astype(np.float32)
        y_gen = np.load(gen_dir / f"ddpm_y_syn_seed{seed}.npy").astype(np.int64)
        ckpt = gen_dir / f"ddpm_seed{seed}.pt"

        gamma_by_class = classwise_rbf_gammas(
            X_train, y_train, labels, seed=seed,
            max_samples_each=args.mmd_sample_per_class,
        )
        print("fixed MMD gammas:", {class_names[c]: round(gamma_by_class[c]["Gamma"], 8) for c in labels})

        mmd_macro, mmd_rows = classwise_mmd2(
            X_gen, y_gen, X_test, y_test,
            class_labels=labels,
            seed=seed,
            gamma_by_class=gamma_by_class,
            max_samples_each=args.mmd_sample_per_class,
        )
        for r in mmd_rows:
            r.update({"Model": "DDPM", "Seed": seed})
        mmd_rows_all.extend(mmd_rows)

        tstr = tstr_metrics(
            X_gen, y_gen, X_test, y_test,
            class_labels=labels,
            seed=seed,
            rf_estimators=args.rf_estimators,
        )
        if ckpt.exists():
            stability = ddpm_stability(
                ckpt, seed, args.device, len(labels),
                n=args.robust_n,
                latent_delta=args.latent_delta,
                ddim_steps=args.ddim_steps,
                guidance=args.guidance,
                x0_clip=args.x0_clip,
                batch=args.robust_batch,
            )
        else:
            print(f"WARNING: checkpoint missing for seed {seed}: {ckpt}; StabilityError=NaN")
            stability = float("nan")

        row = {
            "Model": "DDPM",
            "Seed": seed,
            "DistributionalMMD2": mmd_macro,
            "Diversity": diversity_score(X_gen, seed, args.diversity_sample, args.diversity_pairs),
            "StabilityError": stability,
            "UtilityAccuracy": tstr["UtilityAccuracy"],
            "MacroF1": tstr["MacroF1"],
            "ClassRecallBalance": tstr["ClassRecallBalance"],
            "PhysiologicalValidity": physiological_validity(X_gen, args.amax, args.dmax),
        }
        rows.append(row)
        print(seed, row)

    metrics = [
        "DistributionalMMD2", "Diversity", "StabilityError", "UtilityAccuracy",
        "MacroF1", "ClassRecallBalance", "PhysiologicalValidity",
    ]
    write_csv(out_dir / f"ddpm_{args.dataset}_seed_results.csv", rows)
    write_csv(out_dir / f"ddpm_{args.dataset}_classwise_mmd.csv", mmd_rows_all)
    write_csv(out_dir / f"ddpm_{args.dataset}_summary.csv", summarize_by_model(rows, metrics))
    (out_dir / "evaluation_metadata.json").write_text(json.dumps({
        "dataset": args.dataset,
        "class_names": class_names,
        "prepared_root": str(prepared_root),
        "generated_template": generated_template,
        "mmd": "macro-average class-wise biased empirical RBF-MMD^2 against held-out real beats; per-class gamma fixed from real training data only and shared across generators",
    }, indent=2), encoding="utf-8")
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
