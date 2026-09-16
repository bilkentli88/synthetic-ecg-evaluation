from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half <= 1:
            raise ValueError("Embedding dimension must be at least 4.")
        scale = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        if in_ch % groups != 0 or out_ch % groups != 0:
            raise ValueError(f"Channels must be divisible by groups={groups}.")
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, seq_len=256, base_ch=64, ch_mults=(1, 2, 4), num_classes=5, emb_dim=256, groups=8):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.null_idx = num_classes
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim), nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        chs = [base_ch * m for m in ch_mults]
        self.in_conv = nn.Conv1d(1, base_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = base_ch
        for c in chs:
            self.down_blocks.append(ResBlock1D(prev, c, emb_dim, groups))
            self.downsamples.append(nn.Conv1d(c, c, 4, stride=2, padding=1))
            prev = c
        self.mid = ResBlock1D(prev, prev, emb_dim, groups)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for c in reversed(chs):
            self.upsamples.append(nn.ConvTranspose1d(prev, c, 4, stride=2, padding=1))
            self.up_blocks.append(ResBlock1D(c * 2, c, emb_dim, groups))
            prev = c
        self.out_norm = nn.GroupNorm(groups, prev)
        self.out_conv = nn.Conv1d(prev, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x[:, None, :]
        emb = self.time_mlp(t) + self.class_emb(y)
        h = self.in_conv(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, emb)
        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block(h, emb)
        return self.out_conv(F.silu(self.out_norm(h)))[:, 0, :]


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-4, 0.999)


class ECGDiffusion:
    """1-D class-conditional DDPM architecture used by the saved experiment checkpoints."""

    def __init__(self, seq_len=256, num_classes=5, T=300, base_ch=64, ch_mults=(1, 2, 4),
                 emb_dim=256, p_uncond=0.1, device=None, seed=None, deterministic=True):
        if seed is not None:
            set_seed(seed, deterministic)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.T = T
        self.p_uncond = p_uncond
        self.device = torch.device(device)
        self.model = UNet1D(seq_len, base_ch, ch_mults, num_classes, emb_dim).to(self.device)
        self.betas = cosine_beta_schedule(T).to(self.device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.acp_prev = F.pad(self.acp[:-1], (1, 0), value=1.0)
        self.ema_state = None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _make_loader(self, X, y, batch_size, balanced=True, num_workers=0):
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        if balanced:
            counts = torch.bincount(y_t, minlength=self.num_classes).float()
            counts[counts == 0] = 1.0
            weights = 1.0 / counts[y_t]
            sampler = torch.utils.data.WeightedRandomSampler(weights.double(), len(weights), replacement=True)
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True,
                                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                           num_workers=num_workers, pin_memory=torch.cuda.is_available())

    def fit(self, X, y, epochs=120, batch_size=64, lr=2e-4, seed=19, ema_decay=0.999,
            balanced=True, amp=True, verbose=True):
        set_seed(seed)
        dl = self._make_loader(X, y, batch_size, balanced)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=(amp and self.device.type == "cuda"))
        ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss = total_seen = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                B = xb.size(0)
                t = torch.randint(0, self.T, (B,), device=self.device)
                y_in = yb.clone()
                y_in[torch.rand(B, device=self.device) < self.p_uncond] = self.model.null_idx
                noise = torch.randn_like(xb)
                ac = self.acp[t][:, None]
                x_t = ac.sqrt() * xb + (1 - ac).sqrt() * noise
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(amp and self.device.type == "cuda")):
                    loss = F.mse_loss(self.model(x_t, t, y_in), noise)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                with torch.no_grad():
                    for k, v in self.model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                        else:
                            ema[k] = v.detach().clone()
                total_loss += loss.item() * B; total_seen += B
            if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
                print(f"[seed {seed}] epoch {ep:03d}/{epochs} loss={total_loss/max(total_seen,1):.5f}")
        self.ema_state = ema
        return self

    @torch.no_grad()
    def _sample_batch(self, y, ddim_steps=50, guidance=1.0, use_ema=True, x0_clip=6.0):
        backup = None
        if use_ema and self.ema_state is not None:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema_state)
        self.model.eval()
        x = torch.randn(y.size(0), self.seq_len, device=self.device)
        step_idx = torch.linspace(self.T - 1, 0, ddim_steps, device=self.device).long()
        for i, t_cur_tensor in enumerate(step_idx):
            t_cur = int(t_cur_tensor.item())
            t_batch = torch.full((y.size(0),), t_cur, device=self.device, dtype=torch.long)
            ac_t = self.acp[t_cur]
            if i + 1 < len(step_idx):
                t_next = int(step_idx[i + 1].item())
                ac_next = self.acp[t_next] if t_next > 0 else torch.tensor(1.0, device=self.device)
            else:
                ac_next = torch.tensor(1.0, device=self.device)
            eps_c = self.model(x, t_batch, y)
            if guidance is not None and guidance != 1.0:
                eps_u = self.model(x, t_batch, torch.full_like(y, self.model.null_idx))
                eps = eps_u + guidance * (eps_c - eps_u)
            else:
                eps = eps_c
            x0 = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
            if x0_clip is not None and x0_clip > 0:
                x0 = x0.clamp(-x0_clip, x0_clip)
            x = ac_next.sqrt() * x0 + (1 - ac_next).sqrt() * eps
        if backup is not None:
            self.model.load_state_dict(backup)
        return x.detach().cpu().numpy().astype(np.float32)

    def sample_per_class(self, counts: Dict[int, int], ddim_steps=50, guidance=1.0, batch=512,
                         use_ema=True, x0_clip=6.0) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for cls in sorted(counts):
            done, n = 0, int(counts[cls])
            while done < n:
                b = min(batch, n - done)
                y = torch.full((b,), int(cls), device=self.device, dtype=torch.long)
                Xs.append(self._sample_batch(y, ddim_steps, guidance, use_ema, x0_clip))
                ys.append(np.full(b, int(cls), dtype=np.int64))
                done += b
        return np.concatenate(Xs), np.concatenate(ys)

    def save_checkpoint(self, path: str | Path, extra: Optional[dict] = None) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "ema_state": self.ema_state,
                    "seq_len": self.seq_len, "num_classes": self.num_classes, "T": self.T,
                    "p_uncond": self.p_uncond, "extra": extra or {}}, path)
