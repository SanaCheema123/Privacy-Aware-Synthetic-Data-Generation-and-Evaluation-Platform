"""TabDDPM style diffusion generator for tabular data (PyTorch).

Design (following Kotelnikov et al., 2023, "TabDDPM: Modelling Tabular Data with Diffusion Models"):
  * numerical columns are mapped to a standard normal space with a quantile transform
  * categorical columns are one hot encoded
  * an MLP denoiser with sinusoidal time embedding predicts the noise (epsilon)
  * the model is class conditional on the target column, so class imbalance can be
    handled natively by asking for any number of rows per class
  * cosine noise schedule, ancestral DDPM sampling; one hot blocks are decoded by argmax

Note: the original paper uses multinomial diffusion for categorical columns. This
implementation uses Gaussian diffusion over the one hot block, which is a widely used
simplification. Swap in synthcity's `tabddpm` plugin if the exact formulation is required.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data.schema import Schema
from ..data.transformer import TabularTransformer
from .base import BaseGenerator

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _cosine_betas(T: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(T + 1, dtype=np.float64) / T
    alphas_bar = np.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return np.clip(betas, 1e-5, 0.999)


if nn is not None:

    class TimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        def forward(self, t):
            half = self.dim // 2
            freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
            args = t.float()[:, None] * freqs[None]
            emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            return self.mlp(emb)

    class Denoiser(nn.Module):
        def __init__(self, d_in: int, n_classes: int, hidden: int = 512, layers: int = 4, dropout: float = 0.0):
            super().__init__()
            self.proj = nn.Linear(d_in, hidden)
            self.t_emb = TimeEmbedding(hidden)
            self.y_emb = nn.Embedding(max(n_classes, 1) + 1, hidden)  # last index = unconditional
            blocks = []
            for _ in range(layers):
                blocks += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
            self.body = nn.Sequential(*blocks)
            self.out = nn.Linear(hidden, d_in)

        def forward(self, x, t, y):
            h = self.proj(x) + self.t_emb(t) + self.y_emb(y)
            return self.out(self.body(h))


class TabDDPMGenerator(BaseGenerator):
    name, display_name = "tabddpm", "TabDDPM"
    supports_native_conditioning = True

    def _fit(self, df: pd.DataFrame, schema: Schema) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for TabDDPM. Install it with: pip install torch")
        p = self.params
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.columns_ = list(df.columns)

        self.cond_col_ = schema.target if (schema.target in df.columns and schema.task == "classification") else None
        if self.cond_col_:
            y_raw = df[self.cond_col_].astype(str)
            self.classes_ = sorted(y_raw.unique().tolist())
            self.class_probs_ = y_raw.value_counts(normalize=True).reindex(self.classes_).to_numpy()
            y = y_raw.map({c: i for i, c in enumerate(self.classes_)}).to_numpy()
        else:
            self.classes_, self.class_probs_ = [], np.array([1.0])
            y = np.zeros(len(df), dtype=int)

        feats = [c for c in df.columns if c != self.cond_col_]
        num = [c for c in feats if c in schema.numerical]
        cat = [c for c in feats if c in schema.categorical]
        self.tf_ = TabularTransformer(num, cat, mode="quantile", seed=self.seed).fit(df)
        X = self.tf_.transform(df)
        self.clip_ = float(np.abs(X).max()) + 0.1  # data range in transformed space

        T = int(p.get("num_timesteps", 500))
        self.T_ = T
        betas = _cosine_betas(T)
        alphas = 1.0 - betas
        self.betas_ = torch.tensor(betas, dtype=torch.float32)
        self.alphas_ = torch.tensor(alphas, dtype=torch.float32)
        self.alphas_bar_ = torch.tensor(np.cumprod(alphas), dtype=torch.float32)

        n_classes = max(len(self.classes_), 1)
        self.uncond_idx_ = n_classes
        self.model_ = Denoiser(X.shape[1], n_classes, int(p.get("hidden_dim", 512)),
                               int(p.get("num_layers", 4)), float(p.get("dropout", 0.0))).to(self.device_)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=float(p.get("lr", 1e-3)), weight_decay=1e-5)
        steps = int(p.get("steps", 4000))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        Xt = torch.tensor(X, dtype=torch.float32, device=self.device_)
        Yt = torch.tensor(y, dtype=torch.long, device=self.device_)
        ab = self.alphas_bar_.to(self.device_)
        bs = min(int(p.get("batch_size", 512)), len(Xt))
        self.losses_ = []
        self.model_.train()
        for step in range(steps):
            idx = torch.randint(0, len(Xt), (bs,), device=self.device_)
            x0, yb = Xt[idx], Yt[idx].clone()
            # 10% label dropout keeps an unconditional mode available
            drop = torch.rand(bs, device=self.device_) < 0.1
            yb[drop] = self.uncond_idx_
            t = torch.randint(0, T, (bs,), device=self.device_)
            noise = torch.randn_like(x0)
            a = ab[t].unsqueeze(1)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
            loss = ((self.model_(xt, t, yb) - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 50 == 0:
                self.losses_.append({"step": step, "loss": float(loss.item())})
        self.model_.eval()
        self.model_.cpu()
        self.device_ = "cpu"

    @torch.no_grad() if torch is not None else (lambda f: f)
    def _generate(self, y: np.ndarray) -> pd.DataFrame:
        n = len(y)
        d = self.tf_.output_dim_
        out = []
        g = torch.Generator().manual_seed(self.seed + n)
        for start in range(0, n, 4096):
            yb = torch.tensor(y[start:start + 4096], dtype=torch.long)
            x = torch.randn(len(yb), d, generator=g)
            for t in reversed(range(self.T_)):
                tt = torch.full((len(yb),), t, dtype=torch.long)
                eps = self.model_(x, tt, yb)
                a, ab, b = self.alphas_[t], self.alphas_bar_[t], self.betas_[t]
                ab_prev = self.alphas_bar_[t - 1] if t > 0 else torch.tensor(1.0)
                # predict x0, clip it to the data range, then use the posterior q(x_{t-1} | x_t, x0)
                x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-self.clip_, self.clip_)
                mean = (ab_prev.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ab_prev) / (1 - ab)) * x
                if t > 0:
                    var = b * (1 - ab_prev) / (1 - ab)
                    x = mean + var.sqrt() * torch.randn(x.shape, generator=g)
                else:
                    x = mean
            out.append(x.numpy())
        X = np.vstack(out)
        X[:, :len(self.tf_.numerical)] = np.clip(X[:, :len(self.tf_.numerical)], -5.5, 5.5)
        df = self.tf_.inverse_transform(X)
        if self.cond_col_:
            df[self.cond_col_] = np.array(self.classes_, dtype=object)[y]
        return df[self.columns_]

    def sample(self, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + int(n))
        if self.cond_col_:
            y = rng.choice(len(self.classes_), size=int(n), p=self.class_probs_)
        else:
            y = np.zeros(int(n), dtype=int)
        return self._generate(y)

    def sample_conditional(self, target: str, class_counts: dict[str, int], max_rounds: int = 30) -> pd.DataFrame:
        if target != self.cond_col_:
            return super().sample_conditional(target, class_counts, max_rounds)
        lookup = {c: i for i, c in enumerate(self.classes_)}
        y = np.concatenate([np.full(int(k), lookup[str(c)]) for c, k in class_counts.items() if k > 0])
        np.random.default_rng(self.seed).shuffle(y)
        return self._generate(y)

    def loss_history(self) -> pd.DataFrame | None:
        return pd.DataFrame(getattr(self, "losses_", []))
