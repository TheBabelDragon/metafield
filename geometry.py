#!/usr/bin/env python3
"""
geometry.py

Learned Information Geometry with support for Persistent Attractor Memory.

v1.53: safer Fisher diagnostics (bounded cost), robust encode/train.
"""

from typing import List, Tuple, Optional, Any, Dict

import math
import torch
import torch.nn as nn

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import numpy as np
except ImportError:
    np = None


class _MLP(nn.Module):
    def __init__(self, dims, dtype=torch.float64):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], dtype=dtype))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LearnedInformationGeometry:
    """
    Autoencoder-based learned geometry on field configurations.

    Supports soft attractor influence so high-interestingness basins
    deform the manifold.
    """

    # Skip full Fisher jacrev if decoder output exceeds this (memory guard)
    MAX_FISHER_OUTPUT = 16384

    def __init__(self, input_dim: int, latent_dim: int = 8,
                 hidden_dims: tuple = (256, 128), sigma: float = 1.0, lr: float = 3e-4,
                 attractor_weight: float = 0.4):
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.sigma = float(sigma)
        self.attractor_weight = float(attractor_weight)
        dtype = torch.float64

        enc_dims = [self.input_dim, *hidden_dims, self.latent_dim]
        dec_dims = [self.latent_dim, *reversed(hidden_dims), self.input_dim]

        self.encoder = _MLP(enc_dims, dtype=dtype)
        self.decoder = _MLP(dec_dims, dtype=dtype)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.last_loss: Optional[float] = None
        self._curvature_failures = 0

    def train_on_batch(self, samples: List[torch.Tensor], epochs: int = 30,
                       attractors: Optional[List[Tuple[torch.Tensor, float]]] = None) -> float:
        if not samples:
            return 0.0
        try:
            x = torch.stack([
                s.to(torch.float64).reshape(-1) for s in samples
                if s.numel() == samples[0].numel()
            ])
        except Exception:
            return self.last_loss if self.last_loss is not None else 0.0

        if len(x) < 4:
            return 0.0
        if x.shape[1] != self.input_dim:
            # dimension drift — refuse rather than crash
            return self.last_loss if self.last_loss is not None else 0.0

        for _ in range(epochs):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            recon_loss = torch.mean((x_hat - x) ** 2)

            attractor_loss = torch.tensor(0.0, dtype=torch.float64)
            if attractors:
                for att_latent, strength in attractors:
                    att_latent = att_latent.to(torch.float64).reshape(-1)
                    if att_latent.shape[0] != z.shape[1]:
                        continue
                    dists = torch.norm(z - att_latent.unsqueeze(0), dim=1)
                    weights = torch.exp(-dists * 2.0) * float(strength)
                    if weights.sum() > 1e-8:
                        weighted_recon = (
                            weights * torch.mean((x_hat - x) ** 2, dim=1)
                        ).sum() / weights.sum()
                        attractor_loss = attractor_loss + weighted_recon

            loss = recon_loss + self.attractor_weight * attractor_loss
            if not torch.isfinite(loss):
                break

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()
            self.last_loss = float(loss.item())

        return self.last_loss if self.last_loss is not None else 0.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float64).reshape(-1)
        if x.numel() != self.input_dim:
            # pad or truncate defensively
            if x.numel() < self.input_dim:
                x = torch.nn.functional.pad(x, (0, self.input_dim - x.numel()))
            else:
                x = x[: self.input_dim]
        if x.dim() == 1:
            return self.encoder(x.unsqueeze(0)).squeeze(0)
        return self.encoder(x)

    def get_latent_2d(self, samples: List[torch.Tensor]):
        if not samples:
            return None, None
        x = torch.stack([
            s.to(torch.float64).reshape(-1) for s in samples
            if s.numel() == samples[0].numel()
        ])
        with torch.no_grad():
            z = self.encoder(x)
        if HAS_SKLEARN and z.shape[1] > 2 and np is not None:
            pca = PCA(n_components=2)
            z2d = pca.fit_transform(z.detach().cpu().numpy())
        else:
            z2d = z[:, :2].detach().cpu().numpy()
        colors = [float(torch.abs(s).mean()) for s in samples]
        return z2d, colors

    def fisher_metric(self, z: torch.Tensor) -> torch.Tensor:
        """Pullback Fisher metric G(z) = (1/σ²) Jᵀ J of the decoder."""
        z = z.to(torch.float64).reshape(-1)
        if z.numel() != self.latent_dim:
            raise ValueError(f"z must have shape ({self.latent_dim},), got {tuple(z.shape)}")

        if self.input_dim > self.MAX_FISHER_OUTPUT:
            # Memory guard: approximate metric via finite differences in latent space
            return self._fisher_metric_fd(z)

        def _decode(zz):
            return self.decoder(zz)

        J = torch.func.jacrev(_decode)(z)
        G = (J.transpose(-1, -2) @ J) / (self.sigma ** 2)
        G = 0.5 * (G + G.transpose(-1, -2))
        return G

    def _fisher_metric_fd(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Finite-difference approximation of pullback metric (cheaper for large output)."""
        dim = self.latent_dim
        G = torch.zeros(dim, dim, dtype=torch.float64)
        with torch.no_grad():
            y0 = self.decoder(z)
            for i in range(dim):
                step = torch.zeros(dim, dtype=torch.float64)
                step[i] = eps
                yp = self.decoder(z + step)
                ym = self.decoder(z - step)
                dy_i = (yp - ym) / (2 * eps)
                for j in range(i, dim):
                    stepj = torch.zeros(dim, dtype=torch.float64)
                    stepj[j] = eps
                    yp2 = self.decoder(z + stepj)
                    ym2 = self.decoder(z - stepj)
                    dy_j = (yp2 - ym2) / (2 * eps)
                    gij = float(torch.dot(dy_i, dy_j).item()) / (self.sigma ** 2)
                    G[i, j] = gij
                    G[j, i] = gij
        return G

    def geometry_diagnostics(self, z: torch.Tensor) -> Dict[str, float]:
        out: Dict[str, float] = {
            "train_loss": float(self.last_loss) if self.last_loss is not None else float("nan"),
            "metric_trace": float("nan"),
            "metric_logdet": float("nan"),
            "metric_condition": float("nan"),
        }
        try:
            z = z.detach().to(torch.float64).reshape(-1)
            G = self.fisher_metric(z)
            evals = torch.linalg.eigvalsh(G + 1e-8 * torch.eye(G.shape[0], dtype=G.dtype))
            evals = evals.clamp_min(1e-12)
            out["metric_trace"] = float(evals.sum().item())
            out["metric_logdet"] = float(evals.log().sum().item())
            out["metric_condition"] = float((evals.max() / evals.min()).item())
        except Exception:
            pass
        return out

    def scalar_curvature(self, z: torch.Tensor, n_points: int = 4, eps: float = 1e-3) -> float:
        try:
            z = z.detach().to(torch.float64).reshape(-1)
            G = self.fisher_metric(z)
            eye = torch.eye(G.shape[0], dtype=G.dtype)
            G_inv = torch.linalg.inv(G + eps * eye)

            dim = self.latent_dim
            total = 0.0
            gen = torch.Generator().manual_seed(0)
            for _ in range(n_points):
                z_point = z + torch.randn(dim, generator=gen, dtype=z.dtype) * 0.03
                curv = 0.0
                for i in range(dim):
                    step = torch.zeros(dim, dtype=z.dtype)
                    step[i] = eps
                    try:
                        G_plus = self.fisher_metric(z_point + step)
                        G_minus = self.fisher_metric(z_point - step)
                        dG = (G_plus - G_minus) / (2 * eps)
                        curv += float(torch.trace(G_inv @ dG).item())
                    except Exception:
                        continue
                total += curv
            return float(total / max(1, n_points))
        except Exception as e:
            self._curvature_failures += 1
            if self._curvature_failures <= 3:
                print(f"[geometry] scalar_curvature failed ({self._curvature_failures}): {type(e).__name__}: {e}")
            return float("nan")
