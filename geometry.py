#!/usr/bin/env python3
"""
geometry.py

Learned Information Geometry with support for Persistent Attractor Memory.

The autoencoder can now be gently influenced by attractor centers formed
from heavily replayed high-interestingness experiences. This begins the
transition from "memory as stored states" to "memory as deformation of
the manifold".

v1.51: practical geometry diagnostics (cheap Fisher signals + optional curvature).
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
    """Internal MLP helper. Not part of the public API."""
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

    Supports a soft attractor influence: high-strength attractors gently
    pull the latent space so those regions become more stable basins.
    """

    def __init__(self, input_dim: int, latent_dim: int = 8,
                 hidden_dims: tuple = (256, 128), sigma: float = 1.0, lr: float = 3e-4,
                 attractor_weight: float = 0.4):
        self.latent_dim = latent_dim
        self.sigma = sigma
        self.attractor_weight = attractor_weight
        dtype = torch.float64

        enc_dims = [input_dim, *hidden_dims, latent_dim]
        dec_dims = [latent_dim, *reversed(hidden_dims), input_dim]

        self.encoder = _MLP(enc_dims, dtype=dtype)
        self.decoder = _MLP(dec_dims, dtype=dtype)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.last_loss: Optional[float] = None
        self._curvature_failures = 0

    def train_on_batch(self, samples: List[torch.Tensor], epochs: int = 30,
                       attractors: Optional[List[Tuple[torch.Tensor, float]]] = None) -> float:
        """
        Train the autoencoder on field samples.

        If attractors are provided, a soft attraction loss is added that
        encourages lower reconstruction error near the attractor centers,
        weighted by their strength. This is the first form of
        "memory as deformation of the manifold".
        """
        if not samples:
            return 0.0
        x = torch.stack([s.to(torch.float64).reshape(-1) for s in samples
                         if s.numel() == samples[0].numel()])
        if len(x) < 4:
            return 0.0

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
                        weighted_recon = (weights * torch.mean((x_hat - x) ** 2, dim=1)).sum() / weights.sum()
                        attractor_loss = attractor_loss + weighted_recon

            loss = recon_loss + self.attractor_weight * attractor_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.item())

        return self.last_loss if self.last_loss is not None else 0.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float64).reshape(-1)
        # Ensure batch dim for Linear
        if x.dim() == 1:
            return self.encoder(x.unsqueeze(0)).squeeze(0)
        return self.encoder(x)

    def get_latent_2d(self, samples: List[torch.Tensor]):
        if not samples:
            return None, None
        x = torch.stack([s.to(torch.float64).reshape(-1) for s in samples
                         if s.numel() == samples[0].numel()])
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
        if z.dim() == 0 or z.numel() != self.latent_dim:
            raise ValueError(f"z must have shape ({self.latent_dim},), got {tuple(z.shape)}")

        def _decode(zz):
            return self.decoder(zz)

        J = torch.func.jacrev(_decode)(z)  # (output_dim, latent_dim)
        G = (J.transpose(-1, -2) @ J) / (self.sigma ** 2)
        # Symmetrize numerically
        G = 0.5 * (G + G.transpose(-1, -2))
        return G

    def geometry_diagnostics(self, z: torch.Tensor) -> Dict[str, float]:
        """
        Cheap, always-available geometric signals from the Fisher metric.

        These are the primary continuous-mode geometry outputs. Full scalar
        curvature is optional and more expensive (see scalar_curvature).
        """
        out: Dict[str, float] = {
            "train_loss": float(self.last_loss) if self.last_loss is not None else float("nan"),
            "metric_trace": float("nan"),
            "metric_logdet": float("nan"),
            "metric_condition": float("nan"),
        }
        try:
            with torch.no_grad():
                # We need grads through jacrev, so briefly enable
                z = z.detach().to(torch.float64).reshape(-1)
            G = self.fisher_metric(z)
            # Eigenvalues of the metric
            evals = torch.linalg.eigvalsh(G + 1e-8 * torch.eye(G.shape[0], dtype=G.dtype))
            evals = evals.clamp_min(1e-12)
            out["metric_trace"] = float(evals.sum().item())
            out["metric_logdet"] = float(evals.log().sum().item())
            out["metric_condition"] = float((evals.max() / evals.min()).item())
        except Exception:
            pass
        return out

    def scalar_curvature(self, z: torch.Tensor, n_points: int = 4, eps: float = 1e-3) -> float:
        """
        Approximate scalar curvature at z.

        Intentionally kept light (small n_points) so continuous mode stays
        responsive. Prefer geometry_diagnostics() for routine monitoring.
        """
        try:
            z = z.detach().to(torch.float64).reshape(-1)
            G = self.fisher_metric(z)
            eye = torch.eye(G.shape[0], dtype=G.dtype)
            G_inv = torch.linalg.inv(G + eps * eye)

            dim = self.latent_dim
            total = 0.0
            # Use a fixed small set of random directions for stability
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
