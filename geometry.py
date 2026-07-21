#!/usr/bin/env python3
"""
geometry.py

Learned Information Geometry.

Autoencoder + Fisher geometry on field configurations.
Designed to eventually serve as (or be wrapped as) an Aurora sensing mod.
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class _MLP(nn.Module):
    """Internal MLP helper. Not part of the public API."""
    def __init__(self, dims, dtype=torch.float64):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], dtype=dtype))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LearnedInformationGeometry:
    """
    Autoencoder-based learned geometry on field configurations.

    Provides encoding, reconstruction, Fisher metric, and scalar curvature.
    Intended to be used as a feature/sensing component in larger systems
    (including future Aurora integration).
    """

    def __init__(self, input_dim: int, latent_dim: int = 8,
                 hidden_dims: tuple = (256, 128), sigma: float = 1.0, lr: float = 3e-4):
        self.latent_dim = latent_dim
        self.sigma = sigma
        dtype = torch.float64

        enc_dims = [input_dim, *hidden_dims, latent_dim]
        dec_dims = [latent_dim, *reversed(hidden_dims), input_dim]

        self.encoder = _MLP(enc_dims, dtype=dtype)
        self.decoder = _MLP(dec_dims, dtype=dtype)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.last_loss = None

    def train_on_batch(self, samples: List[torch.Tensor], epochs: int = 30) -> float:
        """Train the autoencoder on a list of field samples."""
        if not samples:
            return 0.0
        x = torch.stack([s.to(torch.float64) for s in samples if s.shape[0] == samples[0].shape[0]])
        if len(x) < 4:
            return 0.0
        for _ in range(epochs):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = torch.mean((x_hat - x) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.item())
        return self.last_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a field configuration into latent space."""
        return self.encoder(x.to(torch.float64))

    def get_latent_2d(self, samples: List[torch.Tensor]) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        """Return 2D projection of latents (for visualization)."""
        if not samples:
            return None, None
        x = torch.stack([s.to(torch.float64) for s in samples if s.shape[0] == samples[0].shape[0]])
        with torch.no_grad():
            z = self.encoder(x)
        if HAS_SKLEARN and z.shape[1] > 2:
            pca = PCA(n_components=2)
            z2d = pca.fit_transform(z.numpy())
        else:
            z2d = z[:, :2].numpy()
        colors = [float(torch.abs(s).mean()) for s in samples]
        return z2d, colors

    def fisher_metric(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Fisher information metric at latent point z."""
        J = torch.func.jacrev(self.decoder)(z)
        G = (J.T @ J) / (self.sigma ** 2)
        return G

    def scalar_curvature(self, z: torch.Tensor, n_points: int = 12, eps: float = 1e-3) -> float:
        """Approximate scalar curvature at/around latent point z."""
        G = self.fisher_metric(z)
        try:
            G_inv = torch.linalg.inv(G + eps * torch.eye(G.shape[0], dtype=G.dtype))
        except:
            return float('nan')

        dim = self.latent_dim
        total = 0.0
        for _ in range(n_points):
            z_point = z + torch.randn_like(z) * 0.05
            curv = 0.0
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        continue
                    step = torch.zeros(dim, dtype=z.dtype)
                    step[i] = eps
                    G_plus = self.fisher_metric(z_point + step)
                    G_minus = self.fisher_metric(z_point - step)
                    dG = (G_plus - G_minus) / (2 * eps)
                    curv += torch.trace(G_inv @ dG)
            total += curv
        return float(total / n_points)
