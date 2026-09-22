#!/usr/bin/env python3
"""
optical_encoder.py

Phase 13 — neural interpreter of the optical substrate.

  BPW34 detector vector (n_detectors,)
        ↓
  OpticalEncoder.encode
        ↓
  latent z (latent_dim,)
        ↓
  geometry / predictor / active_probe

The network interprets the physical body; it does not replace it.
Pure-numpy PCA fallback when torch is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class EncoderStats:
    n_samples: int = 0
    last_recon_mse: float = 0.0
    latent_dim: int = 0
    input_dim: int = 0


if _HAS_TORCH:

    class _OpticalAE(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int, hidden: int = 64):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, input_dim),
                nn.Sigmoid(),
            )

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.encoder(x)

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.decoder(z)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            z = self.encode(x)
            return z, self.decode(z)


class OpticalEncoder:
    """
    Fit an autoencoder (or PCA fallback) on stacks of detector vectors.

    Usage:
        enc = OpticalEncoder(n_detectors=20, latent_dim=8)
        enc.fit(list_of_vectors, epochs=40)
        z = enc.encode(vector)
        x_hat = enc.decode(z)
    """

    def __init__(
        self,
        n_detectors: int,
        latent_dim: int = 8,
        *,
        hidden: int = 64,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.n_detectors = int(n_detectors)
        self.latent_dim = int(min(latent_dim, n_detectors))
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.seed = int(seed)
        self.stats = EncoderStats(
            latent_dim=self.latent_dim, input_dim=self.n_detectors
        )
        self._backend = "none"
        self._model = None
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None

        if _HAS_TORCH:
            torch.manual_seed(self.seed)
            self._model = _OpticalAE(self.n_detectors, self.latent_dim, self.hidden)
            self._opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            self._backend = "torch"
        else:
            self._backend = "pca"

    @property
    def backend(self) -> str:
        return self._backend

    def fit(
        self,
        samples: Sequence[np.ndarray],
        *,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> EncoderStats:
        X = self._stack(samples)
        if X.shape[0] == 0:
            return self.stats

        if self._backend == "torch" and self._model is not None:
            return self._fit_torch(X, epochs=epochs, batch_size=batch_size)
        return self._fit_pca(X)

    def encode(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        if v.size != self.n_detectors:
            raise ValueError(f"expected {self.n_detectors} detectors, got {v.size}")
        if self._backend == "torch" and self._model is not None:
            self._model.eval()
            with torch.no_grad():
                t = torch.from_numpy(v).float().unsqueeze(0)
                z = self._model.encode(t).squeeze(0).numpy()
            return z.astype(np.float32)
        if self._pca_components is None or self._pca_mean is None:
            return np.zeros(self.latent_dim, dtype=np.float32)
        centered = v - self._pca_mean
        return (self._pca_components @ centered).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        lat = np.asarray(z, dtype=np.float32).reshape(-1)
        if lat.size != self.latent_dim:
            raise ValueError(f"expected latent_dim={self.latent_dim}, got {lat.size}")
        if self._backend == "torch" and self._model is not None:
            self._model.eval()
            with torch.no_grad():
                t = torch.from_numpy(lat).float().unsqueeze(0)
                x = self._model.decode(t).squeeze(0).numpy()
            return np.clip(x, 0.0, 1.0).astype(np.float32)
        if self._pca_components is None or self._pca_mean is None:
            return np.zeros(self.n_detectors, dtype=np.float32)
        recon = self._pca_components.T @ lat + self._pca_mean
        return np.clip(recon, 0.0, 1.0).astype(np.float32)

    def reconstruction_error(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        x_hat = self.decode(self.encode(x))
        return float(np.mean((x - x_hat) ** 2))

    def _stack(self, samples: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for s in samples:
            a = np.asarray(s, dtype=np.float32).reshape(-1)
            if a.size == self.n_detectors:
                rows.append(a)
        if not rows:
            return np.zeros((0, self.n_detectors), dtype=np.float32)
        return np.stack(rows, axis=0)

    def _fit_torch(
        self, X: np.ndarray, *, epochs: int, batch_size: int
    ) -> EncoderStats:
        assert self._model is not None
        self._model.train()
        data = torch.from_numpy(X).float()
        n = data.shape[0]
        last_mse = 0.0
        for _ in range(max(1, epochs)):
            perm = torch.randperm(n)
            total = 0.0
            count = 0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch = data[idx]
                self._opt.zero_grad()
                _, recon = self._model(batch)
                loss = torch.mean((recon - batch) ** 2)
                loss.backward()
                self._opt.step()
                total += float(loss.item()) * batch.shape[0]
                count += batch.shape[0]
            last_mse = total / max(count, 1)
        self.stats = EncoderStats(
            n_samples=n,
            last_recon_mse=last_mse,
            latent_dim=self.latent_dim,
            input_dim=self.n_detectors,
        )
        return self.stats

    def _fit_pca(self, X: np.ndarray) -> EncoderStats:
        mean = X.mean(axis=0)
        centered = X - mean
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            components = vt[: self.latent_dim]
        except np.linalg.LinAlgError:
            components = np.eye(self.latent_dim, self.n_detectors, dtype=np.float32)
        self._pca_mean = mean.astype(np.float32)
        self._pca_components = components.astype(np.float32)
        Z = centered @ components.T
        recon = Z @ components + mean
        mse = float(np.mean((X - recon) ** 2))
        self.stats = EncoderStats(
            n_samples=X.shape[0],
            last_recon_mse=mse,
            latent_dim=self.latent_dim,
            input_dim=self.n_detectors,
        )
        self._backend = "pca"
        return self.stats
