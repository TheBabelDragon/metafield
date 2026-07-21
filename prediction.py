#!/usr/bin/env python3
"""
prediction.py

Latent space predictor.

Designed to be simple, swappable, and easy to integrate
as part of a larger Aurora mod or sensing system.
"""

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """
    Predicts a scalar value from a latent vector.

    Currently used to predict action values from latent field representations,
    but the architecture is intentionally simple and extensible.
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
