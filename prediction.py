#!/usr/bin/env python3
"""
prediction.py

LatentPredictor for MetaField.
Designed to be modular for future Aurora integration.
"""

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """
    Simple feedforward predictor from latent space to scalar value.
    Currently predicts action values, but designed to be extensible.
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)
