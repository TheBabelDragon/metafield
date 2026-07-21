#!/usr/bin/env python3
"""
memory.py

Episodic memory system.

This module is designed to be modular and integration-friendly
for future use as (or inside) an Aurora mod.
"""

from typing import List, Dict, Any

import torch


class EpisodicExperience:
    """
    A single stored experience in episodic memory.

    Contains the latent representation, action taken, outcome (ΔH),
    acceptance, and an optional curvature signal.
    """

    def __init__(self, latent: torch.Tensor, action: float, delta_h: float,
                 accepted: bool, curvature: float = 0.0):
        self.latent = latent.detach().cpu().clone()
        self.action = float(action)
        self.delta_h = float(delta_h)
        self.accepted = bool(accepted)
        self.curvature = float(curvature)
        self.priority = 1.0

    def update_priority(self, prediction_error: float = 0.0):
        """Update priority based on multiple signals."""
        self.priority = max(0.5,
                            1.0 +
                            2.0 * abs(self.curvature) +
                            1.5 * self.delta_h +
                            prediction_error * 0.1)


class EpisodicMemory:
    """
    Prioritized episodic memory buffer.

    Stores experiences and allows prioritized sampling.
    Includes `get_stats()` for easy observability (useful for Aurora sensing).
    """

    def __init__(self, max_size: int = 256):
        self.buffer: List[EpisodicExperience] = []
        self.max_size = max_size

    def add(self, exp: EpisodicExperience) -> None:
        """Add a new experience to memory."""
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda e: e.priority)
            self.buffer.pop(0)

    def sample(self, n: int = 16) -> List[EpisodicExperience]:
        """Sample n experiences (prioritized)."""
        if len(self.buffer) < n:
            return self.buffer
        priorities = torch.tensor([e.priority for e in self.buffer], dtype=torch.float32)
        probs = priorities / priorities.sum()
        indices = torch.multinomial(probs, n, replacement=True)
        return [self.buffer[i] for i in indices]

    def get_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics for monitoring / sensing.

        Returns:
            dict with keys: size, avg_priority, max_priority
        """
        if not self.buffer:
            return {"size": 0, "avg_priority": 0.0, "max_priority": 0.0}

        priorities = [e.priority for e in self.buffer]
        return {
            "size": len(self.buffer),
            "avg_priority": sum(priorities) / len(priorities),
            "max_priority": max(priorities),
        }

    def __len__(self) -> int:
        return len(self.buffer)
