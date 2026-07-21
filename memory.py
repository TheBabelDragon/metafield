#!/usr/bin/env python3
"""
memory.py

Episodic memory system for MetaField.
Designed to be modular and suitable for future integration as an Aurora mod.
"""

from typing import List, Dict, Any

import torch


class EpisodicExperience:
    def __init__(self, latent: torch.Tensor, action: float, delta_h: float,
                 accepted: bool, curvature: float = 0.0):
        self.latent = latent.detach().cpu().clone()
        self.action = float(action)
        self.delta_h = float(delta_h)
        self.accepted = bool(accepted)
        self.curvature = float(curvature)
        self.priority = 1.0

    def update_priority(self, prediction_error: float = 0.0):
        self.priority = max(0.5,
                            1.0 +
                            2.0 * abs(self.curvature) +
                            1.5 * self.delta_h +
                            prediction_error * 0.1)


class EpisodicMemory:
    """
    Episodic memory with prioritization.

    This class is designed to be relatively self-contained so it can
    eventually be used as (or wrapped as) an Aurora mod.
    """

    def __init__(self, max_size: int = 256):
        self.buffer: List[EpisodicExperience] = []
        self.max_size = max_size

    def add(self, exp: EpisodicExperience):
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda e: e.priority)
            self.buffer.pop(0)

    def sample(self, n: int = 16):
        if len(self.buffer) < n:
            return self.buffer
        priorities = torch.tensor([e.priority for e in self.buffer], dtype=torch.float32)
        probs = priorities / priorities.sum()
        indices = torch.multinomial(probs, n, replacement=True)
        return [self.buffer[i] for i in indices]

    def get_stats(self) -> Dict[str, Any]:
        """
        Return basic observability statistics.
        Useful for Aurora sensing layer or external monitoring.
        """
        if not self.buffer:
            return {"size": 0, "avg_priority": 0.0, "max_priority": 0.0}

        priorities = [e.priority for e in self.buffer]
        return {
            "size": len(self.buffer),
            "avg_priority": sum(priorities) / len(priorities),
            "max_priority": max(priorities),
        }

    def __len__(self):
        return len(self.buffer)
