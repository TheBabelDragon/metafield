#!/usr/bin/env python3
"""
memory.py

Episodic memory with interestingness and prioritization.

Works together with attractors.py:
  Field → EpisodicMemory (experiences) → AttractorDynamics (landscape) → Geometry
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
        self.interestingness = 0.0
        self.replay_count = 0

    def compute_interestingness(self, prediction_error: float = 0.0,
                                reconstruction_error: float = 0.0) -> float:
        curv_term = abs(self.curvature)
        delta_term = abs(self.delta_h)
        pred_term = max(0.0, prediction_error)
        recon_term = max(0.0, reconstruction_error)

        self.interestingness = (
            1.5 * curv_term +
            1.0 * delta_term +
            2.0 * pred_term +
            1.5 * recon_term
        )
        return self.interestingness

    def update_priority(self, prediction_error: float = 0.0,
                        reconstruction_error: float = 0.0):
        self.compute_interestingness(prediction_error, reconstruction_error)
        self.priority = max(
            0.5,
            1.0 +
            1.2 * self.interestingness +
            0.8 * abs(self.curvature) +
            0.5 * abs(self.delta_h) +
            0.3 * self.replay_count
        )


class EpisodicMemory:
    """Prioritized episodic buffer. Feeds high-interest experiences into AttractorDynamics."""

    def __init__(self, max_size: int = 512,
                 base_exploration_rate: float = 0.15,
                 min_exploration: float = 0.05,
                 max_exploration: float = 0.35):
        self.buffer: List[EpisodicExperience] = []
        self.max_size = max_size
        self.base_exploration_rate = base_exploration_rate
        self.min_exploration = min_exploration
        self.max_exploration = max_exploration

    def add(self, exp: EpisodicExperience) -> None:
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda e: e.priority)
            self.buffer.pop(0)

    def _current_exploration_rate(self) -> float:
        if not self.buffer:
            return self.base_exploration_rate
        avg_interest = sum(e.interestingness for e in self.buffer) / len(self.buffer)
        rate = self.base_exploration_rate * (1.5 / (1.0 + avg_interest))
        return max(self.min_exploration, min(self.max_exploration, rate))

    def sample(self, n: int = 24) -> List[EpisodicExperience]:
        if len(self.buffer) == 0:
            return []
        if len(self.buffer) <= n:
            return list(self.buffer)

        exploration_rate = self._current_exploration_rate()
        n_explore = max(1, int(n * exploration_rate))
        n_biased = n - n_explore

        priorities = torch.tensor([e.priority for e in self.buffer], dtype=torch.float32)
        sharpened = priorities ** 1.5
        probs = sharpened / sharpened.sum()
        biased_indices = torch.multinomial(probs, n_biased, replacement=True)
        explore_indices = torch.randint(0, len(self.buffer), (n_explore,))

        all_indices = torch.cat([biased_indices, explore_indices])
        sampled = [self.buffer[i] for i in all_indices]

        for exp in sampled:
            exp.replay_count += 1

        return sampled

    def get_stats(self) -> Dict[str, Any]:
        if not self.buffer:
            return {
                "size": 0,
                "max_size": self.max_size,
                "avg_priority": 0.0,
                "avg_interestingness": 0.0,
                "exploration_rate": self.base_exploration_rate,
            }
        priorities = [e.priority for e in self.buffer]
        interestingnesses = [e.interestingness for e in self.buffer]
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "avg_priority": sum(priorities) / len(priorities),
            "avg_interestingness": sum(interestingnesses) / len(interestingnesses),
            "exploration_rate": self._current_exploration_rate(),
        }

    def __len__(self) -> int:
        return len(self.buffer)
