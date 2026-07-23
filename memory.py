#!/usr/bin/env python3
"""
memory.py

Episodic memory system with prioritization and interestingness.

Supports adaptive exploration and multi-signal interestingness
(curvature + ΔH + prediction error + reconstruction error).
"""

from typing import List, Dict, Any, Optional

import torch


class EpisodicExperience:
    """
    A single stored experience in episodic memory.

    Contains the latent representation, action, ΔH, acceptance,
    optional curvature, and an interestingness score used for
    prioritization and early emergence signals.
    """

    def __init__(self, latent: torch.Tensor, action: float, delta_h: float,
                 accepted: bool, curvature: float = 0.0):
        self.latent = latent.detach().cpu().clone()
        self.action = float(action)
        self.delta_h = float(delta_h)
        self.accepted = bool(accepted)
        self.curvature = float(curvature)
        self.priority = 1.0
        self.interestingness = 0.0

    def compute_interestingness(self, prediction_error: float = 0.0,
                                reconstruction_error: float = 0.0) -> float:
        """
        Compute a multi-signal interestingness / curiosity score.

        Combines:
        - Absolute curvature (geometric novelty)
        - Absolute ΔH (energetic surprise)
        - Prediction error (model surprise)
        - Reconstruction error (how hard the configuration is for the autoencoder)
        """
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
        """Update priority using the full interestingness signal."""
        self.compute_interestingness(prediction_error, reconstruction_error)

        self.priority = max(
            0.5,
            1.0 +
            1.2 * self.interestingness +
            0.8 * abs(self.curvature) +
            0.5 * abs(self.delta_h)
        )


class EpisodicMemory:
    """
    Prioritized episodic memory buffer with interestingness support.

    Features:
    - Multi-signal interestingness
    - Adaptive exploration rate driven by average interestingness
    - Basic associative retrieval
    - Rich stats for observability / Aurora sensing
    """

    def __init__(self, max_size: int = 256,
                 base_exploration_rate: float = 0.15,
                 min_exploration: float = 0.05,
                 max_exploration: float = 0.35):
        self.buffer: List[EpisodicExperience] = []
        self.max_size = max_size
        self.base_exploration_rate = base_exploration_rate
        self.min_exploration = min_exploration
        self.max_exploration = max_exploration

    def add(self, exp: EpisodicExperience) -> None:
        """Add a new experience to memory."""
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            self.buffer.sort(key=lambda e: e.priority)
            self.buffer.pop(0)

    def _current_exploration_rate(self) -> float:
        """
        Adaptive exploration rate based on average interestingness.

        Low average interestingness → higher exploration (seek novelty)
        High average interestingness → lower exploration (exploit interesting regions)
        """
        if not self.buffer:
            return self.base_exploration_rate

        avg_interest = sum(e.interestingness for e in self.buffer) / len(self.buffer)
        rate = self.base_exploration_rate * (1.5 / (1.0 + avg_interest))
        return max(self.min_exploration, min(self.max_exploration, rate))

    def sample(self, n: int = 16) -> List[EpisodicExperience]:
        """Sample n experiences with adaptive interestingness bias + exploration."""
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
        return [self.buffer[i] for i in all_indices]

    def find_similar(self, query_latent: torch.Tensor, k: int = 5) -> List[EpisodicExperience]:
        """Return the k experiences closest in latent space to the query."""
        if not self.buffer:
            return []

        query = query_latent.detach().cpu().flatten()
        distances = []
        for i, exp in enumerate(self.buffer):
            dist = torch.norm(exp.latent.flatten() - query).item()
            distances.append((dist, i))

        distances.sort(key=lambda x: x[0])
        return [self.buffer[i] for _, i in distances[:k]]

    def get_stats(self) -> Dict[str, Any]:
        """Return rich statistics for monitoring / sensing / emergence tracking."""
        if not self.buffer:
            return {
                "size": 0,
                "avg_priority": 0.0,
                "max_priority": 0.0,
                "avg_interestingness": 0.0,
                "max_interestingness": 0.0,
                "exploration_rate": self.base_exploration_rate,
            }

        priorities = [e.priority for e in self.buffer]
        interestingnesses = [e.interestingness for e in self.buffer]

        return {
            "size": len(self.buffer),
            "avg_priority": sum(priorities) / len(priorities),
            "max_priority": max(priorities),
            "avg_interestingness": sum(interestingnesses) / len(interestingnesses),
            "max_interestingness": max(interestingnesses),
            "exploration_rate": self._current_exploration_rate(),
        }

    def __len__(self) -> int:
        return len(self.buffer)
