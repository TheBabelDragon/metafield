#!/usr/bin/env python3
"""
memory.py

Episodic memory system with prioritization and interestingness.

Designed for modularity and future Aurora integration.
This version supports stronger emergence signals via interestingness-biased sampling.
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

    def compute_interestingness(self, prediction_error: float = 0.0) -> float:
        """
        Compute a simple interestingness / curiosity signal.

        Combines:
        - Absolute curvature (geometric novelty)
        - Absolute ΔH (energetic surprise)
        - Prediction error (model surprise)
        """
        curv_term = abs(self.curvature)
        delta_term = abs(self.delta_h)
        pred_term = max(0.0, prediction_error)

        # Weighted combination — tunable later
        self.interestingness = (
            1.5 * curv_term +
            1.0 * delta_term +
            2.0 * pred_term
        )
        return self.interestingness

    def update_priority(self, prediction_error: float = 0.0):
        """Update priority using interestingness and other signals."""
        self.compute_interestingness(prediction_error)

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

    Supports strongly interestingness-biased sampling (with a small amount of
    pure random exploration) and basic associative retrieval.
    Includes rich `get_stats()` for observability and Aurora sensing.
    """

    def __init__(self, max_size: int = 256, exploration_rate: float = 0.15):
        self.buffer: List[EpisodicExperience] = []
        self.max_size = max_size
        # Fraction of samples drawn uniformly at random (encourages exploration)
        self.exploration_rate = exploration_rate

    def add(self, exp: EpisodicExperience) -> None:
        """Add a new experience to memory."""
        self.buffer.append(exp)
        if len(self.buffer) > self.max_size:
            # Evict lowest priority experiences
            self.buffer.sort(key=lambda e: e.priority)
            self.buffer.pop(0)

    def sample(self, n: int = 16) -> List[EpisodicExperience]:
        """
        Sample n experiences with strong interestingness bias + exploration.

        - Most samples are drawn according to priority (which is driven by interestingness)
        - A small fraction (`exploration_rate`) are drawn uniformly at random
          so the system does not get stuck only replaying the same high-interest items.
        """
        if len(self.buffer) == 0:
            return []
        if len(self.buffer) <= n:
            return list(self.buffer)

        # Decide how many samples come from pure exploration
        n_explore = max(1, int(n * self.exploration_rate))
        n_biased = n - n_explore

        # --- Biased sampling (interestingness / priority driven) ---
        priorities = torch.tensor([e.priority for e in self.buffer], dtype=torch.float32)
        # Sharpen the distribution a bit so high-interest items dominate more
        sharpened = priorities ** 1.5
        probs = sharpened / sharpened.sum()
        biased_indices = torch.multinomial(probs, n_biased, replacement=True)

        # --- Pure exploration (uniform random) ---
        explore_indices = torch.randint(0, len(self.buffer), (n_explore,))

        all_indices = torch.cat([biased_indices, explore_indices])
        return [self.buffer[i] for i in all_indices]

    def find_similar(self, query_latent: torch.Tensor, k: int = 5) -> List[EpisodicExperience]:
        """
        Simple associative retrieval: return the k experiences whose
        latents are closest (by L2 distance) to the query.
        """
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
        """
        Return rich statistics for monitoring / sensing / emergence tracking.

        Returns:
            dict with size, avg_priority, max_priority,
            avg_interestingness, max_interestingness
        """
        if not self.buffer:
            return {
                "size": 0,
                "avg_priority": 0.0,
                "max_priority": 0.0,
                "avg_interestingness": 0.0,
                "max_interestingness": 0.0,
            }

        priorities = [e.priority for e in self.buffer]
        interestingnesses = [e.interestingness for e in self.buffer]

        return {
            "size": len(self.buffer),
            "avg_priority": sum(priorities) / len(priorities),
            "max_priority": max(priorities),
            "avg_interestingness": sum(interestingnesses) / len(interestingnesses),
            "max_interestingness": max(interestingnesses),
        }

    def __len__(self) -> int:
        return len(self.buffer)
