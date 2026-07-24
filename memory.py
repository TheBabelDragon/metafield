#!/usr/bin/env python3
"""
memory.py

Episodic memory with interestingness and prioritization.
Supports external drive-force scaling of exploration (e.g. from Aurora feed).
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
    """
    Prioritized episodic buffer with soft expandable capacity.
    exploration_scale can be driven by Aurora environment feed.
    """

    def __init__(self,
                 soft_capacity: int = 512,
                 soft_capacity_max: int = 8192,
                 absolute_safety_limit: int = 20000,
                 growth_every: int = 200,
                 growth_step: int = 128,
                 base_exploration_rate: float = 0.15,
                 min_exploration: float = 0.05,
                 max_exploration: float = 0.35):
        self.buffer: List[EpisodicExperience] = []
        self.soft_capacity = soft_capacity
        self.soft_capacity_max = soft_capacity_max
        self.absolute_safety_limit = absolute_safety_limit
        self.growth_every = growth_every
        self.growth_step = growth_step
        self.total_added = 0

        self.base_exploration_rate = base_exploration_rate
        self.min_exploration = min_exploration
        self.max_exploration = max_exploration
        self.exploration_scale = 1.0  # set by Aurora drive force

    def set_drive_scale(self, exploration_scale: float = 1.0):
        self.exploration_scale = max(0.4, min(1.8, float(exploration_scale)))

    def add(self, exp: EpisodicExperience) -> None:
        self.buffer.append(exp)
        self.total_added += 1

        if (self.total_added % self.growth_every == 0 and
                self.soft_capacity < self.soft_capacity_max):
            self.soft_capacity = min(
                self.soft_capacity_max,
                self.soft_capacity + self.growth_step
            )

        if len(self.buffer) > self.soft_capacity:
            overflow = len(self.buffer) - self.soft_capacity
            self.buffer.sort(key=lambda e: e.priority)
            self.buffer = self.buffer[overflow:]

        if len(self.buffer) > self.absolute_safety_limit:
            overflow = len(self.buffer) - self.absolute_safety_limit
            self.buffer.sort(key=lambda e: e.priority)
            self.buffer = self.buffer[overflow:]

    def _current_exploration_rate(self) -> float:
        if not self.buffer:
            return self.base_exploration_rate * self.exploration_scale
        avg_interest = sum(e.interestingness for e in self.buffer) / len(self.buffer)
        rate = self.base_exploration_rate * (1.5 / (1.0 + avg_interest))
        rate *= self.exploration_scale
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
                "soft_capacity": self.soft_capacity,
                "soft_capacity_max": self.soft_capacity_max,
                "avg_priority": 0.0,
                "avg_interestingness": 0.0,
                "exploration_rate": self.base_exploration_rate * self.exploration_scale,
                "exploration_scale": self.exploration_scale,
                "total_added": self.total_added,
            }
        priorities = [e.priority for e in self.buffer]
        interestingnesses = [e.interestingness for e in self.buffer]
        return {
            "size": len(self.buffer),
            "soft_capacity": self.soft_capacity,
            "soft_capacity_max": self.soft_capacity_max,
            "avg_priority": sum(priorities) / len(priorities),
            "avg_interestingness": sum(interestingnesses) / len(interestingnesses),
            "exploration_rate": self._current_exploration_rate(),
            "exploration_scale": self.exploration_scale,
            "total_added": self.total_added,
        }

    def __len__(self) -> int:
        return len(self.buffer)
