#!/usr/bin/env python3
"""
memory.py

Episodic memory + Persistent Attractor Memory (first version).

High-interestingness experiences that are heavily replayed become
attractor centers. These attractors begin to influence the geometry,
moving us from "memory as stored states" toward "memory as deformation
of the manifold".
"""

from typing import List, Dict, Any, Optional, Tuple

import torch


class EpisodicExperience:
    """
    A single stored experience in episodic memory.
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
        self.replay_count = 0  # how many times this experience has been replayed

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
            0.3 * self.replay_count  # repeated replay strengthens priority
        )


class Attractor:
    """
    A persistent attractor center in latent space.

    Formed from high-interestingness experiences that have been
    heavily replayed. Strength grows with continued replay.
    """

    def __init__(self, latent: torch.Tensor, strength: float = 1.0):
        self.latent = latent.detach().cpu().clone()
        self.strength = float(strength)
        self.visit_count = 1

    def reinforce(self, amount: float = 1.0):
        self.strength += amount
        self.visit_count += 1

    def decay(self, factor: float = 0.995):
        """Slow decay so unused attractors fade."""
        self.strength *= factor


class EpisodicMemory:
    """
    Prioritized episodic memory + first version of Persistent Attractor Memory.

    Features:
    - Multi-signal interestingness
    - Adaptive exploration
    - Associative retrieval
    - Attractor centers formed from heavily replayed high-interest experiences
    """

    def __init__(self, max_size: int = 256,
                 base_exploration_rate: float = 0.15,
                 min_exploration: float = 0.05,
                 max_exploration: float = 0.35,
                 max_attractors: int = 32):
        self.buffer: List[EpisodicExperience] = []
        self.max_size = max_size
        self.base_exploration_rate = base_exploration_rate
        self.min_exploration = min_exploration
        self.max_exploration = max_exploration

        # Persistent attractors
        self.attractors: List[Attractor] = []
        self.max_attractors = max_attractors

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

    def sample(self, n: int = 16) -> List[EpisodicExperience]:
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

        # Reinforce replay counts and form/strengthen attractors
        for exp in sampled:
            exp.replay_count += 1
            self._maybe_form_or_reinforce_attractor(exp)

        return sampled

    def _maybe_form_or_reinforce_attractor(self, exp: EpisodicExperience):
        """
        If an experience is highly interesting and has been replayed enough,
        turn it into (or reinforce) a persistent attractor.
        """
        if exp.interestingness < 1.0 or exp.replay_count < 3:
            return

        # Check if a nearby attractor already exists
        for att in self.attractors:
            dist = torch.norm(att.latent.flatten() - exp.latent.flatten()).item()
            if dist < 0.8:  # close enough in latent space
                att.reinforce(amount=0.5 + 0.2 * exp.interestingness)
                return

        # Create a new attractor if we have room
        if len(self.attractors) < self.max_attractors:
            strength = 1.0 + 0.5 * exp.interestingness
            self.attractors.append(Attractor(exp.latent, strength=strength))
        else:
            # Replace the weakest attractor if this one is stronger
            self.attractors.sort(key=lambda a: a.strength)
            weakest = self.attractors[0]
            new_strength = 1.0 + 0.5 * exp.interestingness
            if new_strength > weakest.strength:
                self.attractors[0] = Attractor(exp.latent, strength=new_strength)

    def decay_attractors(self):
        """Slowly decay unused attractors."""
        for att in self.attractors:
            att.decay()
        # Remove very weak attractors
        self.attractors = [a for a in self.attractors if a.strength > 0.2]

    def get_attractor_latents(self) -> List[Tuple[torch.Tensor, float]]:
        """Return (latent, strength) pairs for use by the geometry model."""
        return [(a.latent, a.strength) for a in self.attractors]

    def find_similar(self, query_latent: torch.Tensor, k: int = 5) -> List[EpisodicExperience]:
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
        if not self.buffer:
            return {
                "size": 0,
                "avg_priority": 0.0,
                "max_priority": 0.0,
                "avg_interestingness": 0.0,
                "max_interestingness": 0.0,
                "exploration_rate": self.base_exploration_rate,
                "num_attractors": 0,
                "avg_attractor_strength": 0.0,
            }

        priorities = [e.priority for e in self.buffer]
        interestingnesses = [e.interestingness for e in self.buffer]
        strengths = [a.strength for a in self.attractors] if self.attractors else [0.0]

        return {
            "size": len(self.buffer),
            "avg_priority": sum(priorities) / len(priorities),
            "max_priority": max(priorities),
            "avg_interestingness": sum(interestingnesses) / len(interestingnesses),
            "max_interestingness": max(interestingnesses),
            "exploration_rate": self._current_exploration_rate(),
            "num_attractors": len(self.attractors),
            "avg_attractor_strength": sum(strengths) / len(strengths),
        }

    def __len__(self) -> int:
        return len(self.buffer)
