#!/usr/bin/env python3
"""
attractors.py

Force-based Attractor Dynamics with Homeostasis and Adaptive Basins.

Attractors interact via continuous forces. Merge is emergent.
Homeostasis keeps total energy near a budget.
Basin radius and variance adapt to replay consistency:
  - Coherent replay → basin expands (stable concept)
  - Noisy / inconsistent observations → basin stays tight or contracts

Layer separation:
  Field → Attractor Dynamics → Geometry Training
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import math

import torch


class Attractor:
    """
    A single attractor in latent space with adaptive basin.

    State:
      position, strength, radius, age, replay_count, variance
    """

    def __init__(self, position: torch.Tensor, strength: float = 1.0,
                 radius: float = 0.5):
        self.position = position.detach().cpu().clone().to(torch.float64)
        self.strength = float(strength)
        self.radius = float(radius)
        self.age = 0
        self.replay_count = 1
        self.variance = 0.25
        # Running estimate of how consistent recent observations are
        self.consistency = 1.0

    def reinforce(self, amount: float = 1.0,
                  new_position: Optional[torch.Tensor] = None):
        """Strengthen and adapt basin based on observation consistency."""
        self.strength += amount
        self.replay_count += 1
        self.age = 0

        if new_position is not None:
            new_pos = new_position.detach().cpu().to(torch.float64).flatten()
            delta = new_pos - self.position
            dist = torch.norm(delta).item()

            # Update local variance (exponential moving average)
            self.variance = 0.9 * self.variance + 0.1 * (dist ** 2)

            # Consistency: how close the new observation is relative to current radius
            relative = dist / (self.radius + 1e-6)
            # High consistency when observation lands well inside the basin
            obs_consistency = math.exp(-relative * 1.5)
            self.consistency = 0.85 * self.consistency + 0.15 * obs_consistency

            # Soft move toward the observation
            alpha = min(0.3, amount / (self.strength + 1e-6))
            self.position = (1 - alpha) * self.position + alpha * new_pos

            # Adaptive radius:
            # - Consistent replay → slowly expand (stable concept)
            # - Inconsistent → contract or stay tight
            if self.consistency > 0.7:
                self.radius = min(2.5, self.radius + 0.03 * self.consistency)
            elif self.consistency < 0.35:
                self.radius = max(0.2, self.radius * 0.97)

    def decay(self, factor: float = 0.997):
        self.strength *= factor
        self.age += 1
        # Very old, rarely reinforced basins slowly shrink
        if self.age > 50:
            self.radius = max(0.2, self.radius * 0.999)

    def distance_to(self, other: "Attractor") -> float:
        return torch.norm(self.position - other.position).item()


class AttractorDynamics:
    """
    Force-based attractor landscape with homeostasis and adaptive basins.
    """

    def __init__(self,
                 max_attractors: int = 48,
                 energy_budget: float = 40.0,
                 merge_tolerance: float = 0.15,
                 attraction_scale: float = 0.8,
                 repulsion_scale: float = 0.4,
                 step_size: float = 0.05,
                 min_strength: float = 0.15):
        self.attractors: List[Attractor] = []
        self.max_attractors = max_attractors
        self.energy_budget = energy_budget
        self.merge_tolerance = merge_tolerance
        self.attraction_scale = attraction_scale
        self.repulsion_scale = repulsion_scale
        self.step_size = step_size
        self.min_strength = min_strength

    def __len__(self) -> int:
        return len(self.attractors)

    def total_energy(self) -> float:
        return sum(a.strength for a in self.attractors)

    def reinforce_from_latent(self, latent: torch.Tensor, interestingness: float = 1.0):
        latent = latent.detach().cpu().to(torch.float64).flatten()

        if not self.attractors:
            self.attractors.append(Attractor(latent, strength=1.0 + 0.5 * interestingness))
            return

        distances = [torch.norm(a.position - latent).item() for a in self.attractors]
        nearest_idx = min(range(len(distances)), key=lambda i: distances[i])
        nearest = self.attractors[nearest_idx]
        dist = distances[nearest_idx]

        # Use adaptive radius for membership decision
        if dist < nearest.radius * 1.8:
            nearest.reinforce(amount=0.4 + 0.3 * interestingness, new_position=latent)
        else:
            strength = 1.0 + 0.5 * interestingness
            if len(self.attractors) < self.max_attractors:
                self.attractors.append(Attractor(latent, strength=strength))
            else:
                self.attractors.sort(key=lambda a: a.strength)
                if strength > self.attractors[0].strength:
                    self.attractors[0] = Attractor(latent, strength=strength)

    def _pairwise_force(self, a: Attractor, b: Attractor) -> torch.Tensor:
        delta = b.position - a.position
        dist = torch.norm(delta).item() + 1e-8
        direction = delta / dist

        r_sum = a.radius + b.radius
        strength_factor = math.sqrt(a.strength * b.strength)

        attract = self.attraction_scale * strength_factor * math.exp(
            -((dist - 0.6 * r_sum) ** 2) / (0.4 * r_sum + 1e-6)
        )

        if dist < r_sum:
            repel = self.repulsion_scale * strength_factor * ((r_sum - dist) / (r_sum + 1e-6))
        else:
            repel = 0.0

        return direction * (attract - repel)

    def _apply_homeostasis(self):
        total = self.total_energy()
        if total <= self.energy_budget or not self.attractors:
            return

        excess = total - self.energy_budget
        pressure = min(0.25, excess / (total + 1e-8))

        self.attractors.sort(key=lambda a: a.strength)
        n_weak = max(1, len(self.attractors) // 2)

        for i, a in enumerate(self.attractors):
            if i < n_weak:
                a.strength *= (1.0 - 0.15 * pressure)
            else:
                a.strength *= (1.0 - 0.03 * pressure)

    def step(self):
        n = len(self.attractors)
        if n < 2:
            for a in self.attractors:
                a.decay()
            self.attractors = [a for a in self.attractors if a.strength >= self.min_strength]
            return

        forces = [torch.zeros_like(a.position) for a in self.attractors]

        for i in range(n):
            for j in range(i + 1, n):
                f_ij = self._pairwise_force(self.attractors[i], self.attractors[j])
                forces[i] = forces[i] + f_ij
                forces[j] = forces[j] - f_ij

        for a, f in zip(self.attractors, forces):
            a.position = a.position + self.step_size * f

        for a in self.attractors:
            a.decay()

        self._apply_homeostasis()
        self._emergent_merge()

        self.attractors = [a for a in self.attractors if a.strength >= self.min_strength]

    def _emergent_merge(self):
        merged = True
        while merged and len(self.attractors) >= 2:
            merged = False
            n = len(self.attractors)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.attractors[i].distance_to(self.attractors[j])
                    if dist < self.merge_tolerance:
                        a, b = self.attractors[i], self.attractors[j]
                        total_str = a.strength + b.strength
                        w_a = a.strength / (total_str + 1e-8)
                        w_b = b.strength / (total_str + 1e-8)

                        new_pos = w_a * a.position + w_b * b.position
                        new_strength = total_str * 0.9
                        new_radius = max(a.radius, b.radius) * 1.1
                        new_att = Attractor(new_pos, strength=new_strength, radius=new_radius)
                        new_att.replay_count = a.replay_count + b.replay_count
                        new_att.age = min(a.age, b.age)
                        new_att.variance = (a.variance + b.variance) / 2
                        new_att.consistency = (a.consistency + b.consistency) / 2

                        self.attractors.pop(j)
                        self.attractors.pop(i)
                        self.attractors.append(new_att)
                        merged = True
                        break
                if merged:
                    break

    def get_landscape(self) -> List[Tuple[torch.Tensor, float]]:
        return [(a.position.clone(), a.strength) for a in self.attractors]

    def get_stats(self) -> Dict[str, Any]:
        if not self.attractors:
            return {
                "num_attractors": 0,
                "avg_strength": 0.0,
                "max_strength": 0.0,
                "avg_radius": 0.0,
                "avg_age": 0.0,
                "avg_consistency": 0.0,
                "total_energy": 0.0,
                "energy_budget": self.energy_budget,
            }
        strengths = [a.strength for a in self.attractors]
        radii = [a.radius for a in self.attractors]
        ages = [a.age for a in self.attractors]
        consistencies = [a.consistency for a in self.attractors]
        return {
            "num_attractors": len(self.attractors),
            "avg_strength": sum(strengths) / len(strengths),
            "max_strength": max(strengths),
            "avg_radius": sum(radii) / len(radii),
            "avg_age": sum(ages) / len(ages),
            "avg_consistency": sum(consistencies) / len(consistencies),
            "total_energy": sum(strengths),
            "energy_budget": self.energy_budget,
        }
