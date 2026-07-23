#!/usr/bin/env python3
"""
attractors.py

Force-based Attractor Dynamics subsystem.

Attractors interact continuously via attraction and repulsion forces.
Merge emerges when attractors drift close enough; it is not a hard rule.

Layer separation:
  Field → Attractor Dynamics → Geometry Training
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import math

import torch


class Attractor:
    """
    A single attractor in latent space.

    State:
      position, strength, radius, age, replay_count, covariance (isotropic for now)
    """

    def __init__(self, position: torch.Tensor, strength: float = 1.0,
                 radius: float = 0.5):
        self.position = position.detach().cpu().clone().to(torch.float64)
        self.strength = float(strength)
        self.radius = float(radius)
        self.age = 0
        self.replay_count = 1
        # Isotropic covariance for now (scalar variance)
        self.variance = 0.25

    def reinforce(self, amount: float = 1.0, new_position: Optional[torch.Tensor] = None):
        """Strengthen and optionally nudge position toward a new observation."""
        self.strength += amount
        self.replay_count += 1
        self.age = 0  # reset age on reinforcement

        if new_position is not None:
            # Soft move toward the new observation, weighted by relative strength
            alpha = min(0.3, amount / (self.strength + 1e-6))
            self.position = (1 - alpha) * self.position + alpha * new_position.detach().cpu().to(torch.float64)

    def decay(self, factor: float = 0.997):
        """Slow decay (friction / forgetting)."""
        self.strength *= factor
        self.age += 1

    def distance_to(self, other: "Attractor") -> float:
        return torch.norm(self.position - other.position).item()


class AttractorDynamics:
    """
    Manages the set of attractors and evolves them via continuous forces.

    Forces:
      - Attraction that grows with proximity (up to a scale set by radii)
      - Repulsion when attractors get too close
      - Influence scaled by strength
      - Decay acts as friction

    Merge is emergent: when two attractors drift below a small separation,
    they are replaced by a single combined attractor.
    """

    def __init__(self,
                 max_attractors: int = 48,
                 merge_tolerance: float = 0.15,
                 attraction_scale: float = 0.8,
                 repulsion_scale: float = 0.4,
                 step_size: float = 0.05,
                 min_strength: float = 0.15):
        self.attractors: List[Attractor] = []
        self.max_attractors = max_attractors
        self.merge_tolerance = merge_tolerance
        self.attraction_scale = attraction_scale
        self.repulsion_scale = repulsion_scale
        self.step_size = step_size
        self.min_strength = min_strength

    def __len__(self) -> int:
        return len(self.attractors)

    def reinforce_from_latent(self, latent: torch.Tensor, interestingness: float = 1.0):
        """
        Reinforce an existing nearby attractor or nucleate a new one.
        Called when a high-interestingness experience is replayed.
        """
        latent = latent.detach().cpu().to(torch.float64).flatten()

        if not self.attractors:
            self.attractors.append(Attractor(latent, strength=1.0 + 0.5 * interestingness))
            return

        # Find nearest attractor
        distances = [torch.norm(a.position - latent).item() for a in self.attractors]
        nearest_idx = min(range(len(distances)), key=lambda i: distances[i])
        nearest = self.attractors[nearest_idx]
        dist = distances[nearest_idx]

        # If close enough relative to its radius, reinforce it
        if dist < nearest.radius * 1.8:
            nearest.reinforce(amount=0.4 + 0.3 * interestingness, new_position=latent)
            # Slightly grow radius with consistent replay
            nearest.radius = min(2.0, nearest.radius + 0.02)
        else:
            # Nucleate a new attractor if we have room, or replace weakest
            strength = 1.0 + 0.5 * interestingness
            if len(self.attractors) < self.max_attractors:
                self.attractors.append(Attractor(latent, strength=strength))
            else:
                self.attractors.sort(key=lambda a: a.strength)
                if strength > self.attractors[0].strength:
                    self.attractors[0] = Attractor(latent, strength=strength)

    def _pairwise_force(self, a: Attractor, b: Attractor) -> torch.Tensor:
        """
        Continuous force exerted on a by b.

        Attraction at moderate distance, repulsion when too close.
        Magnitude scaled by strengths.
        """
        delta = b.position - a.position
        dist = torch.norm(delta).item() + 1e-8
        direction = delta / dist

        # Characteristic scales
        r_sum = a.radius + b.radius
        strength_factor = math.sqrt(a.strength * b.strength)

        # Soft attraction (peaks around moderate separation)
        attract = self.attraction_scale * strength_factor * math.exp(-((dist - 0.6 * r_sum) ** 2) / (0.4 * r_sum + 1e-6))

        # Repulsion when closer than the combined radii
        if dist < r_sum:
            repel = self.repulsion_scale * strength_factor * ((r_sum - dist) / (r_sum + 1e-6))
        else:
            repel = 0.0

        force_mag = attract - repel
        return direction * force_mag

    def step(self):
        """
        One dynamics step: apply pairwise forces, then check for emergent merges.
        """
        n = len(self.attractors)
        if n < 2:
            # Still apply decay
            for a in self.attractors:
                a.decay()
            self.attractors = [a for a in self.attractors if a.strength >= self.min_strength]
            return

        # Compute forces
        forces = [torch.zeros_like(a.position) for a in self.attractors]

        for i in range(n):
            for j in range(i + 1, n):
                f_ij = self._pairwise_force(self.attractors[i], self.attractors[j])
                forces[i] = forces[i] + f_ij
                forces[j] = forces[j] - f_ij  # Newton's third law

        # Update positions
        for a, f in zip(self.attractors, forces):
            a.position = a.position + self.step_size * f

        # Decay
        for a in self.attractors:
            a.decay()

        # Emergent merge: if two attractors are closer than merge_tolerance,
        # replace them with a single combined attractor.
        self._emergent_merge()

        # Remove very weak attractors
        self.attractors = [a for a in self.attractors if a.strength >= self.min_strength]

    def _emergent_merge(self):
        """Merge pairs that have drifted below the separation tolerance."""
        merged = True
        while merged and len(self.attractors) >= 2:
            merged = False
            n = len(self.attractors)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = self.attractors[i].distance_to(self.attractors[j])
                    if dist < self.merge_tolerance:
                        # Combine into a single attractor
                        a, b = self.attractors[i], self.attractors[j]
                        total_str = a.strength + b.strength
                        w_a = a.strength / (total_str + 1e-8)
                        w_b = b.strength / (total_str + 1e-8)

                        new_pos = w_a * a.position + w_b * b.position
                        new_strength = total_str * 0.9  # slight loss on merge
                        new_radius = max(a.radius, b.radius) * 1.1
                        new_att = Attractor(new_pos, strength=new_strength, radius=new_radius)
                        new_att.replay_count = a.replay_count + b.replay_count
                        new_att.age = min(a.age, b.age)

                        # Remove the two and add the combined one
                        self.attractors.pop(j)
                        self.attractors.pop(i)
                        self.attractors.append(new_att)
                        merged = True
                        break
                if merged:
                    break

    def get_landscape(self) -> List[Tuple[torch.Tensor, float]]:
        """Return (position, strength) pairs for geometry training."""
        return [(a.position.clone(), a.strength) for a in self.attractors]

    def get_stats(self) -> Dict[str, Any]:
        if not self.attractors:
            return {
                "num_attractors": 0,
                "avg_strength": 0.0,
                "max_strength": 0.0,
                "avg_radius": 0.0,
                "avg_age": 0.0,
            }
        strengths = [a.strength for a in self.attractors]
        radii = [a.radius for a in self.attractors]
        ages = [a.age for a in self.attractors]
        return {
            "num_attractors": len(self.attractors),
            "avg_strength": sum(strengths) / len(strengths),
            "max_strength": max(strengths),
            "avg_radius": sum(radii) / len(radii),
            "avg_age": sum(ages) / len(ages),
        }
