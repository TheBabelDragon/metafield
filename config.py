#!/usr/bin/env python3
"""
config.py

Shared configuration and constants for MetaField.
Lightweight for now, will grow as the hybrid system develops.
"""

from dataclasses import dataclass

import torch


@dataclass
class MetaFieldConfig:
    """Base configuration dataclass. Can be expanded later."""
    L: int = 4
    beta: float = 5.5
    hmc_n_leapfrog: int = 20
    hmc_step_size: float = 0.012
    hmc_trajectories: int = 25
    include_fermions: bool = True
    seed: int = 42
    device: str = "cpu"
    dtype: torch.dtype = torch.complex128
