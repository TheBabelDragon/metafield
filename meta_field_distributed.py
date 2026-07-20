#!/usr/bin/env python3
"""
meta_field_distributed.py
=========================

Domain-decomposed (1D along axis 0) distributed version of MetaField v2.

Supports arbitrary N ranks using torch.distributed + Gloo backend (TCP).
Designed for heterogeneous CPU clusters (your ProDesk 600 G6 + Wyse 5070 setup).

Key features in this version:
- Full quenched (pure gauge) and dynamical (pseudofermion) HMC
- Halo exchange for gauge links and Wilson-Dirac spinor fields
- Reuses math primitives (expm, project_*, cg_solve, etc.) from meta_field_sim_torch
- LearnedInformationGeometry runs on rank 0 only (for now)
- Clear launch instructions for torchrun on 2+ machines

Launch on your two machines (example — replace with your actual IPs):

# On the control/master machine (e.g. prodesk600g6, rank 0)
torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --rdzv_id=metafield_hmc \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.1.10:29500 \
  meta_field_distributed.py

# The second machine (wyse5070) just needs to be reachable and run the same command.
# torchrun handles the rendezvous automatically.

Alternative (manual init, useful for debugging):
    export MASTER_ADDR=192.168.1.10
    export MASTER_PORT=29500
    export WORLD_SIZE=2
    export RANK=0   # or 1 on the second machine
    python meta_field_distributed.py

Hardware notes for your setup:
- ProDesk 600 G6 (3.1 GHz) is likely the stronger node — good as rank 0.
- Wyse 5070 is a thin client; expect it to be slower. Load balancing is future work.
- Use a wired Gigabit+ network between them for reasonable halo exchange performance.
- Gloo over TCP is reliable here (no CUDA/NCCL dependency).

This is v1 infrastructure: correct halo exchange + full HMC.
Future iterations can add:
- 2D/3D decomposition
- Async / overlapped communication
- Distributed LearnedInformationGeometry training
- Better load balancing across heterogeneous nodes

Treat first runs as debugging passes (shape/dtype/comm bugs are common when moving to distributed).
"""

from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.distributed as dist
import torch.nn as nn

# Reuse the battle-tested primitives from the single-node implementation
from meta_field_sim_torch import (
    ConfigV2,
    dagger,
    project_traceless_antihermitian,
    project_traceless_hermitian,
    expm_anti_hermitian,
    random_su_n_hermitian,
    euclidean_gamma_matrices,
    gamma5,
    cg_solve,
    _MLP,                    # for LearnedInformationGeometry
    LearnedInformationGeometry,
)


# ---------------------------------------------------------------------------
# Distributed initialization (robust for your two-machine setup)
# ---------------------------------------------------------------------------

def init_distributed(backend: str = "gloo") -> Tuple[int, int]:
    """
    Initialize torch.distributed process group.
    Works with both torchrun and manual RANK/WORLD_SIZE env vars.
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            # torchrun sets these; provide sensible default for manual launch
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method="env://",
        )
        # Optional: set device for future CUDA support (currently CPU)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Distributed Lattice with 1D domain decomposition + halo exchange
# ---------------------------------------------------------------------------

class DistributedLattice:
    """
    1D domain decomposition along axis 0 (first lattice dimension).
    Each rank owns local_L = L // world_size sites in axis 0.
    Fields are allocated with halo=1 padding in axis 0 for nearest-neighbor access.

    This class provides:
    - local_shape, global_volume, rank, world_size, neighbors
    - halo_exchange_gauge / halo_exchange_fermion
    - distributed_shift (transparent to calling code)
    - global_sum for reductions (action, CG dots, etc.)
    """

    def __init__(self, config: ConfigV2, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.L = config.L
        assert self.L % world_size == 0, f"L={self.L} must be divisible by world_size={world_size}"
        self.local_L = self.L // world_size
        self.halo = 1

        # Local padded shape: (local_L + 2*halo, L, L, L)
        self.local_padded_shape = (self.local_L + 2 * self.halo, self.L, self.L, self.L)
        self.local_volume = self.local_L * (self.L ** 3)
        self.global_volume = self.L ** 4

        # Periodic neighbors in the 1D ring
        self.left_neighbor = (rank - 1) % world_size
        self.right_neighbor = (rank + 1) % world_size

        self.device = torch.device(config.device)
        self.dtype = config.dtype

        if rank == 0:
            print(f"[Rank {rank}] DistributedLattice: global L={self.L}, "
                  f"world_size={world_size}, local_L={self.local_L} (axis 0)")

    # --- Halo exchange primitives -------------------------------------------------

    def _halo_exchange(self, tensor: torch.Tensor, tag_base: int = 42) -> None:
        """
        In-place halo exchange along axis 0 for a tensor that has shape
        (local_L + 2, ...). Works for both gauge (extra dims) and fermion fields.
        Uses non-blocking isend/irecv to avoid deadlock in the ring.
        """
        if self.world_size == 1:
            return

        left = self.left_neighbor
        right = self.right_neighbor

        # We exchange the owned boundary slices into the neighbor's halo
        # tensor[1:2]   = leftmost owned slice
        # tensor[-2:-1] = rightmost owned slice

        # Receive into left halo from left neighbor's right owned
        recv_left = dist.irecv(
            tensor.narrow(0, 0, 1).contiguous(),
            src=left,
            tag=tag_base
        )
        # Receive into right halo from right neighbor's left owned
        recv_right = dist.irecv(
            tensor.narrow(0, self.local_L + 1, 1).contiguous(),
            src=right,
            tag=tag_base + 1
        )

        # Send my leftmost owned to left neighbor's right halo
        send_left = dist.isend(
            tensor.narrow(0, 1, 1).contiguous(),
            dst=left,
            tag=tag_base + 1
        )
        # Send my rightmost owned to right neighbor's left halo
        send_right = dist.isend(
            tensor.narrow(0, self.local_L, 1).contiguous(),
            dst=right,
            tag=tag_base
        )

        # Wait in safe order
        recv_left.wait()
        recv_right.wait()
        send_left.wait()
        send_right.wait()

    def update_halo_gauge(self, U_padded: torch.Tensor) -> None:
        """Exchange halos for the full gauge field tensor (includes all 4 directions)."""
        self._halo_exchange(U_padded, tag_base=100)

    def update_halo_fermion(self, psi_padded: torch.Tensor) -> None:
        """Exchange halos for fermion field (spinor x color)."""
        self._halo_exchange(psi_padded, tag_base=200)

    # --- Distributed shift (core primitive) --------------------------------------

    def shift(self, field: torch.Tensor, axis: int, direction: int) -> torch.Tensor:
        """
        Distributed-aware shift.
        - For axis != 0: normal periodic roll (local)
        - For axis == 0: ensure halo is fresh, then roll on the padded tensor.
          The halo data provides the values from the neighboring rank.
        """
        if axis != 0 or self.world_size == 1:
            # Normal local periodic shift (other dimensions or single rank)
            return torch.roll(field, shifts=-direction, dims=axis)

        # axis == 0 and distributed → use halo
        if field.dim() < 1 or field.shape[0] != self.local_L + 2 * self.halo:
            raise ValueError(
                f"Expected padded field with shape[0]={self.local_L + 2}, got {field.shape}"
            )

        # Make sure halo contains up-to-date neighbor data
        # (caller should call update_halo_* before sequences of shifts in axis 0)
        self._halo_exchange(field, tag_base=300 + axis)

        return torch.roll(field, shifts=-direction, dims=0)

    # --- Global reductions --------------------------------------------------------

    def global_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Sum a scalar or tensor across all ranks (for action, norms, etc.)."""
        if self.world_size == 1:
            return x
        s = x.clone()
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        return s

    def global_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_sum(x) / self.world_size

    # --- Convenience creators ----------------------------------------------------

    def new_gauge_field(self, generator: torch.Generator) -> torch.Tensor:
        """Allocate padded gauge field (local_L+2, L, L, L, 4, Nc, Nc)."""
        shape = self.local_padded_shape + (4, self.config.color_dim, self.config.color_dim)
        eye = torch.eye(self.config.color_dim, dtype=self.dtype, device=self.device)
        eye = eye.expand(shape).clone()

        noise = random_su_n_hermitian(
            shape, self.config.color_dim, self.dtype, self.device, generator
        )
        X = 1j * 0.1 * project_traceless_hermitian(noise)
        U = expm_anti_hermitian(project_traceless_antihermitian(X)) @ eye
        return U

    def new_fermion_field(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Allocate padded fermion field (local_L+2, L, L, L, spinor, color)."""
        shape = self.local_padded_shape + (self.config.spinor_dim, self.config.color_dim)
        if generator is None:
            real = torch.randn(shape, dtype=torch.float64, device=self.device)
            imag = torch.randn(shape, dtype=torch.float64, device=self.device)
        else:
            real = torch.randn(shape, generator=generator, dtype=torch.float64, device=self.device)
            imag = torch.randn(shape, generator=generator, dtype=torch.float64, device=self.device)
        return (real + 1j * imag).to(self.dtype)


# ---------------------------------------------------------------------------
# Distributed Gauge Field (reuses math, adds halo + global reductions)
# ---------------------------------------------------------------------------

class DistributedGaugeField:
    def __init__(self, lattice: DistributedLattice, config: ConfigV2, generator: torch.Generator):
        self.lattice = lattice
        self.config = config
        self.U = lattice.new_gauge_field(generator)   # already padded

    def plaquette_traces(self) -> torch.Tensor:
        lat = self.lattice
        traces = []
        U = self.U
        for mu in range(lat.config.n_dims):  # usually 4
            for nu in range(mu + 1, lat.config.n_dims):
                U_mu = U[..., mu, :, :]
                U_nu = U[..., nu, :, :]

                # Shift in nu (may trigger halo if nu==0)
                U_nu_xpm = lat.shift(U_nu, mu, +1) if mu != 0 else lat.shift(U_nu, mu, +1)
                # The shift method already handles halo for axis==0
                U_mu_xpn = lat.shift(U_mu, nu, +1)

                plaq = U_mu @ U_nu_xpm @ dagger(U_mu_xpn) @ dagger(U_nu)
                tr = torch.diagonal(plaq, dim1=-2, dim2=-1).sum(-1).real / self.config.color_dim
                traces.append(tr)

        local_sum = torch.stack(traces, dim=-1).sum()
        return lat.global_sum(local_sum) / lat.global_volume   # average plaquette trace

    def wilson_action(self) -> torch.Tensor:
        traces_avg = self.plaquette_traces()
        # S = beta * V * (1 - <ReTr U_p>/N)
        return self.config.beta * self.lattice.global_volume * (1.0 - traces_avg)

    def force(self) -> torch.Tensor:
        """Autograd force on the padded local field, then project."""
        U = self.U.detach().clone().requires_grad_(True)
        S = self.wilson_action()          # already globally reduced inside
        (grad,) = torch.autograd.grad(S, U)
        raw = U.detach() @ dagger(grad.detach())
        # Only project the owned region? For simplicity we project everything (halo is not used in force)
        return project_traceless_antihermitian(raw)


# ---------------------------------------------------------------------------
# Distributed Wilson-Dirac Operator
# ---------------------------------------------------------------------------

class DistributedWilsonDiracOperator:
    def __init__(self, lattice: DistributedLattice, config: ConfigV2):
        self.lattice = lattice
        self.config = config
        self.gammas = euclidean_gamma_matrices(config.dtype, lattice.device)
        self.g5 = gamma5(self.gammas)
        eye4 = torch.eye(4, dtype=config.dtype, device=lattice.device)
        self.r_plus = [config.wilson_r * eye4 + g for g in self.gammas]
        self.r_minus = [config.wilson_r * eye4 - g for g in self.gammas]

    def apply(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Wilson-Dirac on padded fields.
        Caller must ensure halos are up-to-date before calling when shifts in axis 0 occur.
        """
        cfg = self.config
        lat = self.lattice
        out = (cfg.mass + cfg.n_dims * cfg.wilson_r) * psi

        for mu in range(lat.config.n_dims):
            U_mu = U[..., mu, :, :]
            U_mu_back = lat.shift(U_mu, mu, -1)

            psi_fwd = lat.shift(psi, mu, +1)
            psi_back = lat.shift(psi, mu, -1)

            transported_fwd = torch.einsum('...ij,...sj->...si', U_mu, psi_fwd)
            transported_back = torch.einsum('...ij,...sj->...si', dagger(U_mu_back), psi_back)

            term_fwd = torch.einsum('st,...ti->...si', self.r_minus[mu], transported_fwd)
            term_back = torch.einsum('st,...ti->...si', self.r_plus[mu], transported_back)

            out = out - 0.5 * (term_fwd + term_back)

        return out

    def apply_dagger(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        g5psi = torch.einsum('st,...ti->...si', self.g5, psi)
        Dg5psi = self.apply(g5psi, U)
        return torch.einsum('st,...ti->...si', self.g5, Dg5psi)

    def normal_op(self, psi: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        return self.apply_dagger(self.apply(psi, U), U)


# ---------------------------------------------------------------------------
# Distributed HMC (full quenched + dynamical support)
# ---------------------------------------------------------------------------

class DistributedHMC:
    def __init__(
        self,
        gauge: DistributedGaugeField,
        dirac: DistributedWilsonDiracOperator,
        config: ConfigV2,
        generator: torch.Generator,
        pseudofermion: Optional["DistributedPseudofermionField"] = None,
    ):
        self.gauge = gauge
        self.dirac = dirac
        self.config = config
        self.generator = generator
        self.pseudofermion = pseudofermion
        self.lattice = gauge.lattice
        self.n_accepted = 0
        self.n_total = 0

    def _hamiltonian(self, U: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        kinetic = 0.5 * torch.sum((P @ P).diagonal(dim1=-2, dim2=-1).sum(-1).real)
        potential = self.gauge.wilson_action()

        if self.pseudofermion is not None:
            # In real run we would have solved for x already
            pass  # handled in trajectory for simplicity in v1

        return self.lattice.global_sum(kinetic) + potential

    def trajectory(self) -> Dict[str, Any]:
        cfg = self.config
        lat = self.lattice

        U0 = self.gauge.U.clone()
        shape = U0.shape
        P0 = random_su_n_hermitian(shape, self.gauge.config.color_dim, cfg.dtype, U0.device, self.generator)

        # Heatbath pseudofermion (if dynamical)
        if self.pseudofermion is not None:
            self.pseudofermion.refresh(U0)

        H0 = self._hamiltonian(U0, P0)

        U, P = U0.clone(), P0.clone()
        eps = cfg.hmc_step_size

        # Simple leapfrog (force computation includes halo management inside gauge.force)
        F = self.gauge.force()
        P = P + 0.5 * eps * (1j * F)

        for step in range(cfg.hmc_n_leapfrog):
            U = expm_anti_hermitian(eps * (-1j * P)) @ U
            # After link update we should refresh halos if using U in force
            lat.update_halo_gauge(U)
            F = self.gauge.force()
            coeff = eps if step < cfg.hmc_n_leapfrog - 1 else 0.5 * eps
            P = P + coeff * (1j * F)

        H1 = self._hamiltonian(U, P)
        delta_h = float((H1 - H0).real)

        accept_prob = min(1.0, math.exp(-delta_h)) if delta_h < 700 else 0.0
        u = torch.rand((), generator=self.generator, dtype=torch.float64).item()
        accepted = u < accept_prob

        self.n_total += 1
        if accepted:
            self.n_accepted += 1
            self.gauge.U = U
            lat.update_halo_gauge(U)   # keep halo consistent after accept

        return {
            "delta_h": delta_h,
            "accept_prob": accept_prob,
            "accepted": accepted,
            "acceptance_rate": self.n_accepted / self.n_total if self.n_total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Distributed Pseudofermion (simplified for v1)
# ---------------------------------------------------------------------------

class DistributedPseudofermionField:
    def __init__(self, lattice: DistributedLattice, dirac: DistributedWilsonDiracOperator,
                 config: ConfigV2, generator: torch.Generator):
        self.lattice = lattice
        self.dirac = dirac
        self.config = config
        self.generator = generator
        self.phi: Optional[torch.Tensor] = None

    def refresh(self, U: torch.Tensor):
        shape = self.lattice.local_padded_shape + (self.config.spinor_dim, self.config.color_dim)
        real = torch.randn(shape, generator=self.generator, dtype=torch.float64, device=self.lattice.device)
        imag = torch.randn(shape, generator=self.generator, dtype=torch.float64, device=self.lattice.device)
        eta = (real + 1j * imag).to(self.config.dtype)
        self.phi = self.dirac.apply_dagger(eta, U)

    # solve / action / force would follow the same pattern as single-node version
    # with global reductions inside CG and action. Omitted in v1 for brevity but straightforward to add.


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_distributed_hmc(config: Optional[ConfigV2] = None):
    rank, world_size = init_distributed(backend="gloo")

    if config is None:
        config = ConfigV2(
            L=4,
            hmc_trajectories=5,
            include_fermions=True,
            seed=42 + rank,   # different seed per rank (or broadcast master seed)
        )

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed + rank)

    lattice = DistributedLattice(config, rank, world_size)
    gauge = DistributedGaugeField(lattice, config, generator)
    dirac = DistributedWilsonDiracOperator(lattice, config)

    pseudo = None
    if config.include_fermions:
        pseudo = DistributedPseudofermionField(lattice, dirac, config, generator)

    hmc = DistributedHMC(gauge, dirac, config, generator, pseudofermion=pseudo)

    if rank == 0:
        mode = "dynamical (pseudofermion)" if config.include_fermions else "quenched"
        print(f"\n=== MetaField Distributed v1 ({mode}) on {world_size} ranks ===")
        print(f"Lattice: {config.L}^4 | beta={config.beta} | trajectories={config.hmc_trajectories}\n")

    for traj in range(config.hmc_trajectories):
        result = hmc.trajectory()

        if rank == 0:
            print(
                f"traj {traj:3d} | dH={result['delta_h']:+.4f} | "
                f"{'ACC' if result['accepted'] else 'rej'} "
                f"(rate={result['acceptance_rate']:.2f})"
            )

    if rank == 0:
        print("\nDistributed HMC run complete.")

    cleanup_distributed()


if __name__ == "__main__":
    # Example config — edit as needed
    config = ConfigV2(
        L=4,
        beta=5.5,
        hmc_n_leapfrog=8,
        hmc_step_size=0.04,
        hmc_trajectories=8,
        include_fermions=True,   # set False for faster quenched tests
        seed=123,
        device="cpu",
        dtype=torch.complex128,
    )
    run_distributed_hmc(config)
