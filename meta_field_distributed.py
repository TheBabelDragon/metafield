#!/usr/bin/env python3
"""
meta_field_distributed.py v1.8
Critical MD fix + enhanced diagnostics
"""

from __future__ import annotations

import os
import sys
import math
import socket
import argparse
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.distributed as dist

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

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
)


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def print_banner(rank: int, world_size: int, role: str, master_addr: str, master_port: int, diagnostic: bool = False):
    print("\n" + "=" * 72)
    print("  MetaField Distributed v1.8 - MD Fix + Diagnostics")
    print("=" * 72)
    print(f"   Role: {role.upper()} | Rank {rank}/{world_size}")
    if diagnostic:
        print("   [DIAGNOSTIC MODE ENABLED]")
    if world_size > 1 and role in ("control", "auto"):
        print(f"\n[CONTROL] Worker command printed above")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["auto", "control", "worker"], default="auto")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--master-addr", default="auto")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--backend", default="gloo")
    p.add_argument("--include-fermions", type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=True)
    p.add_argument("--diagnostic", action="store_true", default=False)
    return p.parse_args()


def init_distributed(args):
    role = args.role
    world_size = args.world_size
    master_addr = args.master_addr if args.master_addr != "auto" else get_local_ip()
    master_port = args.master_port
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", 0))
    if role == "control": rank = 0
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ["RANK"] = str(rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size, init_method="env://")
    print_banner(rank, world_size, role, master_addr, master_port, args.diagnostic)
    return rank, world_size, master_addr, master_port


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def simple_sparkline(data: List[float], width: int = 50) -> str:
    if not data: return ""
    min_v, max_v = min(data), max(data)
    if max_v == min_v: return "=" * width
    scale = (max_v - min_v) or 1.0
    return ''.join(chr(0x2581 + min(7, int((v - min_v) / scale * 7))) for v in data[:width])


class DistributedLattice:
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.L = config.L
        assert self.L % world_size == 0
        self.local_L = self.L // world_size
        self.halo = 1
        self.local_padded_shape = (self.local_L + 2 * self.halo, self.L, self.L, self.L)
        self.local_volume = self.local_L * (self.L ** 3)
        self.global_volume = self.L ** 4
        self.left_neighbor = (rank - 1) % world_size
        self.right_neighbor = (rank + 1) % world_size
        self.device = torch.device(config.device)
        self.dtype = config.dtype

    def _halo_exchange(self, tensor, tag_base=42):
        if self.world_size == 1: return
        left, right = self.left_neighbor, self.right_neighbor
        recv_l = dist.irecv(tensor.narrow(0, 0, 1).contiguous(), src=left, tag=tag_base)
        recv_r = dist.irecv(tensor.narrow(0, self.local_L + 1, 1).contiguous(), src=right, tag=tag_base+1)
        send_l = dist.isend(tensor.narrow(0, 1, 1).contiguous(), dst=left, tag=tag_base+1)
        send_r = dist.isend(tensor.narrow(0, self.local_L, 1).contiguous(), dst=right, tag=tag_base)
        recv_l.wait(); recv_r.wait(); send_l.wait(); send_r.wait()

    def update_halo_gauge(self, U): self._halo_exchange(U, 100)
    def update_halo_fermion(self, psi): self._halo_exchange(psi, 200)

    def shift(self, field, axis, direction):
        if axis != 0 or self.world_size == 1:
            return torch.roll(field, shifts=-direction, dims=axis)
        self._halo_exchange(field, 300 + axis)
        return torch.roll(field, shifts=-direction, dims=0)

    def global_sum(self, x):
        if self.world_size == 1: return x
        s = x.clone()
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        return s


class DistributedGaugeField:
    def __init__(self, lattice, config, generator):
        self.lattice = lattice
        self.config = config
        shape = lattice.local_padded_shape + (4, config.color_dim, config.color_dim)
        eye = torch.eye(config.color_dim, dtype=config.dtype, device=lattice.device).expand(shape).clone()
        noise = random_su_n_hermitian(shape, config.color_dim, config.dtype, lattice.device, generator)
        X = 1j * 0.1 * project_traceless_hermitian(noise)
        self.U = expm_anti_hermitian(project_traceless_antihermitian(X)) @ eye

    def plaquette_traces(self, U=None):
        if U is None: U = self.U
        lat = self.lattice
        traces = []
        for mu in range(4):
            for nu in range(mu+1, 4):
                U_mu = U[..., mu, :, :]
                U_nu_xpm = lat.shift(U[..., nu, :, :], mu, +1)
                U_mu_xpn = lat.shift(U_mu, nu, +1)
                plaq = U_mu @ U_nu_xpm @ dagger(U_mu_xpn) @ dagger(U[..., nu, :, :])
                tr = torch.diagonal(plaq, -2, -1).sum(-1).real / self.config.color_dim
                traces.append(tr)
        return lat.global_sum(torch.stack(traces).sum()) / lat.global_volume

    def wilson_action(self, U=None):
        if U is None: U = self.U
        return self.config.beta * self.lattice.global_volume * (1.0 - self.plaquette_traces(U))

    def force(self, U=None):
        if U is None: U = self.U
        U_req = U.detach().clone().requires_grad_(True)
        S = self.wilson_action(U_req)
        grad = torch.autograd.grad(S, U_req)[0]
        return project_traceless_antihermitian(U_req.detach() @ dagger(grad.detach()))


class DistributedWilsonDiracOperator:
    def __init__(self, lattice, config):
        self.lattice = lattice
        self.config = config
        self.gammas = euclidean_gamma_matrices(config.dtype, lattice.device)
        self.g5 = gamma5(self.gammas)
        eye4 = torch.eye(4, dtype=config.dtype, device=lattice.device)
        self.r_plus = [config.wilson_r * eye4 + g for g in self.gammas]
        self.r_minus = [config.wilson_r * eye4 - g for g in self.gammas]

    def apply(self, psi, U):
        cfg, lat = self.config, self.lattice
        out = (cfg.mass + cfg.n_dims * cfg.wilson_r) * psi
        for mu in range(4):
            U_mu = U[..., mu, :, :]
            U_mu_back = lat.shift(U_mu, mu, -1)
            psi_fwd = lat.shift(psi, mu, +1)
            psi_back = lat.shift(psi, mu, -1)
            out = out - 0.5 * (
                torch.einsum('st,...ti->...si', self.r_minus[mu], torch.einsum('...ij,...sj->...si', U_mu, psi_fwd)) +
                torch.einsum('st,...ti->...si', self.r_plus[mu], torch.einsum('...ij,...sj->...si', dagger(U_mu_back), psi_back))
            )
        return out

    def apply_dagger(self, psi, U):
        return torch.einsum('st,...ti->...si', self.g5, self.apply(torch.einsum('st,...ti->...si', self.g5, psi), U))

    def normal_op(self, psi, U):
        return self.apply_dagger(self.apply(psi, U), U)


class DistributedPseudofermionField:
    def __init__(self, lattice, dirac, config, generator):
        self.lattice = lattice
        self.dirac = dirac
        self.config = config
        self.generator = generator
        self.phi = None

    def refresh(self, U):
        shape = self.lattice.local_padded_shape + (self.config.spinor_dim, self.config.color_dim)
        real = torch.randn(shape, generator=self.generator, dtype=torch.float64, device=self.lattice.device)
        imag = torch.randn(shape, generator=self.generator, dtype=torch.float64, device=self.lattice.device)
        eta = (real + 1j * imag).to(self.config.dtype)
        self.phi = self.dirac.apply_dagger(eta, U)


class DistributedHMC:
    def __init__(self, gauge, dirac, config, generator, pseudo=None, diagnostic=False):
        self.gauge = gauge
        self.dirac = dirac
        self.config = config
        self.generator = generator
        self.pseudo = pseudo
        self.lattice = gauge.lattice
        self.diagnostic = diagnostic
        self.n_accepted = self.n_total = 0
        self.action_history: List[float] = []
        self.delta_h_history: List[float] = []
        self.accepted_history: List[bool] = []

    def trajectory(self):
        cfg, lat = self.config, self.lattice
        U0 = self.gauge.U.clone()
        P0 = random_su_n_hermitian(U0.shape, cfg.color_dim, cfg.dtype, U0.device, self.generator)
        if self.pseudo:
            self.pseudo.refresh(U0)

        action0 = float(self.gauge.wilson_action(U0).real)   # use local U0
        kinetic0 = 0.5 * torch.sum((P0 @ P0).diagonal(dim1=-2, dim2=-1).sum(-1).real)
        H0 = lat.global_sum(kinetic0) + action0

        if self.diagnostic and self.lattice.rank == 0:
            print(f"  [Diag] Start Action = {action0:.4f}")

        U, P = U0.clone(), P0.clone()
        eps = cfg.hmc_step_size

        F = self.gauge.force(U)          # pass local U
        P += 0.5 * eps * (1j * F)

        for step in range(cfg.hmc_n_leapfrog):
            U = expm_anti_hermitian(eps * (-1j * P)) @ U
            lat.update_halo_gauge(U)
            F = self.gauge.force(U)      # pass local U
            coeff = eps if step < cfg.hmc_n_leapfrog - 1 else 0.5 * eps
            P += coeff * (1j * F)

        action1 = float(self.gauge.wilson_action(U).real)   # use final local U
        kinetic1 = 0.5 * torch.sum((P @ P).diagonal(dim1=-2, dim2=-1).sum(-1).real)
        H1 = lat.global_sum(kinetic1) + action1
        delta_h = float((H1 - H0).real)

        if self.diagnostic and self.lattice.rank == 0:
            print(f"  [Diag] End Action   = {action1:.4f} | raw ΔH = {delta_h:+.4f}")

        accept_prob = min(1.0, math.exp(-delta_h)) if delta_h < 700 else 0.0
        accepted = torch.rand((), generator=self.generator).item() < accept_prob

        self.n_total += 1
        if accepted:
            self.n_accepted += 1
            self.gauge.U = U.clone()     # only update self.U on accept
            lat.update_halo_gauge(self.gauge.U)

        self.action_history.append(action1)
        self.delta_h_history.append(delta_h)
        self.accepted_history.append(accepted)

        return {"delta_h": delta_h, "accepted": accepted, "acceptance_rate": self.n_accepted / max(1, self.n_total)}


def main():
    args = parse_args()
    rank, world_size, master_addr, master_port = init_distributed(args)

    config = ConfigV2(
        L=4,
        beta=5.5,
        hmc_n_leapfrog=20,
        hmc_step_size=0.012,
        hmc_trajectories=12,
        include_fermions=args.include_fermions,
        seed=42 + rank,
        device="cpu",
        dtype=torch.complex128,
    )

    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed + rank * 777)

    lat = DistributedLattice(config, rank, world_size)
    gauge = DistributedGaugeField(lat, config, gen)
    dirac = DistributedWilsonDiracOperator(lat, config)
    pseudo = DistributedPseudofermionField(lat, dirac, config, gen) if config.include_fermions else None
    hmc = DistributedHMC(gauge, dirac, config, gen, pseudo, diagnostic=args.diagnostic)

    if rank == 0:
        mode = "DYNAMICAL + fermions" if config.include_fermions else "QUENCHED (gauge only)"
        print(f"Starting {mode} HMC on {world_size} rank(s)...\n")

    for t in range(config.hmc_trajectories):
        res = hmc.trajectory()
        if rank == 0:
            status = "ACCEPTED" if res["accepted"] else "REJECTED"
            print(f"traj {t:02d} | dH={res['delta_h']:+.4f} | {status} (rate={res['acceptance_rate']:.2f})")

    if rank == 0:
        print("\nRun finished.")

        if args.diagnostic and hmc.delta_h_history:
            print("\n=== DIAGNOSTIC SUMMARY ===")
            print(f"Final acceptance rate : {hmc.n_accepted / max(1, hmc.n_total):.2%}")
            print(f"Mean |\u0394H|            : {sum(abs(d) for d in hmc.delta_h_history) / len(hmc.delta_h_history):.2f}")
            print(f"Action changed?       : {min(hmc.action_history) != max(hmc.action_history)}")

            print("\n\u0394H sparkline:")
            print(simple_sparkline([abs(d) for d in hmc.delta_h_history]))

            if HAS_MATPLOTLIB:
                try:
                    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                    axs[0].plot(hmc.action_history, marker='.', label='Action')
                    axs[0].set_ylabel('Wilson Action')
                    axs[1].plot(hmc.delta_h_history, marker='o', label='\u0394H')
                    axs[1].axhline(0, color='gray', ls='--')
                    axs[1].set_ylabel('\u0394H')
                    axs[2].plot([int(a) for a in hmc.accepted_history], marker='o', color='green')
                    axs[2].set_ylabel('Accepted')
                    axs[2].set_xlabel('Trajectory')
                    plt.tight_layout()
                    plt.savefig('hmc_diagnostics.png', dpi=150)
                    print("\nSaved hmc_diagnostics.png")
                except Exception as e:
                    print(f"Plot error: {e}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
