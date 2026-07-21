#!/usr/bin/env python3
"""
meta_field_distributed.py v1.34

Final night improvements:
- Made system summary interval configurable (--summary-interval)
- Continued hybrid readiness
"""

from __future__ import annotations

import os
import sys
import math
import socket
import argparse
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# === Hybrid modular imports ===
from memory import EpisodicMemory, EpisodicExperience
from prediction import LatentPredictor
from geometry import LearnedInformationGeometry


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


def get_real_lan_ip() -> str:
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if not ip.startswith("127."):
                return ip
    except:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127."):
            return ip
    except:
        pass

    return "127.0.0.1"


def print_banner(rank: int, world_size: int, role: str, master_addr: str, master_port: int, diagnostic: bool = False):
    print("\n" + "=" * 72)
    print("  MetaField Distributed v1.34")
    print("=" * 72)
    print(f"   Role: {role.upper()} | Rank {rank}/{world_size}")
    if diagnostic:
        print("   [DIAGNOSTIC + MEMORY + PREDICTION]")
    print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["auto", "control", "worker"], default="auto")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--master-addr", default="auto")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--backend", default="gloo")

    def parse_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1', 'yes', 'y'):
            return True
        if v.lower() in ('false', '0', 'no', 'n'):
            return False
        return True

    p.add_argument("--include-fermions", type=parse_bool, nargs='?', const=True, default=True)
    p.add_argument("--diagnostic", action="store_true", default=False)
    p.add_argument("--save-plots", action="store_true", default=False)
    p.add_argument("--continuous", action="store_true", default=False)
    p.add_argument("--summary-interval", type=int, default=50,
                   help="How often to print system summary in continuous mode (default: 50)")
    return p.parse_args()


def init_distributed(args):
    role = args.role
    world_size = args.world_size

    if role == "control":
        rank = 0
    elif role == "worker":
        rank = args.rank if args.rank is not None else 1
    else:
        rank = args.rank if args.rank is not None else int(os.environ.get("RANK", 0))

    if args.master_addr != "auto":
        master_addr = args.master_addr
    else:
        master_addr = get_real_lan_ip()

    master_port = args.master_port

    if world_size > 1:
        if master_addr.startswith("127."):
            print("\n[CRITICAL ERROR] Your system is resolving to localhost.")
            print("Please comment out 127.0.1.1 in /etc/hosts on both machines.")
            sys.exit(1)

        print(f"[Distributed] Initializing process group... (role={role}, rank={rank}, world_size={world_size}, master={master_addr})")

        try:
            dist.init_process_group(
                backend=args.backend,
                init_method="env://",
                rank=rank,
                world_size=world_size
            )
            print("[Distributed] Process group initialized successfully.")
        except Exception as e:
            print(f"[Distributed] Failed to initialize process group: {e}")
            sys.exit(1)

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
        self.field_samples: List[torch.Tensor] = []

    def trajectory(self):
        cfg, lat = self.config, self.lattice
        U0 = self.gauge.U.clone()
        P0 = random_su_n_hermitian(U0.shape, cfg.color_dim, cfg.dtype, U0.device, self.generator)
        if self.pseudo:
            self.pseudo.refresh(U0)

        action0 = float(self.gauge.wilson_action(U0).real)
        kinetic0 = 0.5 * torch.sum((P0 @ P0).diagonal(dim1=-2, dim2=-1).sum(-1).real)
        H0 = lat.global_sum(kinetic0) + action0

        if self.diagnostic and self.lattice.rank == 0:
            print(f"  [Diag] Start Action = {action0:.4f}")

        U, P = U0.clone(), P0.clone()
        eps = cfg.hmc_step_size
        F = self.gauge.force(U)
        P += 0.5 * eps * (1j * F)

        for step in range(cfg.hmc_n_leapfrog):
            U = expm_anti_hermitian(eps * (-1j * P)) @ U
            lat.update_halo_gauge(U)
            F = self.gauge.force(U)
            coeff = eps if step < cfg.hmc_n_leapfrog - 1 else 0.5 * eps
            P += coeff * (1j * F)

        action1 = float(self.gauge.wilson_action(U).real)
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
            self.gauge.U = U.clone()
            lat.update_halo_gauge(self.gauge.U)

            if self.diagnostic and self.lattice.rank == 0:
                flat = self.gauge.U.detach().flatten().real[:8192]
                self.field_samples.append(flat)

        self.action_history.append(action1)
        self.delta_h_history.append(delta_h)
        self.accepted_history.append(accepted)

        return {"delta_h": delta_h, "accepted": accepted, "acceptance_rate": self.n_accepted / max(1, self.n_total)}


def main():
    args = parse_args()
    rank, world_size, master_addr, master_port = init_distributed(args)

    if args.continuous:
        hmc_trajectories = 10**9
    else:
        hmc_trajectories = 25

    config = ConfigV2(
        L=4,
        beta=5.5,
        hmc_n_leapfrog=20,
        hmc_step_size=0.012,
        hmc_trajectories=hmc_trajectories,
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

    memory = EpisodicMemory() if (args.diagnostic and rank == 0) else None
    predictor = LatentPredictor().to(torch.float64) if (args.diagnostic and rank == 0) else None
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=5e-4) if predictor is not None else None

    geometry = None

    if rank == 0:
        mode = "DYNAMICAL + fermions" if config.include_fermions else "QUENCHED (gauge only)"
        run_mode = "CONTINUOUS (Ctrl+C to stop)" if args.continuous else f"{hmc_trajectories} trajectories"
        print(f"Starting {mode} HMC ({run_mode}) on {world_size} rank(s)...\n")

    interrupted = False
    summary_interval = args.summary_interval

    try:
        for t in range(config.hmc_trajectories):
            res = hmc.trajectory()

            if rank == 0 and args.diagnostic and memory is not None and len(hmc.field_samples) > 0:
                if geometry is None:
                    input_dim = hmc.field_samples[0].shape[0]
                    geometry = LearnedInformationGeometry(input_dim=input_dim, latent_dim=8)

                with torch.no_grad():
                    z = geometry.encode(hmc.field_samples[-1])

                exp = EpisodicExperience(
                    latent=z,
                    action=hmc.action_history[-1] if hmc.action_history else 0.0,
                    delta_h=res["delta_h"],
                    accepted=res["accepted"],
                )
                memory.add(exp)

                if predictor is not None and len(memory.buffer) > 8 and t % 25 == 0:
                    batch = memory.sample(16)
                    recent = hmc.field_samples[-min(16, len(hmc.field_samples)):]
                    x_batch = torch.stack(recent).to(torch.float64)
                    z_batch = geometry.encode(x_batch)

                    actions = torch.tensor([e.action for e in batch], dtype=torch.float64)
                    target_mean = actions.mean()
                    target_std = actions.std() + 1e-6
                    target = ((actions - target_mean) / target_std).unsqueeze(1)

                    pred = predictor(z_batch)
                    pred_loss = torch.mean((pred - target) ** 2)

                    predictor_optimizer.zero_grad()
                    pred_loss.backward()
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                    predictor_optimizer.step()

                # Configurable periodic system summary
                if args.continuous and t > 0 and t % summary_interval == 0:
                    mem_stats = memory.get_stats()
                    recent_pred_loss = None

                    if predictor is not None and len(memory.buffer) > 8:
                        recent = hmc.field_samples[-min(8, len(hmc.field_samples)):]
                        x_batch = torch.stack(recent).to(torch.float64)
                        z_batch = geometry.encode(x_batch)
                        actions = torch.tensor([e.action for e in memory.sample(8)], dtype=torch.float64)
                        target_mean = actions.mean()
                        target_std = actions.std() + 1e-6
                        target = ((actions - target_mean) / target_std).unsqueeze(1)
                        with torch.no_grad():
                            pred = predictor(z_batch)
                            recent_pred_loss = torch.mean((pred - target) ** 2).item()

                    health = "Good"
                    if recent_pred_loss is not None and recent_pred_loss > 0.1:
                        health = "Warning (high pred loss)"
                    elif mem_stats.get('size', 0) < 20:
                        health = "Building memory..."

                    print(f"[Summary @ {t}] Health: {health} | Mem: {mem_stats.get('size', 0)} | AvgPrio: {mem_stats.get('avg_priority', 0):.2f}", end="")
                    if recent_pred_loss is not None:
                        print(f" | PredLoss: {recent_pred_loss:.2e}")
                    else:
                        print()

            if rank == 0:
                status = "ACCEPTED" if res["accepted"] else "REJECTED"
                print(f"traj {t:02d} | dH={res['delta_h']:+.4f} | {status} (rate={res['acceptance_rate']:.2f})")

    except KeyboardInterrupt:
        interrupted = True
        if rank == 0:
            print("\nInterrupted by user (Ctrl+C). Shutting down cleanly...")

    if rank == 0:
        print("\nRun finished." + (" (interrupted)" if interrupted else ""))

        if args.diagnostic and len(hmc.field_samples) > 10:
            print("\n=== Learned Information Geometry ===")
            input_dim = hmc.field_samples[0].shape[0]
            geometry = LearnedInformationGeometry(input_dim=input_dim, latent_dim=8)

            num_samples = len(hmc.field_samples)
            geom_epochs = max(30, min(200, num_samples // 5))

            loss = geometry.train_on_batch(hmc.field_samples, epochs=geom_epochs)
            print(f"Final AE loss after {geom_epochs} epochs: {loss:.4e}")

            with torch.no_grad():
                z = geometry.encode(hmc.field_samples[-1])
                R = geometry.scalar_curvature(z)

            print(f"\nLatent scalar curvature (improved): {R:.4f}")

            if args.save_plots and HAS_MATPLOTLIB:
                try:
                    z2d, colors = geometry.get_latent_2d(hmc.field_samples)
                    if z2d is not None:
                        plt.figure(figsize=(7, 5))
                        scatter = plt.scatter(z2d[:, 0], z2d[:, 1], c=colors, cmap='viridis', s=50, alpha=0.85)
                        plt.colorbar(scatter, label='Mean |field|')
                        plt.title('2D Latent Space of Field Configurations')
                        plt.tight_layout()
                        plt.savefig('latent_space.png', dpi=150)
                        print("Saved latent_space.png")

                    with torch.no_grad():
                        x = torch.stack(hmc.field_samples[:min(64, len(hmc.field_samples))]).to(torch.float64)
                        x_hat = geometry.decoder(geometry.encoder(x))
                        recon_error = torch.mean((x_hat - x) ** 2, dim=1)

                    plt.figure(figsize=(6, 4))
                    plt.plot(recon_error.numpy())
                    plt.title('Reconstruction Error per Sample')
                    plt.xlabel('Sample')
                    plt.ylabel('MSE')
                    plt.tight_layout()
                    plt.savefig('reconstruction_error.png', dpi=120)
                    print("Saved reconstruction_error.png")
                except Exception as e:
                    print(f"Plot error: {e}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
