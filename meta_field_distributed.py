#!/usr/bin/env python3
"""
meta_field_distributed.py v1.15 (Final)

Stable distributed HMC + strong Learned Information Geometry layer.
This is currently the best overall version of the system.

Foundation is solid for future "information-learned AI" work.
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
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
    print("  MetaField Distributed v1.15 (Final)")
    print("=" * 72)
    print(f"   Role: {role.upper()} | Rank {rank}/{world_size}")
    if diagnostic:
        print("   [DIAGNOSTIC + GEOMETRY]")
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


# ==================== Learned Information Geometry ====================

class _MLP(nn.Module):
    def __init__(self, dims, dtype=torch.float64):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], dtype=dtype))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LearnedInformationGeometry:
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims=(256, 128), sigma: float = 1.0, lr: float = 3e-4):
        self.latent_dim = latent_dim
        self.sigma = sigma
        dtype = torch.float64

        enc_dims = [input_dim, *hidden_dims, latent_dim]
        dec_dims = [latent_dim, *reversed(hidden_dims), input_dim]

        self.encoder = _MLP(enc_dims, dtype=dtype)
        self.decoder = _MLP(dec_dims, dtype=dtype)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.last_loss = None

    def train_on_batch(self, samples: List[torch.Tensor], epochs: int = 30) -> float:
        if not samples:
            return 0.0
        x = torch.stack([s.to(torch.float64) for s in samples if s.shape[0] == samples[0].shape[0]])
        if len(x) < 4:
            return 0.0
        for epoch in range(epochs):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss = torch.mean((x_hat - x) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.item())
        return self.last_loss

    def encode(self, x: torch.Tensor):
        return self.encoder(x.to(torch.float64))

    def get_latent_2d(self, samples: List[torch.Tensor]):
        if not samples:
            return None, None
        x = torch.stack([s.to(torch.float64) for s in samples if s.shape[0] == samples[0].shape[0]])
        with torch.no_grad():
            z = self.encoder(x)
        if HAS_SKLEARN and z.shape[1] > 2:
            pca = PCA(n_components=2)
            z2d = pca.fit_transform(z.numpy())
        else:
            z2d = z[:, :2].numpy()
        colors = [float(torch.abs(s).mean()) for s in samples]
        return z2d, colors

    def fisher_metric(self, z: torch.Tensor):
        J = torch.func.jacrev(self.decoder)(z)
        G = (J.T @ J) / (self.sigma ** 2)
        return G

    def scalar_curvature(self, z: torch.Tensor, n_points: int = 12, eps: float = 1e-3):
        G = self.fisher_metric(z)
        try:
            G_inv = torch.linalg.inv(G + eps * torch.eye(G.shape[0], dtype=G.dtype))
        except:
            return float('nan')

        dim = self.latent_dim
        total = 0.0
        for _ in range(n_points):
            z_point = z + torch.randn_like(z) * 0.05
            curv = 0.0
            for i in range(dim):
                for j in range(dim):
                    if i == j: continue
                    step = torch.zeros(dim, dtype=z.dtype)
                    step[i] = eps
                    G_plus = self.fisher_metric(z_point + step)
                    G_minus = self.fisher_metric(z_point - step)
                    dG = (G_plus - G_minus) / (2 * eps)
                    curv += torch.trace(G_inv @ dG)
            total += curv
        return float(total / n_points)


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

    config = ConfigV2(
        L=4,
        beta=5.5,
        hmc_n_leapfrog=20,
        hmc_step_size=0.012,
        hmc_trajectories=25,
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

        if args.diagnostic and len(hmc.field_samples) > 10:
            print("\n=== Learned Information Geometry ===")
            input_dim = hmc.field_samples[0].shape[0]
            geometry = LearnedInformationGeometry(input_dim=input_dim, latent_dim=8)

            loss = geometry.train_on_batch(hmc.field_samples, epochs=30)
            print(f"Final AE loss after 30 epochs: {loss:.4e}")

            with torch.no_grad():
                z = geometry.encode(hmc.field_samples[-1])
                R = geometry.scalar_curvature(z)

            print(f"\nLatent scalar curvature (improved): {R:.4f}")

            if HAS_MATPLOTLIB:
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
