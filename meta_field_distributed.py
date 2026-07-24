#!/usr/bin/env python3
"""
meta_field_distributed.py v1.47

Aurora environment feed (read-only): start prompt + live drive force.
Security overlay + continuous singleton lock retained.
Richer local stats export for sensing (file-only).
"""

from __future__ import annotations

import os
import sys
import math
import socket
import argparse
from typing import List, Optional

import torch
import torch.distributed as dist

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from memory import EpisodicMemory, EpisodicExperience
from prediction import LatentPredictor
from geometry import LearnedInformationGeometry
from attractors import AttractorDynamics
from security import ContinuousLock, ContinuousLockError, write_local_stats, control_enabled
from aurora_feed import AuroraFeed

from meta_field_sim_torch import (
    ConfigV2,
    dagger,
    project_traceless_antihermitian,
    project_traceless_hermitian,
    expm_anti_hermitian,
    random_su_n_hermitian,
    euclidean_gamma_matrices,
    gamma5,
)

VERSION = "1.47"


def get_real_lan_ip() -> str:
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def print_banner(rank, world_size, role, diagnostic=False):
    print("\n" + "=" * 72)
    print(f"  MetaField Distributed v{VERSION} (Aurora Environment Feed)")
    print("=" * 72)
    print(f"   Role: {role.upper()} | Rank {rank}/{world_size}")
    if diagnostic:
        print("   [DIAGNOSTIC + MEMORY + ATTRACTORS + AURORA FEED]")
    ctrl = "enabled" if control_enabled() else "disabled (set METAFIELD_CONTROL_TOKEN to enable)"
    print(f"   Control surface: {ctrl}")
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
        if v.lower() in ("true", "1", "yes", "y"):
            return True
        if v.lower() in ("false", "0", "no", "n"):
            return False
        return True

    p.add_argument("--include-fermions", type=parse_bool, nargs="?", const=True, default=True)
    p.add_argument("--diagnostic", action="store_true", default=False)
    p.add_argument("--save-plots", action="store_true", default=False)
    p.add_argument("--continuous", action="store_true", default=False)
    p.add_argument("--summary-interval", type=int, default=30,
                   help="How often (in trajectories) to print system summary & export stats")
    p.add_argument("--export-stats", action="store_true", default=False)
    p.add_argument("--aurora-feed", action="store_true", default=False,
                   help="Read Aurora sensing as start prompt + live drive force (read-only)")
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

    master_addr = args.master_addr if args.master_addr != "auto" else get_real_lan_ip()
    master_port = args.master_port

    if world_size > 1:
        if master_addr.startswith("127."):
            print("\n[CRITICAL ERROR] Resolving to localhost. Fix /etc/hosts.")
            sys.exit(1)
        print(f"[Distributed] Initializing... rank={rank} world_size={world_size} master={master_addr}")
        try:
            dist.init_process_group(backend=args.backend, init_method="env://", rank=rank, world_size=world_size)
            print("[Distributed] OK")
        except Exception as e:
            print(f"[Distributed] Failed: {e}")
            sys.exit(1)

    print_banner(rank, world_size, role, args.diagnostic)
    return rank, world_size, master_addr, master_port


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


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
        if self.world_size == 1:
            return
        left, right = self.left_neighbor, self.right_neighbor
        recv_l = dist.irecv(tensor.narrow(0, 0, 1).contiguous(), src=left, tag=tag_base)
        recv_r = dist.irecv(tensor.narrow(0, self.local_L + 1, 1).contiguous(), src=right, tag=tag_base + 1)
        send_l = dist.isend(tensor.narrow(0, 1, 1).contiguous(), dst=left, tag=tag_base + 1)
        send_r = dist.isend(tensor.narrow(0, self.local_L, 1).contiguous(), dst=right, tag=tag_base)
        recv_l.wait(); recv_r.wait(); send_l.wait(); send_r.wait()

    def update_halo_gauge(self, U):
        self._halo_exchange(U, 100)

    def shift(self, field, axis, direction):
        if axis != 0 or self.world_size == 1:
            return torch.roll(field, shifts=-direction, dims=axis)
        self._halo_exchange(field, 300 + axis)
        return torch.roll(field, shifts=-direction, dims=0)

    def global_sum(self, x):
        if self.world_size == 1:
            return x
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
        if U is None:
            U = self.U
        lat = self.lattice
        traces = []
        for mu in range(4):
            for nu in range(mu + 1, 4):
                U_mu = U[..., mu, :, :]
                U_nu_xpm = lat.shift(U[..., nu, :, :], mu, +1)
                U_mu_xpn = lat.shift(U_mu, nu, +1)
                plaq = U_mu @ U_nu_xpm @ dagger(U_mu_xpn) @ dagger(U[..., nu, :, :])
                tr = torch.diagonal(plaq, -2, -1).sum(-1).real / self.config.color_dim
                traces.append(tr)
        return lat.global_sum(torch.stack(traces).sum()) / lat.global_volume

    def wilson_action(self, U=None):
        if U is None:
            U = self.U
        return self.config.beta * self.lattice.global_volume * (1.0 - self.plaquette_traces(U))

    def force(self, U=None):
        if U is None:
            U = self.U
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
                torch.einsum("st,...ti->...si", self.r_minus[mu], torch.einsum("...ij,...sj->...si", U_mu, psi_fwd))
                + torch.einsum("st,...ti->...si", self.r_plus[mu], torch.einsum("...ij,...sj->...si", dagger(U_mu_back), psi_back))
            )
        return out

    def apply_dagger(self, psi, U):
        return torch.einsum("st,...ti->...si", self.g5, self.apply(torch.einsum("st,...ti->...si", self.g5, psi), U))


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
            print(f"  [Diag] End Action   = {action1:.4f} | raw dH = {delta_h:+.4f}")

        accept_prob = min(1.0, math.exp(-delta_h)) if delta_h < 700 else 0.0
        accepted = torch.rand((), generator=self.generator).item() < accept_prob

        self.n_total += 1
        if accepted:
            self.n_accepted += 1
            self.gauge.U = U.clone()
            lat.update_halo_gauge(self.gauge.U)
            if self.diagnostic and self.lattice.rank == 0:
                self.field_samples.append(self.gauge.U.detach().flatten().real[:8192])

        self.action_history.append(action1)
        self.delta_h_history.append(delta_h)
        self.accepted_history.append(accepted)
        return {"delta_h": delta_h, "accepted": accepted, "acceptance_rate": self.n_accepted / max(1, self.n_total)}


def apply_drive(feed: Optional[AuroraFeed], memory, attractor_dyn):
    if feed is None:
        return None
    snap = feed.poll()
    drive = feed.drive_force(snap)
    if memory is not None:
        memory.set_drive_scale(drive.exploration_scale)
    if attractor_dyn is not None:
        attractor_dyn.set_drive_scale(drive.energy_scale)
    return drive


def _compute_health(mem_stats, att_stats, recent_pred_loss, acceptance_rate, recent_abs_dh):
    """Produce a compact health string from multiple signals."""
    if mem_stats.get("size", 0) < 20:
        return "Building memory..."
    if recent_pred_loss is not None and recent_pred_loss > 0.15:
        return "Warning (high pred loss)"
    if acceptance_rate < 0.4 and recent_abs_dh is not None and recent_abs_dh > 2.0:
        return "Warning (HMC struggling)"
    if att_stats.get("num_attractors", 0) == 0 and mem_stats.get("size", 0) > 40:
        return "Quiet (no attractors yet)"
    return "Good"


def main():
    args = parse_args()
    rank, world_size, master_addr, master_port = init_distributed(args)

    cont_lock = None
    if args.continuous and rank == 0:
        cont_lock = ContinuousLock()
        try:
            cont_lock.acquire()
            print("[Security] Continuous singleton lock acquired.")
        except ContinuousLockError as e:
            print(f"\n[Security] {e}")
            sys.exit(1)

    aurora = None
    interest_gate = 0.8
    drive = None
    if args.aurora_feed and rank == 0:
        aurora = AuroraFeed(enabled=True)
        print(aurora.start_prompt())

    hmc_trajectories = 10**9 if args.continuous else 25

    config = ConfigV2(
        L=4, beta=5.5, hmc_n_leapfrog=20, hmc_step_size=0.012,
        hmc_trajectories=hmc_trajectories,
        include_fermions=args.include_fermions,
        seed=42 + rank, device="cpu", dtype=torch.complex128,
    )

    torch.manual_seed(config.seed)
    gen = torch.Generator().manual_seed(config.seed + rank * 777)

    lat = DistributedLattice(config, rank, world_size)
    gauge = DistributedGaugeField(lat, config, gen)
    dirac = DistributedWilsonDiracOperator(lat, config)
    pseudo = DistributedPseudofermionField(lat, dirac, config, gen) if config.include_fermions else None
    hmc = DistributedHMC(gauge, dirac, config, gen, pseudo, diagnostic=args.diagnostic)

    memory = EpisodicMemory() if (args.diagnostic and rank == 0) else None
    attractor_dyn = AttractorDynamics() if (args.diagnostic and rank == 0) else None
    predictor = LatentPredictor().to(torch.float64) if (args.diagnostic and rank == 0) else None
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=5e-4) if predictor is not None else None
    geometry = None

    drive = apply_drive(aurora, memory, attractor_dyn)
    if drive is not None:
        interest_gate = max(0.5, min(1.2, 0.8 + drive.interest_bias))

    if rank == 0:
        mode = "DYNAMICAL + fermions" if config.include_fermions else "QUENCHED"
        run_mode = "CONTINUOUS" if args.continuous else f"{hmc_trajectories} traj"
        print(f"Starting {mode} HMC ({run_mode}) on {world_size} rank(s)...\n")

    interrupted = False
    summary_interval = args.summary_interval

    try:
        for t in range(config.hmc_trajectories):
            res = hmc.trajectory()

            if aurora is not None and rank == 0 and t > 0 and t % summary_interval == 0:
                drive = apply_drive(aurora, memory, attractor_dyn)
                if drive is not None:
                    interest_gate = max(0.5, min(1.2, 0.8 + drive.interest_bias))

            if rank == 0 and args.diagnostic and memory is not None and len(hmc.field_samples) > 0:
                if geometry is None:
                    geometry = LearnedInformationGeometry(input_dim=hmc.field_samples[0].shape[0], latent_dim=8)

                with torch.no_grad():
                    z = geometry.encode(hmc.field_samples[-1])

                recon_error = 0.0
                try:
                    with torch.no_grad():
                        x = hmc.field_samples[-1].to(torch.float64).unsqueeze(0)
                        x_hat = geometry.decoder(geometry.encoder(x))
                        recon_error = float(torch.mean((x_hat - x) ** 2).item())
                except Exception:
                    pass

                exp = EpisodicExperience(
                    latent=z,
                    action=hmc.action_history[-1] if hmc.action_history else 0.0,
                    delta_h=res["delta_h"],
                    accepted=res["accepted"],
                )
                exp.update_priority(reconstruction_error=recon_error)
                memory.add(exp)

                if predictor is not None and len(memory.buffer) > 8 and t % 25 == 0:
                    batch = memory.sample(24)
                    for e in batch:
                        if e.interestingness > interest_gate:
                            attractor_dyn.reinforce_from_latent(e.latent, interestingness=e.interestingness)
                    attractor_dyn.step()

                    recent = hmc.field_samples[-min(24, len(hmc.field_samples)):]
                    x_batch = torch.stack(recent).to(torch.float64)
                    z_batch = geometry.encode(x_batch)
                    actions = torch.tensor([e.action for e in batch], dtype=torch.float64)
                    target_mean, target_std = actions.mean(), actions.std() + 1e-6
                    target = ((actions - target_mean) / target_std).unsqueeze(1)
                    pred = predictor(z_batch)
                    pred_loss = torch.mean((pred - target) ** 2)
                    predictor_optimizer.zero_grad()
                    pred_loss.backward()
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                    predictor_optimizer.step()
                    for e in batch:
                        e.update_priority(prediction_error=float(pred_loss.item()), reconstruction_error=recon_error)

                if args.continuous and t > 0 and t % summary_interval == 0:
                    mem_stats = memory.get_stats()
                    att_stats = attractor_dyn.get_stats()
                    recent_pred_loss = None
                    if predictor is not None and len(memory.buffer) > 8:
                        recent = hmc.field_samples[-min(12, len(hmc.field_samples)):]
                        x_batch = torch.stack(recent).to(torch.float64)
                        z_batch = geometry.encode(x_batch)
                        actions = torch.tensor([e.action for e in memory.sample(12)], dtype=torch.float64)
                        tm, ts = actions.mean(), actions.std() + 1e-6
                        target = ((actions - tm) / ts).unsqueeze(1)
                        with torch.no_grad():
                            recent_pred_loss = torch.mean((predictor(z_batch) - target) ** 2).item()

                    # Recent HMC health signals
                    recent_dh = hmc.delta_h_history[-min(20, len(hmc.delta_h_history)):]
                    recent_abs_dh = sum(abs(d) for d in recent_dh) / max(1, len(recent_dh))
                    acceptance_rate = res["acceptance_rate"]

                    health = _compute_health(
                        mem_stats, att_stats, recent_pred_loss,
                        acceptance_rate, recent_abs_dh
                    )

                    aurora_mode = drive.mode if drive is not None else "off"
                    print(
                        f"[Summary @ {t}] Health: {health} | "
                        f"Mem: {mem_stats.get('size', 0)}/{mem_stats.get('soft_capacity', 512)} | "
                        f"AvgInterest: {mem_stats.get('avg_interestingness', 0):.2f} | "
                        f"Attractors: {att_stats.get('num_attractors', 0)} "
                        f"(E={att_stats.get('total_energy', 0):.1f}/{att_stats.get('energy_budget', 80):.0f}) | "
                        f"Explore: {mem_stats.get('exploration_rate', 0):.2f} | "
                        f"Accept: {acceptance_rate:.2f} | "
                        f"Aurora: {aurora_mode}",
                        end="",
                    )
                    if recent_pred_loss is not None:
                        print(f" | PredLoss: {recent_pred_loss:.2e}")
                    else:
                        print()

                    if args.export_stats:
                        write_local_stats({
                            "schema_version": 2,
                            "version": VERSION,
                            "traj": t,
                            "health": health,
                            "hmc": {
                                "acceptance_rate": acceptance_rate,
                                "recent_abs_dh": recent_abs_dh,
                                "n_total": hmc.n_total,
                                "n_accepted": hmc.n_accepted,
                            },
                            "memory": mem_stats,
                            "attractors": att_stats,
                            "prediction": {"recent_loss": recent_pred_loss},
                            "geometry": {"recon_error": recon_error},
                            "aurora": aurora.get_stats() if aurora else {},
                            "control_enabled": control_enabled(),
                        })

            if rank == 0:
                status = "ACCEPTED" if res["accepted"] else "REJECTED"
                print(f"traj {t:02d} | dH={res['delta_h']:+.4f} | {status} (rate={res['acceptance_rate']:.2f})")

    except KeyboardInterrupt:
        interrupted = True
        if rank == 0:
            print("\nInterrupted (Ctrl+C). Shutting down...")

    finally:
        if cont_lock is not None:
            cont_lock.release()
            if rank == 0:
                print("[Security] Continuous lock released.")

    if rank == 0:
        print("\nRun finished." + (" (interrupted)" if interrupted else ""))

    cleanup_distributed()


if __name__ == "__main__":
    main()
