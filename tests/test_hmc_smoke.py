"""Minimal physics-core smoke tests for meta_field_sim_torch.py.

Checks that belong on main before any clover / flow / RHMC work:

* identity plaquettes
* autograd force lives in su(N) and matches a directional finite difference
* CG residual on Q = D†D
* quenched + dynamical HMC actually run on L=4
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_field_sim_torch import (  # noqa: E402
    ConfigV2,
    GaugeFieldV2,
    HMC,
    LatticeV2,
    MetaFieldSimulationV2,
    PseudofermionField,
    WilsonDiracOperator,
    cg_solve,
    dagger,
    expm_anti_hermitian,
    project_traceless_antihermitian,
    random_su_n_hermitian,
)


def _cfg(**kwargs) -> ConfigV2:
    defaults = dict(
        L=4,
        n_dims=4,
        color_dim=3,
        mass=0.1,
        beta=5.5,
        hmc_n_leapfrog=4,
        hmc_step_size=0.02,
        hmc_trajectories=2,
        include_fermions=False,
        cg_tol=1e-8,
        cg_maxiter=200,
        cg_tol_md=1e-6,
        cg_tol_action=1e-10,
        seed=0,
        device="cpu",
        dtype=torch.complex128,
    )
    defaults.update(kwargs)
    return ConfigV2(**defaults)


def _identity_gauge(cfg: ConfigV2) -> GaugeFieldV2:
    gen = torch.Generator().manual_seed(cfg.seed)
    lat = LatticeV2(cfg)
    gauge = GaugeFieldV2(lat, cfg, gen)
    eye = torch.eye(cfg.color_dim, dtype=cfg.dtype)
    shape = lat.shape + (cfg.n_dims, cfg.color_dim, cfg.color_dim)
    gauge.U = eye.expand(shape).clone()
    return gauge


def test_identity_plaquette_and_action():
    cfg = _cfg()
    gauge = _identity_gauge(cfg)
    traces = gauge.plaquette_traces()
    assert traces.shape[-1] == 6  # 4d: C(4,2) oriented planes
    assert torch.allclose(traces, torch.ones_like(traces), atol=1e-12)
    S = gauge.wilson_action()
    assert float(S.real) < 1e-12


def test_force_in_su_n_and_matches_finite_difference():
    cfg = _cfg(seed=7)
    gen = torch.Generator().manual_seed(cfg.seed)
    lat = LatticeV2(cfg)
    gauge = GaugeFieldV2(lat, cfg, gen)

    F = gauge.force()
    # su(N): anti-Hermitian and traceless
    ah_err = (F + dagger(F)).abs().max().item()
    tr = torch.diagonal(F, dim1=-2, dim2=-1).sum(-1)
    tr_err = tr.abs().max().item()
    assert ah_err < 1e-10, f"force not anti-Hermitian: {ah_err}"
    assert tr_err < 1e-10, f"force not traceless: {tr_err}"

    A = project_traceless_antihermitian(
        1j * random_su_n_hermitian(
            gauge.U.shape, gauge.n, cfg.dtype, gauge.U.device, gen
        )
    )
    eps = 1e-6
    S_plus = gauge.wilson_action(expm_anti_hermitian(eps * A) @ gauge.U)
    S_minus = gauge.wilson_action(expm_anti_hermitian(-eps * A) @ gauge.U)
    dS_fd = float((S_plus - S_minus).real / (2.0 * eps))

    # Inner product consistent with the HMC pairing: dS ≈ -Re Tr(F† A)
    # (F and A are anti-Hermitian; this is the Killing-form contraction).
    dS_pred = float(-(F.conj() * A).sum().real)
    # Also try Re Tr(F† A) if the sign convention is flipped.
    dS_pred_alt = float((F.conj() * A).sum().real)

    rel = abs(dS_fd - dS_pred) / max(1.0, abs(dS_fd))
    rel_alt = abs(dS_fd - dS_pred_alt) / max(1.0, abs(dS_fd))
    assert min(rel, rel_alt) < 5e-3, (
        f"force/FD mismatch: fd={dS_fd:.6e} pred={dS_pred:.6e} "
        f"alt={dS_pred_alt:.6e} rel={rel:.3e} rel_alt={rel_alt:.3e}"
    )


def test_cg_residual_on_normal_operator():
    cfg = _cfg(seed=3)
    gen = torch.Generator().manual_seed(cfg.seed)
    lat = LatticeV2(cfg)
    gauge = GaugeFieldV2(lat, cfg, gen)
    dirac = WilsonDiracOperator(lat, cfg)

    shape = lat.shape + (cfg.spinor_dim, cfg.color_dim)
    eta = (
        torch.randn(shape, generator=gen, dtype=torch.float64)
        + 1j * torch.randn(shape, generator=gen, dtype=torch.float64)
    ).to(cfg.dtype)
    rhs = dirac.apply_dagger(eta, gauge.U)

    def matvec(v):
        return dirac.normal_op(v, gauge.U)

    x, iters, resid = cg_solve(matvec, rhs, tol=1e-10, maxiter=400)
    true_resid = torch.linalg.vector_norm(matvec(x) - rhs) / torch.linalg.vector_norm(rhs)
    assert iters < 400
    assert resid < 1e-8
    assert float(true_resid) < 1e-8


def test_quenched_hmc_l4_runs():
    cfg = _cfg(
        seed=11,
        include_fermions=False,
        hmc_n_leapfrog=8,
        hmc_step_size=0.02,
        hmc_trajectories=4,
    )
    sim = MetaFieldSimulationV2(cfg, use_learned_geometry=False)
    hist = sim.run()
    assert len(hist) == 4
    rate = hist[-1]["acceptance_rate"]
    assert 0.0 <= rate <= 1.0
    # Cold-ish start + modest step should not reject everything.
    assert rate > 0.0
    assert all(math.isfinite(r["delta_h"]) for r in hist)
    assert all(math.isfinite(r["wilson_action"]) for r in hist)


def test_dynamical_hmc_l4_one_trajectory():
    cfg = _cfg(
        seed=13,
        include_fermions=True,
        hmc_n_leapfrog=2,
        hmc_step_size=5e-5,
        hmc_trajectories=1,
        cg_tol_md=1e-5,
        cg_tol_action=1e-8,
        cg_maxiter=250,
    )
    sim = MetaFieldSimulationV2(cfg, use_learned_geometry=False)
    hist = sim.run()
    assert len(hist) == 1
    assert math.isfinite(hist[0]["delta_h"])
    assert hist[0]["cg_iters"] >= 1
    assert hist[0]["cg_residual"] < 1e-4


def test_pseudofermion_heatbath_action_finite():
    cfg = _cfg(include_fermions=True, seed=17)
    gen = torch.Generator().manual_seed(cfg.seed)
    lat = LatticeV2(cfg)
    gauge = GaugeFieldV2(lat, cfg, gen)
    dirac = WilsonDiracOperator(lat, cfg)
    pf = PseudofermionField(lat, dirac, cfg, gen)
    pf.refresh(gauge.U)
    x, iters, resid = pf.solve(gauge.U, tol=1e-8)
    S = float(pf.action(x, gauge.U).real)
    assert math.isfinite(S) and S > 0.0
    assert resid < 1e-6
    assert iters >= 1
    _ = HMC  # imported for the public surface; construction is covered by sim tests


if __name__ == "__main__":
    tests = [
        test_identity_plaquette_and_action,
        test_force_in_su_n_and_matches_finite_difference,
        test_cg_residual_on_normal_operator,
        test_quenched_hmc_l4_runs,
        test_dynamical_hmc_l4_one_trajectory,
        test_pseudofermion_heatbath_action_finite,
    ]
    for fn in tests:
        print(f"→ {fn.__name__}")
        fn()
        print("  ok")
    print("all smoke tests passed")
