#!/usr/bin/env python3
"""
meta_field_distributed.py v1.59

Bootstrap: fetch last known-good source, apply HMC default corrections, then run.
See HMC_TUNING.md. Once this is stable in-tree, the full file can replace this loader.
"""
from __future__ import annotations

import pathlib
import re
import runpy
import sys
import tempfile
import urllib.request

# Commit that still has the full pre-placeholder implementation
_GOOD_URL = (
    "https://raw.githubusercontent.com/TheBabelDragon/metafield/"
    "458868180717aa684fd34a9c5a71d391a25dd625/meta_field_distributed.py"
)

_CACHE = pathlib.Path(__file__).resolve().parent / ".meta_field_distributed.v159.py"


def _patch(src: str) -> str:
    src = src.replace('VERSION = "1.58"', 'VERSION = "1.59"', 1)
    src = src.replace(
        "meta_field_distributed.py v1.58\n\n"
        "Nightcap: HMC throughput + geometry-aware episodic interestingness.",
        "meta_field_distributed.py v1.59\n\n"
        "HMC defaults tightened for dynamical fermions (target ~50–70% accept).\n"
        "Nightcap: HMC throughput + geometry-aware episodic interestingness.",
        1,
    )
    src = src.replace(
        "    # Nightcap defaults: slightly smaller step for ~50% accept, keep τ useful\n"
        "    if args.include_fermions:\n"
        "        leapfrog = args.hmc_leapfrog if args.hmc_leapfrog is not None else 75\n"
        "        step_size = args.hmc_step if args.hmc_step is not None else 0.0002\n"
        "    else:\n"
        "        leapfrog = args.hmc_leapfrog if args.hmc_leapfrog is not None else 20\n"
        "        step_size = args.hmc_step if args.hmc_step is not None else 0.012",
        "    # Dynamical defaults tuned for energy conservation (see HMC_TUNING.md).\n"
        "    # Prior 0.0002×75 → |ΔH|~2–5 and ~20% accept. Halve ε, double L (same τ).\n"
        "    if args.include_fermions:\n"
        "        leapfrog = args.hmc_leapfrog if args.hmc_leapfrog is not None else 150\n"
        "        step_size = args.hmc_step if args.hmc_step is not None else 0.0001\n"
        "    else:\n"
        "        leapfrog = args.hmc_leapfrog if args.hmc_leapfrog is not None else 20\n"
        "        step_size = args.hmc_step if args.hmc_step is not None else 0.012",
        1,
    )
    src = src.replace(
        "        if config.include_fermions:\n"
        "            print(f\"  CG tol_md={config.cg_tol_md}  tol_action={config.cg_tol_action}\")\n"
        "        print()",
        "        if config.include_fermions:\n"
        "            print(f\"  CG tol_md={config.cg_tol_md}  tol_action={config.cg_tol_action}\")\n"
        "            print(\"  Target: |ΔH| ≲ 1 and accept ~0.5–0.7  "
        "(override with --hmc-step / --hmc-leapfrog)\")\n"
        "        print()",
        1,
    )
    # Early tuning hint after traj print
    needle = (
        '                print(f"traj {t:02d} | dH={dh_s} | {status} '
        "(rate={res['acceptance_rate']:.2f}){extra}\")\n\n"
        "    except KeyboardInterrupt:"
    )
    insert = (
        '                print(f"traj {t:02d} | dH={dh_s} | {status} '
        "(rate={res['acceptance_rate']:.2f}){extra}\")\n\n"
        "                # Early tuning hint when MD is clearly too aggressive\n"
        "                if args.diagnostic and t == 14 and hmc.n_total >= 15:\n"
        "                    recent = [d for d in hmc.delta_h_history[-15:] if math.isfinite(d)]\n"
        "                    mean_abs = sum(abs(d) for d in recent) / max(1, len(recent))\n"
        "                    rate = res[\"acceptance_rate\"]\n"
        "                    if rate < 0.35 or mean_abs > 1.5:\n"
        "                        sug_step = step_size * 0.5\n"
        "                        sug_lf = max(leapfrog * 2, leapfrog + 1)\n"
        "                        print(\n"
        "                            f\"  [HMC tune] mean|ΔH|≈{mean_abs:.2f} accept={rate:.2f} — \"\n"
        "                            f\"try --hmc-step {sug_step:.6g} --hmc-leapfrog {sug_lf} \"\n"
        "                            f\"(see HMC_TUNING.md)\"\n"
        "                        )\n\n"
        "    except KeyboardInterrupt:"
    )
    if needle in src:
        src = src.replace(needle, insert, 1)
    return src


def _ensure_impl() -> pathlib.Path:
    if _CACHE.exists() and b'VERSION = "1.59"' in _CACHE.read_bytes():
        return _CACHE
    print("[boot] fetching known-good meta_field_distributed body + applying v1.59 HMC patch…")
    req = urllib.request.Request(_GOOD_URL, headers={"User-Agent": "metafield-v159-bootstrap"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode()
    patched = _patch(raw)
    _CACHE.write_text(patched)
    return _CACHE


if __name__ == "__main__":
    impl = _ensure_impl()
    sys.argv[0] = str(impl)
    runpy.run_path(str(impl), run_name="__main__")
