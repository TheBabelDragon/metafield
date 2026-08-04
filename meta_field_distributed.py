#!/usr/bin/env python3
"""
meta_field_distributed.py v1.60

Bootstrap: fetch known-good HMC body (commit 4588681), apply patches, run.
Caches to .meta_field_distributed.v160.py — offline after first run.
See HMC_TUNING.md.
"""
from __future__ import annotations

import pathlib
import runpy
import sys
import urllib.request

_GOOD_URL = (
    "https://raw.githubusercontent.com/TheBabelDragon/metafield/"
    "458868180717aa684fd34a9c5a71d391a25dd625/meta_field_distributed.py"
)
_CACHE = pathlib.Path(__file__).resolve().parent / ".meta_field_distributed.v160.py"


def _patch(src: str) -> str:
    src = src.replace('VERSION = "1.58"', 'VERSION = "1.60"', 1)
    src = src.replace(
        "meta_field_distributed.py v1.58\n\n"
        "Nightcap: HMC throughput + geometry-aware episodic interestingness.",
        "meta_field_distributed.py v1.60\n\n"
        "Single-machine first. HMC step 5e-5 × 300 leapfrog. See HMC_TUNING.md.\n"
        "Nightcap: HMC throughput + geometry-aware episodic interestingness.",
        1,
    )
    src = src.replace(
        'p.add_argument("--world-size", type=int, default=2)',
        'p.add_argument("--world-size", type=int, default=1, metavar="N",\n'
        '                    help="process count (default: 1 local). "\n'
        '                         "Use N>1 only with torchrun / RANK+WORLD_SIZE set.")',
        1,
    )
    old_init = '''def init_distributed(args):
    role = args.role
    world_size = args.world_size
    if role == "control":
        rank = 0
    elif role == "worker":
        rank = args.rank if args.rank is not None else 1
    else:
        rank = args.rank if args.rank is not None else int(os.environ.get("RANK", 0))

    master_addr = args.master_addr if args.master_addr != "auto" else get_real_lan_ip()

    if world_size > 1:
        if master_addr.startswith("127."):
            print("\\n[CRITICAL ERROR] Resolving to localhost. Fix /etc/hosts.")
            sys.exit(1)
        print(f"[Distributed] Initializing... rank={rank} world_size={world_size} master={master_addr}")
        try:
            dist.init_process_group(backend=args.backend, init_method="env://", rank=rank, world_size=world_size)
            print("[Distributed] OK")
        except Exception as e:
            print(f"[Distributed] Failed: {e}")
            sys.exit(1)

    print_banner(rank, world_size, role, args.diagnostic)
    return rank, world_size, master_addr, args.master_port'''
    new_init = '''def init_distributed(args):
    role = args.role
    world_size = args.world_size

    env_world = os.environ.get("WORLD_SIZE")
    env_rank = os.environ.get("RANK")
    if env_world is not None:
        try:
            world_size = int(env_world)
        except ValueError:
            pass

    if role == "control":
        rank = 0
    elif role == "worker":
        rank = args.rank if args.rank is not None else 1
    else:
        if args.rank is not None:
            rank = args.rank
        elif env_rank is not None:
            rank = int(env_rank)
        else:
            rank = 0

    master_addr = args.master_addr if args.master_addr != "auto" else get_real_lan_ip()

    launched = env_world is not None or env_rank is not None
    if world_size > 1 and not launched and args.rank is None and role == "auto":
        print(
            f"[Distributed] world-size={world_size} requested but no RANK/WORLD_SIZE env "
            f"(not under torchrun). Falling back to world-size=1 for local run."
        )
        print("  Tip: for real multi-rank use: torchrun --nproc_per_node=2 meta_field_distributed.py ...")
        world_size = 1
        rank = 0

    if world_size > 1:
        if master_addr.startswith("127."):
            print("\\n[CRITICAL ERROR] master addr resolved to localhost.")
            print("  Fix /etc/hosts hostname mapping, or pass --master-addr <LAN-IP>.")
            sys.exit(1)
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        print(f"[Distributed] Initializing... rank={rank} world_size={world_size} master={master_addr}")
        try:
            dist.init_process_group(
                backend=args.backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
            print("[Distributed] OK")
        except Exception as e:
            print(f"[Distributed] Failed: {e}")
            print("  For a single machine just omit --world-size (defaults to 1).")
            sys.exit(1)

    print_banner(rank, world_size, role, args.diagnostic)
    return rank, world_size, master_addr, args.master_port'''
    if old_init in src:
        src = src.replace(old_init, new_init, 1)
    src = src.replace(
        "    # Nightcap defaults: slightly smaller step for ~50% accept, keep τ useful\n"
        "    if args.include_fermions:\n"
        "        leapfrog = args.hmc_leapfrog if args.hmc_leapfrog is not None else 75\n"
        "        step_size = args.hmc_step if args.hmc_step is not None else 0.0002\n"
        "    else:\n"
        "        leapfrog = args.hmc_leapfrog if args.hmc_leapfrog is not None else 20\n"
        "        step_size = args.hmc_step if args.hmc_step is not None else 0.012",
        "    # Dynamical defaults (HMC_TUNING.md). 5e-5×300 keeps τ≈0.015.\n"
        "    if args.include_fermions:\n"
        "        leapfrog = args.hmc_leapfrog if args.hmc_leapfrog is not None else 300\n"
        "        step_size = args.hmc_step if args.hmc_step is not None else 5e-5\n"
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
    needle = (
        '                print(f"traj {t:02d} | dH={dh_s} | {status} '
        "(rate={res['acceptance_rate']:.2f}){extra}\")\n\n"
        "    except KeyboardInterrupt:"
    )
    insert = (
        '                print(f"traj {t:02d} | dH={dh_s} | {status} '
        "(rate={res['acceptance_rate']:.2f}){extra}\")\n\n"
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
    if _CACHE.exists() and b'VERSION = "1.60"' in _CACHE.read_bytes():
        return _CACHE
    print("[boot] fetching known-good body + applying v1.60 patches…")
    req = urllib.request.Request(_GOOD_URL, headers={"User-Agent": "metafield-v160-bootstrap"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode()
    patched = _patch(raw)
    _CACHE.write_text(patched)
    return _CACHE


if __name__ == "__main__":
    impl = _ensure_impl()
    sys.argv[0] = str(impl)
    runpy.run_path(str(impl), run_name="__main__")
