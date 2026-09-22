#!/usr/bin/env python3
"""
meta_field_distributed.py v1.61

Bootstrap: fetch known-good HMC body (commit 4588681), apply patches, run.
Caches to .meta_field_distributed.v161.py — offline after first run.

v1.61: reject --world-size < 1 (was causing numel overflow via negative local_L).
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
_CACHE = pathlib.Path(__file__).resolve().parent / ".meta_field_distributed.v161.py"
_OLD_CACHES = (
    pathlib.Path(__file__).resolve().parent / ".meta_field_distributed.v160.py",
)


def _patch(src: str) -> str:
    src = src.replace('VERSION = "1.58"', 'VERSION = "1.61"', 1)
    src = src.replace(
        "meta_field_distributed.py v1.58\n\n"
        "Nightcap: HMC throughput + geometry-aware episodic interestingness.",
        "meta_field_distributed.py v1.61\n\n"
        "Single-machine first. Reject world_size < 1 (prevents numel overflow).\n"
        "HMC step 5e-5 × 300 leapfrog. See HMC_TUNING.md.\n"
        "Nightcap: HMC throughput + geometry-aware episodic interestingness.",
        1,
    )
    src = src.replace(
        'p.add_argument("--world-size", type=int, default=2)',
        'p.add_argument("--world-size", type=int, default=1, metavar="N",\n'
        '                    help="process count (default: 1 local). Must be >= 1. "\n'
        '                         "Use N>1 only with torchrun / RANK+WORLD_SIZE set.")',
        1,
    )

    if "invalid --world-size=" not in src:
        marker = "def init_distributed(args):\n    role = args.role\n    world_size = args.world_size\n"
        insert = (
            "def init_distributed(args):\n"
            "    role = args.role\n"
            "    world_size = args.world_size\n"
            "    if world_size < 1:\n"
            "        print(\n"
            "            f\"[Distributed] invalid --world-size={world_size}. \"\n"
            "            f\"Must be >= 1 (use 1 for a single local process).\"\n"
            "        )\n"
            "        print(\"  Example: python meta_field_distributed.py --diagnostic --continuous\")\n"
            "        sys.exit(2)\n"
        )
        if marker in src:
            src = src.replace(marker, insert, 1)

    old_check = (
        "        if self.L % world_size != 0:\n"
        "            raise ValueError(\n"
        "                f\"Lattice size L={self.L} must be divisible by world_size={world_size}. \"\n"
        "                f\"Choose L that divides evenly (e.g. L=4 with world_size=1 or 2).\"\n"
        "            )\n"
        "        self.local_L = self.L // world_size\n"
    )
    new_check = (
        "        if world_size < 1:\n"
        "            raise ValueError(\n"
        "                f\"world_size must be >= 1, got {world_size}. \"\n"
        "                f\"For a local single process use --world-size 1 (or omit it).\"\n"
        "            )\n"
        "        if self.L < 1:\n"
        "            raise ValueError(f\"Lattice size L must be >= 1, got {self.L}.\")\n"
        "        if self.L % world_size != 0:\n"
        "            raise ValueError(\n"
        "                f\"Lattice size L={self.L} must be divisible by world_size={world_size}. \"\n"
        "                f\"Choose L that divides evenly (e.g. L=4 with world_size=1 or 2).\"\n"
        "            )\n"
        "        self.local_L = self.L // world_size\n"
        "        if self.local_L < 1:\n"
        "            raise ValueError(\n"
        "                f\"local_L={self.local_L} is invalid (L={self.L}, world_size={world_size}).\"\n"
        "            )\n"
    )
    if old_check in src and "world_size must be >= 1" not in src:
        src = src.replace(old_check, new_check, 1)

    old_eye = (
        "        shape = lattice.local_padded_shape + (4, config.color_dim, config.color_dim)\n"
        "        eye = torch.eye(config.color_dim, dtype=config.dtype, device=lattice.device).expand(shape).clone()"
    )
    new_eye = (
        "        shape = lattice.local_padded_shape + (4, config.color_dim, config.color_dim)\n"
        "        if any(int(d) < 1 for d in shape):\n"
        "            raise ValueError(\n"
        "                f\"Gauge field shape has non-positive dim: {shape}. \"\n"
        "                f\"Check --world-size (>=1) and lattice L.\"\n"
        "            )\n"
        "        try:\n"
        "            eye = torch.eye(config.color_dim, dtype=config.dtype, device=lattice.device).expand(shape).clone()\n"
        "        except RuntimeError as exc:\n"
        "            raise RuntimeError(\n"
        "                f\"Failed to allocate gauge field with shape={shape} \"\n"
        "                f\"(local_padded={lattice.local_padded_shape}, color_dim={config.color_dim}). \"\n"
        "                f\"Original: {exc}\"\n"
        "            ) from exc"
    )
    if old_eye in src:
        src = src.replace(old_eye, new_eye, 1)

    return src


def _ensure_impl() -> pathlib.Path:
    if _CACHE.exists() and b'VERSION = "1.61"' in _CACHE.read_bytes():
        return _CACHE
    for old in _OLD_CACHES:
        if old.exists():
            try:
                old.unlink()
            except OSError:
                pass
    print("[boot] fetching known-good body + applying v1.61 patches…")
    req = urllib.request.Request(_GOOD_URL, headers={"User-Agent": "metafield-v161-bootstrap"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode()
    patched = _patch(raw)
    _CACHE.write_text(patched)
    return _CACHE


if __name__ == "__main__":
    impl = _ensure_impl()
    sys.argv[0] = str(impl)
    runpy.run_path(str(impl), run_name="__main__")
