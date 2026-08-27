"""Wilson + HMC core for MetaField.

On main this path was overwritten with a raising placeholder. The known-good
~930-line implementation lives at commit 4588681 (blob 43d9b430). This module
fetches that exact body once, caches it next to the repo, and executes it in
place so imports and the L=4 smoke tests see the real classes.

To vendor the file permanently (no network):

    curl -L https://raw.githubusercontent.com/TheBabelDragon/metafield/458868180717aa684fd34a9c5a71d391a25dd625/meta_field_sim_torch.py \\
      -o meta_field_sim_torch.py
"""
from __future__ import annotations

import pathlib
import runpy
import sys
import urllib.request

_GOOD_COMMIT = "458868180717aa684fd34a9c5a71d391a25dd625"
_GOOD_URL = (
    "https://raw.githubusercontent.com/TheBabelDragon/metafield/"
    f"{_GOOD_COMMIT}/meta_field_sim_torch.py"
)
_CACHE = pathlib.Path(__file__).resolve().parent / ".meta_field_sim_torch.4588681.py"
_MARKER = "class MetaFieldSimulationV2"


def _ensure_body() -> pathlib.Path:
    if _CACHE.exists() and _MARKER.encode() in _CACHE.read_bytes():
        return _CACHE
    req = urllib.request.Request(_GOOD_URL, headers={"User-Agent": "metafield-sim-restore"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    if _MARKER.encode() not in raw:
        raise RuntimeError(
            f"refusing to cache a body that does not define {_MARKER!r} "
            f"(fetched {_GOOD_URL})"
        )
    _CACHE.write_bytes(raw)
    return _CACHE


def _load() -> None:
    body = _ensure_body()
    ns = runpy.run_path(str(body), run_name=__name__)
    g = globals()
    for key, val in ns.items():
        if key == "__name__":
            continue
        g[key] = val


_load()


if __name__ == "__main__":
    # Re-run the vendored __main__ block if the body defined a simulation entry.
    if "MetaFieldSimulationV2" in globals() and "ConfigV2" in globals():
        config = ConfigV2(  # type: ignore[name-defined]
            L=4,
            beta=5.5,
            hmc_n_leapfrog=10,
            hmc_step_size=0.05,
            hmc_trajectories=4,
            include_fermions=False,
            seed=42,
        )
        sim = MetaFieldSimulationV2(config, use_learned_geometry=False)  # type: ignore[name-defined]
        sim.run()
