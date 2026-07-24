#!/usr/bin/env python3
"""
metafield_sensing / entrypoint.py  (v0.2.2)

MetaField sensing integration mod for Aurora Swarm.

Reads local stats export only (file-based). No Redis / network publish
until control token + overlord security overlay are fully in place.

Schema v3+ includes geometry train loss + curvature.
Clean shutdown now writes health="stopped" so we don't report zombie data.
"""

from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Prefer MetaField's security helper when available on PYTHONPATH
try:
    from security import read_local_stats, STATS_PATH, control_enabled
except ImportError:
    read_local_stats = None
    STATS_PATH = Path("/tmp/metafield/stats.json")
    def control_enabled():
        return False


def get_metafield_stats() -> Dict[str, Any]:
    """
    Collect stats from local MetaField export.

    Source: stats.json written by --export-stats (file-only).
    Does not open network sockets or touch Redis.

    Returns a normalized dict that is safe for Aurora sensing consumers.
    """
    if read_local_stats is not None:
        data = read_local_stats()
    else:
        data = None
        if STATS_PATH.exists():
            try:
                import json
                data = json.loads(STATS_PATH.read_text())
            except Exception:
                data = None

    if data is None:
        return {
            "schema_version": 0,
            "health": "no_export",
            "memory": {"size": 0, "soft_capacity": 0},
            "attractors": {"num_attractors": 0, "total_energy": 0.0},
            "prediction": {"recent_loss": None},
            "hmc": {},
            "geometry": {},
            "aurora": {},
            "control_enabled": control_enabled(),
            "source": str(STATS_PATH),
            "live": False,
        }

    health = data.get("health", "unknown")
    # Explicit stopped signal from clean lock release
    live = data.get("live", health not in ("stopped", "no_export"))

    schema = data.get("schema_version", 1)
    return {
        "schema_version": schema,
        "version": data.get("version", "unknown"),
        "traj": data.get("traj"),
        "health": health,
        "memory": data.get("memory", {}),
        "attractors": data.get("attractors", {}),
        "prediction": data.get("prediction", {}),
        "hmc": data.get("hmc", {}),
        "geometry": data.get("geometry", {}),
        "aurora": data.get("aurora", {}),
        "control_enabled": data.get("control_enabled", control_enabled()),
        "source": str(STATS_PATH),
        "live": live,
        "stopped_at": data.get("stopped_at"),
    }


def on_sensing_tick(context: Any = None) -> None:
    """
    Periodic sensing tick.

    Read-only. Does not publish to Redis until security overlay allows it.
    Prints a compact, operator-friendly line.
    """
    stats = get_metafield_stats()

    if not stats.get("live"):
        health = stats.get("health", "no_export")
        if health == "stopped":
            stopped = stats.get("stopped_at", "?")
            print(f"[metafield_sensing] process stopped (at {stopped})")
        else:
            print("[metafield_sensing] no live export yet")
        return

    mem = stats.get("memory", {})
    att = stats.get("attractors", {})
    hmc = stats.get("hmc", {})
    pred = stats.get("prediction", {})
    geom = stats.get("geometry", {})
    health = stats.get("health", "?")
    traj = stats.get("traj", "?")

    parts = [
        f"traj={traj}",
        f"health={health}",
        f"mem={mem.get('size', 0)}/{mem.get('soft_capacity', '?')}",
        f"attractors={att.get('num_attractors', 0)}",
        f"E={att.get('total_energy', 0):.1f}/{att.get('energy_budget', 80):.0f}",
    ]

    if hmc:
        parts.append(f"accept={hmc.get('acceptance_rate', 0):.2f}")
        if hmc.get("recent_abs_dh") is not None:
            parts.append(f"|dH|={hmc['recent_abs_dh']:.2f}")

    if geom.get("train_loss") is not None:
        parts.append(f"geom={geom['train_loss']:.2e}")
    if geom.get("scalar_curvature") is not None:
        parts.append(f"R={geom['scalar_curvature']:.2e}")

    if pred.get("recent_loss") is not None:
        parts.append(f"pred={pred['recent_loss']:.2e}")

    aurora = stats.get("aurora", {})
    if aurora.get("mode"):
        parts.append(f"aurora={aurora.get('mode')}")

    print("[metafield_sensing] " + " | ".join(parts))


def on_node_status(context: Any = None) -> None:
    pass


def register() -> None:
    print("[metafield_sensing] Registering hooks (local-file mode, no Redis)...")
    print("[metafield_sensing] Control surface: "
          + ("enabled" if control_enabled() else "disabled"))
    print("[metafield_sensing] Ready (read-only local stats, schema v3+)")


if __name__ == "__main__":
    register()
    on_sensing_tick()
