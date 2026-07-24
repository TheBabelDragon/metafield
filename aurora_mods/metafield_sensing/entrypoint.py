#!/usr/bin/env python3
"""
metafield_sensing / entrypoint.py

MetaField sensing integration mod for Aurora Swarm.

Reads local stats export only (file-based). No Redis / network publish
until control token + overlord security overlay are fully in place.
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
            "memory": {"size": 0, "soft_capacity": 0},
            "attractors": {"num_attractors": 0, "total_energy": 0.0},
            "prediction": {"recent_loss": None},
            "health": "no_export",
            "control_enabled": control_enabled(),
            "source": str(STATS_PATH),
        }

    return {
        "memory": data.get("memory", {}),
        "attractors": data.get("attractors", {}),
        "prediction": data.get("prediction", {}),
        "health": data.get("health", "unknown"),
        "traj": data.get("traj"),
        "control_enabled": data.get("control_enabled", control_enabled()),
        "source": str(STATS_PATH),
    }


def on_sensing_tick(context: Any = None) -> None:
    """
    Periodic sensing tick.

    Read-only. Does not publish to Redis until security overlay allows it.
    """
    stats = get_metafield_stats()
    # Local log only — no network
    print(f"[metafield_sensing] tick health={stats.get('health')} "
          f"mem={stats.get('memory', {}).get('size')} "
          f"attractors={stats.get('attractors', {}).get('num_attractors')}")


def on_node_status(context: Any = None) -> None:
    pass


def register() -> None:
    print("[metafield_sensing] Registering hooks (local-file mode, no Redis)...")
    print("[metafield_sensing] Control surface: "
          + ("enabled" if control_enabled() else "disabled"))
    print("[metafield_sensing] Ready (read-only local stats)")


if __name__ == "__main__":
    register()
    on_sensing_tick()
