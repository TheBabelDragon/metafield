#!/usr/bin/env python3
"""
metafield_sensing / entrypoint.py  (v0.2.3)

Schema v4 includes Fisher metric diagnostics + scalar curvature.
"""

from typing import Dict, Any, Optional
import math
from pathlib import Path

try:
    from security import read_local_stats, STATS_PATH, control_enabled
except ImportError:
    read_local_stats = None
    STATS_PATH = Path("/tmp/metafield/stats.json")
    def control_enabled():
        return False


def get_metafield_stats() -> Dict[str, Any]:
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
    live = data.get("live", health not in ("stopped", "no_export"))

    return {
        "schema_version": data.get("schema_version", 1),
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
    stats = get_metafield_stats()

    if not stats.get("live"):
        health = stats.get("health", "no_export")
        if health == "stopped":
            print(f"[metafield_sensing] process stopped (at {stats.get('stopped_at', '?')})")
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
    logdet = geom.get("metric_logdet")
    if logdet is not None and not (isinstance(logdet, float) and math.isnan(logdet)):
        parts.append(f"logdetG={logdet:.2f}")
    R = geom.get("scalar_curvature")
    if R is not None and not (isinstance(R, float) and math.isnan(R)):
        parts.append(f"R={R:.2e}")

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
    print("[metafield_sensing] Ready (read-only local stats, schema v4)")


if __name__ == "__main__":
    register()
    on_sensing_tick()
