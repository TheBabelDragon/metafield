#!/usr/bin/env python3
"""
aurora_feed.py

Read-only Aurora environment feed for MetaField.

Provides:
  - Start prompt / initial context from Aurora sensing + swarm state
  - Live statistics used as drive force (exploration, energy pressure, interest bias)

Security stance:
  - Read-only by default (no publish, no commands)
  - Redis optional; degrades gracefully if unavailable
  - Never opens control surface without METAFIELD_CONTROL_TOKEN
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DriveForce:
    """
    Normalized drive signals for MetaField dynamics.

    All values are multipliers or biases in sensible ranges.
    """
    exploration_scale: float = 1.0   # multiply exploration rate
    energy_scale: float = 1.0        # multiply effective energy budget pressure
    interest_bias: float = 0.0       # added to interestingness gate threshold adjustment
    activity: float = 0.5            # 0 = quiet, 1 = busy environment
    mode: str = "neutral"            # neutral | scale_up | scale_down | security
    reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class AuroraFeed:
    """
    Optional live feed from Aurora Redis sensing channels.

    Channels (read):
      aurora:sensing:context
      aurora:sensing:events
      aurora:sensing:heartbeat
      (and plain GET keys used by SensingIntegration)
    """

    def __init__(self,
                 redis_url: Optional[str] = None,
                 enabled: bool = True,
                 stale_seconds: float = 45.0):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.enabled = enabled
        self.stale_seconds = stale_seconds
        self._r = None
        self._last_context: Dict[str, Any] = {}
        self._last_poll = 0.0
        self._available = False

        if enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import redis
            self._r = redis.from_url(self.redis_url, decode_responses=True, socket_connect_timeout=1.5)
            self._r.ping()
            self._available = True
        except Exception:
            self._r = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._r is not None

    def _get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            val = self._r.get(key)
            if not val:
                return None
            if isinstance(val, str):
                return json.loads(val)
            return val if isinstance(val, dict) else None
        except Exception:
            return None

    def poll(self) -> Dict[str, Any]:
        """Pull latest environment snapshot (best-effort)."""
        if not self.enabled:
            return {"status": "disabled"}

        if not self.available:
            self._connect()
            if not self.available:
                return {"status": "unavailable", "reason": "redis_unreachable"}

        ctx = (
            self._get_json("aurora:sensing:context")
            or self._get_json("sensing:context")
            or {}
        )
        hb = self._get_json("aurora:sensing:heartbeat") or self._get_json("sensing:heartbeat") or {}
        events = self._get_json("aurora:sensing:events") or {}

        # Age check
        ts = 0.0
        if isinstance(hb, dict):
            ts = float(hb.get("timestamp", 0) or 0)
        age = time.time() - ts if ts > 0 else None
        stale = age is not None and age > self.stale_seconds

        snapshot = {
            "status": "stale" if stale else "live",
            "heartbeat_age": age,
            "context": ctx if isinstance(ctx, dict) else {},
            "events": events if isinstance(events, dict) else events,
            "heartbeat": hb if isinstance(hb, dict) else {},
            "polled_at": time.time(),
        }
        self._last_context = snapshot
        self._last_poll = time.time()
        return snapshot

    def start_prompt(self) -> str:
        """Human-readable start context for continuous runs."""
        snap = self.poll()
        if snap.get("status") == "disabled":
            return "[Aurora] Feed disabled."
        if snap.get("status") == "unavailable":
            return "[Aurora] No live feed (Redis unreachable). Running local-only."

        ctx = snap.get("context") or {}
        tracks = ctx.get("tracks") or []
        events = ctx.get("events") or []
        n_tracks = len(tracks) if isinstance(tracks, list) else 0
        status = snap.get("status")
        age = snap.get("heartbeat_age")
        age_s = f"{age:.0f}s" if isinstance(age, (int, float)) else "?"

        lines = [
            f"[Aurora] Feed: {status} (heartbeat age {age_s})",
            f"[Aurora] Environment tracks: {n_tracks}",
        ]
        if events:
            lines.append(f"[Aurora] Recent events: {str(events)[:120]}")

        drive = self.drive_force(snap)
        lines.append(
            f"[Aurora] Drive force: mode={drive.mode} "
            f"explore×{drive.exploration_scale:.2f} "
            f"energy×{drive.energy_scale:.2f} "
            f"({drive.reason or 'neutral'})"
        )
        return "\n".join(lines)

    def drive_force(self, snapshot: Optional[Dict[str, Any]] = None) -> DriveForce:
        """
        Translate Aurora environment into MetaField drive force.

        Mirrors Aurora PolicyEngine spirit:
          high occupancy → scale_down (less exploration, tighter energy)
          empty → scale_up (more exploration)
          anomaly → security (raise interest sensitivity)
        """
        snap = snapshot or self._last_context or self.poll()
        if snap.get("status") in ("disabled", "unavailable"):
            return DriveForce(mode="neutral", reason="no_aurora")

        ctx = snap.get("context") or {}
        tracks = ctx.get("tracks") or []
        events = ctx.get("events") or []
        n_tracks = len(tracks) if isinstance(tracks, list) else 0

        event_str = str(events).upper()
        if "ANOMALY" in event_str or "SECURITY" in event_str:
            return DriveForce(
                exploration_scale=0.7,
                energy_scale=0.85,
                interest_bias=-0.15,  # lower gate → more memories feed attractors
                activity=min(1.0, 0.5 + 0.1 * n_tracks),
                mode="security",
                reason="physical_anomaly_or_security",
                raw=snap,
            )

        if n_tracks >= 3:
            return DriveForce(
                exploration_scale=0.75,
                energy_scale=0.8,
                interest_bias=0.05,
                activity=min(1.0, 0.4 + 0.15 * n_tracks),
                mode="scale_down",
                reason="high_occupancy",
                raw=snap,
            )

        if n_tracks == 0:
            return DriveForce(
                exploration_scale=1.25,
                energy_scale=1.15,
                interest_bias=-0.05,
                activity=0.2,
                mode="scale_up",
                reason="empty_environment",
                raw=snap,
            )

        return DriveForce(
            exploration_scale=1.0,
            energy_scale=1.0,
            interest_bias=0.0,
            activity=0.4 + 0.1 * n_tracks,
            mode="neutral",
            reason="normal",
            raw=snap,
        )

    def get_stats(self) -> Dict[str, Any]:
        snap = self._last_context or {}
        drive = self.drive_force(snap if snap else None)
        return {
            "available": self.available,
            "status": snap.get("status", "unknown"),
            "mode": drive.mode,
            "exploration_scale": drive.exploration_scale,
            "energy_scale": drive.energy_scale,
            "interest_bias": drive.interest_bias,
            "activity": drive.activity,
            "reason": drive.reason,
        }
