#!/usr/bin/env python3
"""
security.py

Backend security overlay for MetaField continuous / control paths.

Goals:
- Prohibit duplicate continuous runs (singleton lock)
- Keep control surfaces closed by default
- Require an explicit token for any future control / overlord commands
- Provide a safe local stats export path (no Redis until auth is ready)

This is intentionally conservative so Aurora merge can proceed without
opening unauthenticated channels.
"""

from __future__ import annotations

import atexit
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _runtime_dir() -> Path:
    """Stable per-user runtime directory for locks and local export."""
    base = os.environ.get("METAFIELD_RUNTIME_DIR")
    if base:
        d = Path(base)
    else:
        d = Path(tempfile.gettempdir()) / "metafield"
    d.mkdir(parents=True, exist_ok=True)
    return d


LOCK_PATH = _runtime_dir() / "continuous.lock"
STATS_PATH = _runtime_dir() / "stats.json"
TOKEN_ENV = "METAFIELD_CONTROL_TOKEN"


# ---------------------------------------------------------------------------
# Continuous singleton lock (duplicate path prohibition)
# ---------------------------------------------------------------------------

class ContinuousLockError(RuntimeError):
    """Raised when another continuous instance already holds the lock."""
    pass


class ContinuousLock:
    """
    File-based singleton lock for --continuous mode.

    Prevents two continuous MetaField processes from running at once
    on the same machine (duplicate continuous path).
    """

    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self.held = False

    def acquire(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                old_pid = int(data.get("pid", -1))
            except Exception:
                old_pid = -1

            if old_pid > 0 and _pid_alive(old_pid):
                raise ContinuousLockError(
                    f"Another continuous MetaField is already running "
                    f"(pid={old_pid}, lock={self.path}). "
                    f"Duplicate continuous path is prohibited."
                )
            # Stale lock from a dead process — remove it
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass

        payload = {
            "pid": os.getpid(),
            "started": _now_iso(),
        }
        self.path.write_text(json.dumps(payload, indent=2))
        self.held = True
        atexit.register(self.release)

    def release(self) -> None:
        if not self.held:
            return
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if int(data.get("pid", -1)) == os.getpid():
                    self.path.unlink(missing_ok=True)
        except Exception:
            pass
        self.held = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Control token (overlord / remote command gate)
# ---------------------------------------------------------------------------

def get_control_token() -> Optional[str]:
    """Return the configured control token, or None if unset."""
    tok = os.environ.get(TOKEN_ENV, "").strip()
    return tok or None


def require_control_token(provided: Optional[str]) -> None:
    """
    Gate any future control / overlord action.

    - If METAFIELD_CONTROL_TOKEN is unset, control is disabled (fail closed).
    - If set, provided token must match (constant-time compare).
    """
    expected = get_control_token()
    if expected is None:
        raise PermissionError(
            f"Control surface is disabled. Set {TOKEN_ENV} to enable "
            f"authenticated overlord/control commands."
        )
    if not provided or not secrets.compare_digest(provided, expected):
        raise PermissionError("Invalid or missing control token.")


def control_enabled() -> bool:
    return get_control_token() is not None


# ---------------------------------------------------------------------------
# Local stats export (no Redis — file only, explicit)
# ---------------------------------------------------------------------------

def write_local_stats(stats: Dict[str, Any], path: Path = STATS_PATH) -> None:
    """
    Write a local stats snapshot for sensing/debug.

    - File-only (no network)
    - Restrictive permissions where the OS allows
    - Safe intermediate until Redis + auth are ready
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, default=str))
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    tmp.replace(path)


def read_local_stats(path: Path = STATS_PATH) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
