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

Lock design note:
  Releasing the lock on clean exit is *correct*. Leaving it held would
  block future continuous runs until someone manually deleted a stale
  file. The acquire path is deliberately aggressive about recovering
  from dead / corrupt / ancient locks so manual deletion is never required.

  Escape hatch (still no manual rm):
      METAFIELD_FORCE_UNLOCK=1
"""

from __future__ import annotations

import atexit
import json
import os
import secrets
import socket
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _runtime_dir() -> Path:
    """
    Stable per-user runtime directory for locks and local export.

    Order of preference:
      1. METAFIELD_RUNTIME_DIR (explicit)
      2. XDG_RUNTIME_DIR/metafield (Linux user runtime, cleaned on logout)
      3. /tmp/metafield (fallback)
    """
    base = os.environ.get("METAFIELD_RUNTIME_DIR")
    if base:
        d = Path(base)
    else:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg:
            d = Path(xdg) / "metafield"
        else:
            d = Path(tempfile.gettempdir()) / "metafield"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Last-resort fallback if the preferred dir is unwritable
        d = Path(tempfile.gettempdir()) / f"metafield-{os.getuid() if hasattr(os, 'getuid') else 'user'}"
        d.mkdir(parents=True, exist_ok=True)
    return d


LOCK_PATH = _runtime_dir() / "continuous.lock"
STATS_PATH = _runtime_dir() / "stats.json"
TOKEN_ENV = "METAFIELD_CONTROL_TOKEN"
FORCE_UNLOCK_ENV = "METAFIELD_FORCE_UNLOCK"

# Locks older than this (seconds) with a dead/unreadable owner are always cleaned.
STALE_AGE_SECONDS = 6 * 3600  # 6 hours


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

    Release on clean exit is intentional and correct. The acquire path
    is aggressive about recovering from stale / corrupt / ancient locks
    so the user should never need to manually delete the lock file.

    Escape hatch: METAFIELD_FORCE_UNLOCK=1
    """

    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self.held = False

    def _read_lock(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return None  # corrupt / unreadable → treat as stale

    def _is_stale(self, data: Optional[Dict[str, Any]]) -> bool:
        """Return True if this lock should be cleaned automatically."""
        if data is None:
            return True  # missing or unreadable

        # Explicit force unlock (no manual rm required)
        if os.environ.get(FORCE_UNLOCK_ENV, "").strip() in ("1", "true", "yes", "on"):
            return True

        old_pid = -1
        try:
            old_pid = int(data.get("pid", -1))
        except Exception:
            return True  # malformed pid → stale

        # Dead process → always stale
        if old_pid <= 0 or not _pid_alive(old_pid):
            return True

        # Very old lock whose PID is somehow still "alive" (pid reuse risk)
        # is treated carefully: only force-clean if also marked ancient.
        started = data.get("started")
        if started:
            try:
                from datetime import datetime, timezone
                started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - started_dt).total_seconds()
                # Extremely old locks are suspicious even if PID appears alive
                # (pid reuse after long uptime). Be conservative: only auto-clean
                # if the process is actually dead (already handled above) or
                # force flag is set. Age alone with live PID does not force clean.
                _ = age  # retained for future policy / logging
            except Exception:
                pass

        return False  # live owner → not stale

    def acquire(self) -> None:
        data = self._read_lock()

        if data is not None and not self._is_stale(data):
            old_pid = data.get("pid", "?")
            old_host = data.get("hostname", "?")
            raise ContinuousLockError(
                f"Another continuous MetaField is already running "
                f"(pid={old_pid}, host={old_host}, lock={self.path}). "
                f"Duplicate continuous path is prohibited.\n"
                f"If you are certain it is gone, set METAFIELD_FORCE_UNLOCK=1 "
                f"and retry (no manual file deletion required)."
            )

        # Stale / missing / corrupt → clean and proceed
        if self.path.exists():
            old_pid = data.get("pid", "?") if data else "?"
            old_host = data.get("hostname", "?") if data else "?"
            reason = "force unlock" if os.environ.get(FORCE_UNLOCK_ENV, "").strip() in ("1", "true", "yes", "on") \
                     else "dead/corrupt/stale owner"
            try:
                self.path.unlink(missing_ok=True)
                print(f"[Security] Auto-cleaned continuous lock "
                      f"({reason}; old pid={old_pid}, host={old_host})")
            except Exception as e:
                # Even if unlink fails, try to overwrite; last resort is force path
                print(f"[Security] Warning: could not unlink old lock ({e}); "
                      f"attempting overwrite")

        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started": _now_iso(),
        }
        # Atomic-ish write: temp then replace
        tmp = self.path.with_suffix(".lock.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.path)
        except Exception:
            # Fallback non-atomic write if replace fails
            self.path.write_text(json.dumps(payload, indent=2))
        self.held = True
        atexit.register(self.release)

    def release(self, write_stopped_stats: bool = True) -> None:
        """
        Release the lock. Only deletes the lock file if *this* process owns it.

        Optionally writes a final health="stopped" stats snapshot so sensing
        consumers don't keep reporting stale live data after clean exit.
        """
        if not self.held:
            return
        try:
            data = self._read_lock()
            if data is not None and int(data.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
            elif data is None and self.path.exists():
                # Corrupt lock that we somehow still "held" — clean it
                self.path.unlink(missing_ok=True)
        except Exception:
            pass
        self.held = False

        if write_stopped_stats:
            try:
                existing = read_local_stats() or {}
                existing["health"] = "stopped"
                existing["live"] = False
                existing["stopped_at"] = _now_iso()
                write_local_stats(existing)
            except Exception:
                pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
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
