#!/usr/bin/env python3
"""
security.py

Backend security overlay for MetaField continuous / control paths.

Goals:
- Prohibit duplicate continuous runs (singleton lock)
- Keep control surfaces closed by default
- Require an explicit token for any future control / overlord commands
- Provide a safe local stats export path (no Redis until auth is ready)
- Residual helpers for mint: live continuous owner, runtime dir permissions

Escape hatch (still no manual rm):
    METAFIELD_FORCE_UNLOCK=1
"""

from __future__ import annotations

import atexit
import json
import os
import secrets
import socket
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _runtime_dir() -> Path:
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
        d = Path(tempfile.gettempdir()) / f"metafield-{os.getuid() if hasattr(os, 'getuid') else 'user'}"
        d.mkdir(parents=True, exist_ok=True)
    return d


LOCK_PATH = _runtime_dir() / "continuous.lock"
STATS_PATH = _runtime_dir() / "stats.json"
TOKEN_ENV = "METAFIELD_CONTROL_TOKEN"
FORCE_UNLOCK_ENV = "METAFIELD_FORCE_UNLOCK"
STALE_AGE_SECONDS = 6 * 3600


class ContinuousLockError(RuntimeError):
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


def continuous_owner_alive(path: Path = LOCK_PATH) -> Tuple[bool, Optional[int]]:
    """
    True if continuous.lock exists and names a live PID.
    Used by mint so credits require a living continuous process.
    """
    if not path.exists():
        return False, None
    try:
        data = json.loads(path.read_text())
        pid = int(data.get("pid", -1))
    except Exception:
        return False, None
    if not _pid_alive(pid):
        return False, pid
    return True, pid


def assert_runtime_dir_safe(path: Optional[Path] = None) -> None:
    """
    Refuse world-writable runtime dirs (stats/claims poisoning surface).
    Best-effort on platforms without POSIX modes.
    """
    d = path or _runtime_dir()
    try:
        mode = d.stat().st_mode
    except Exception:
        return
    if mode & stat.S_IWOTH:
        raise PermissionError(
            f"Runtime dir is world-writable: {d} (mode={oct(mode & 0o777)}). "
            f"chmod go-w or set METAFIELD_RUNTIME_DIR to a private path."
        )
    # Also warn-level: group-writable on multi-user hosts is risky; fail only if
    # METAFIELD_STRICT_RUNTIME=1
    if os.environ.get("METAFIELD_STRICT_RUNTIME", "").strip() in ("1", "true", "yes"):
        if mode & stat.S_IWGRP:
            raise PermissionError(
                f"Runtime dir is group-writable under STRICT mode: {d}"
            )


class ContinuousLock:
    def __init__(self, path: Path = LOCK_PATH):
        self.path = path
        self.held = False

    def _read_lock(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return None

    def _is_stale(self, data: Optional[Dict[str, Any]]) -> bool:
        if data is None:
            return True
        if os.environ.get(FORCE_UNLOCK_ENV, "").strip() in ("1", "true", "yes", "on"):
            return True
        try:
            old_pid = int(data.get("pid", -1))
        except Exception:
            return True
        if old_pid <= 0 or not _pid_alive(old_pid):
            return True
        return False

    def acquire(self) -> None:
        assert_runtime_dir_safe(self.path.parent)
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
                print(f"[Security] Warning: could not unlink old lock ({e}); "
                      f"attempting overwrite")

        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started": _now_iso(),
        }
        tmp = self.path.with_suffix(".lock.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            try:
                os.chmod(tmp, 0o600)
            except Exception:
                pass
            tmp.replace(self.path)
        except Exception:
            self.path.write_text(json.dumps(payload, indent=2))
        self.held = True
        atexit.register(self.release)

    def release(self, write_stopped_stats: bool = True) -> None:
        if not self.held:
            return
        try:
            data = self._read_lock()
            if data is not None and int(data.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
            elif data is None and self.path.exists():
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


def get_control_token() -> Optional[str]:
    tok = os.environ.get(TOKEN_ENV, "").strip()
    return tok or None


def require_control_token(provided: Optional[str]) -> None:
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


def write_local_stats(stats: Dict[str, Any], path: Path = STATS_PATH) -> None:
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


def stats_freshness_seconds(path: Path = STATS_PATH) -> Optional[float]:
    """Seconds since stats.json mtime, or None if missing."""
    if not path.exists():
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except Exception:
        return None
