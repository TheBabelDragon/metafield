#!/usr/bin/env python3
"""
credit_mint.py

Internal swarm-credit mint — token-gated, Python-route, residual-hardened.

Residual controls (v final):
- continuous.lock must name a *live* PID
- stats.json mtime must be fresh
- runtime dir must not be world-writable
- claims sealed with HMAC-SHA256 over evidence_hash using control token
- finite metrics, abs(|dH|), credit caps, flock, env clamps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from security import (
    STATS_PATH,
    get_control_token,
    require_control_token,
    control_enabled,
    _runtime_dir,
    read_local_stats,
    continuous_owner_alive,
    assert_runtime_dir_safe,
    stats_freshness_seconds,
)
from schemas.work_claim import WorkClaim


CLAIMS_PATH = _runtime_dir() / "work_claims.jsonl"
MINT_STATE_PATH = _runtime_dir() / "mint_state.json"
MINT_LOCK_PATH = _runtime_dir() / "mint.lock"


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


MIN_TRAJ_DELTA = _env_int("METAFIELD_MINT_MIN_TRAJ_DELTA", 10, 1, 1_000_000)
MIN_ACCEPT = _env_float("METAFIELD_MINT_MIN_ACCEPT", 0.35, 0.0, 1.0)
MAX_ABS_DH = _env_float("METAFIELD_MINT_MAX_ABS_DH", 3.0, 0.01, 1e6)
COOLDOWN_SEC = _env_float("METAFIELD_MINT_COOLDOWN_SEC", 60.0, 1.0, 86400.0)
BASE_CREDIT = _env_float("METAFIELD_MINT_BASE_CREDIT", 1.0, 0.0, 100.0)
MAX_CREDIT_PER_CLAIM = _env_float("METAFIELD_MINT_MAX_CREDIT", 10.0, 0.0, 1000.0)
MAX_TOTAL_CREDIT = _env_float("METAFIELD_MINT_MAX_TOTAL", 1_000_000.0, 1.0, 1e12)
MAX_CLAIMS_FILE_BYTES = _env_int("METAFIELD_MINT_MAX_CLAIMS_BYTES", 50_000_000, 1_000_000, 500_000_000)
MAX_STATS_AGE_SEC = _env_float("METAFIELD_MINT_MAX_STATS_AGE", 120.0, 5.0, 3600.0)
REQUIRE_LIVE_CONTINUOUS = os.environ.get("METAFIELD_MINT_REQUIRE_CONTINUOUS", "1").strip().lower() not in (
    "0", "false", "no", "off",
)


class CreditMintError(RuntimeError):
    pass


def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


class _MintFileLock:
    def __init__(self, path: Path = MINT_LOCK_PATH):
        self.path = path
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            os.close(self._fd)
            self._fd = None
            return False
        except Exception:
            return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None


class CreditMint:
    def __init__(
        self,
        claims_path: Path = CLAIMS_PATH,
        state_path: Path = MINT_STATE_PATH,
        stats_path: Path = STATS_PATH,
    ):
        self.claims_path = claims_path
        self.state_path = state_path
        self.stats_path = stats_path

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"last_traj": -1, "last_mint_ts": 0.0, "total_credit": 0.0}
        try:
            data = json.loads(self.state_path.read_text())
            if not isinstance(data, dict):
                raise ValueError("state not object")
            return data
        except Exception:
            return {"last_traj": -1, "last_mint_ts": 0.0, "total_credit": 0.0}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            pass
        tmp.replace(self.state_path)

    def _append_claim(self, claim: WorkClaim) -> None:
        self.claims_path.parent.mkdir(parents=True, exist_ok=True)
        if self.claims_path.exists() and self.claims_path.stat().st_size > MAX_CLAIMS_FILE_BYTES:
            raise CreditMintError(
                f"claims log exceeds {MAX_CLAIMS_FILE_BYTES} bytes — rotate/archive before minting"
            )
        line = json.dumps(claim.to_dict(), default=str) + "\n"
        with open(self.claims_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(self.claims_path, 0o600)
        except Exception:
            pass

    def _score_credit(self, stats: Dict[str, Any]) -> Tuple[float, str]:
        hmc = stats.get("hmc") if isinstance(stats.get("hmc"), dict) else {}
        accept = max(0.0, min(1.0, _finite(hmc.get("acceptance_rate"), 0.0)))
        abs_dh = abs(_finite(hmc.get("recent_abs_dh"), 0.0))
        health = str(stats.get("health") or "unknown")[:128]

        if not stats.get("live", False):
            return 0.0, "not_live"
        if health in ("stopped", "no_export"):
            return 0.0, f"health={health}"
        if accept < MIN_ACCEPT:
            return 0.0, f"accept={accept:.2f}<{MIN_ACCEPT}"
        if abs_dh > MAX_ABS_DH:
            return 0.0, f"|dH|={abs_dh:.2f}>{MAX_ABS_DH}"

        quality = min(1.0, accept / 0.65) * max(0.2, 1.0 - (abs_dh / MAX_ABS_DH))
        if not math.isfinite(quality):
            return 0.0, "non_finite_quality"
        credit = min(MAX_CREDIT_PER_CLAIM, BASE_CREDIT * quality)
        if not math.isfinite(credit) or credit <= 0.0:
            return 0.0, "non_positive_credit"
        reason = f"accept={accept:.2f} |dH|={abs_dh:.2f} health={health} q={quality:.2f}"
        return round(credit, 4), reason

    def _assert_evidence_binding(self) -> None:
        """Residual: refuse mint without live continuous owner + fresh stats."""
        assert_runtime_dir_safe()

        if REQUIRE_LIVE_CONTINUOUS:
            alive, pid = continuous_owner_alive()
            if not alive:
                raise CreditMintError(
                    f"no live continuous owner (lock pid={pid}). "
                    f"Start MetaField --continuous before minting."
                )

        age = stats_freshness_seconds(self.stats_path)
        if age is None:
            raise CreditMintError(f"no stats at {self.stats_path}")
        if age > MAX_STATS_AGE_SEC:
            raise CreditMintError(
                f"stats stale ({age:.0f}s > {MAX_STATS_AGE_SEC:.0f}s). "
                f"Need live --export-stats writer."
            )

    def try_mint_from_stats(
        self,
        token: Optional[str] = None,
        force_token_from_env: bool = True,
    ) -> Optional[WorkClaim]:
        provided = token
        if provided is None and force_token_from_env:
            provided = get_control_token()
        require_control_token(provided)
        assert provided is not None

        lock = _MintFileLock()
        if not lock.acquire():
            raise CreditMintError("another mint process holds mint.lock")

        try:
            self._assert_evidence_binding()

            stats = read_local_stats(self.stats_path)
            if not stats or not isinstance(stats, dict):
                raise CreditMintError(f"no stats at {self.stats_path} (run with --export-stats)")

            state = self._load_state()
            try:
                traj = int(stats.get("traj") or 0)
            except (TypeError, ValueError):
                traj = 0
            traj = max(0, min(traj, 10**12))
            now = time.time()

            last_traj = int(state.get("last_traj", -1) or -1)
            last_ts = _finite(state.get("last_mint_ts"), 0.0)
            total = _finite(state.get("total_credit"), 0.0)

            if total >= MAX_TOTAL_CREDIT:
                raise CreditMintError(f"total_credit cap {MAX_TOTAL_CREDIT} reached")

            if traj - last_traj < MIN_TRAJ_DELTA:
                return None
            if now - last_ts < COOLDOWN_SEC:
                return None

            credit, reason = self._score_credit(stats)
            if credit <= 0.0:
                return None

            credit = min(credit, max(0.0, MAX_TOTAL_CREDIT - total))
            if credit <= 0.0:
                return None

            hmc = stats.get("hmc") if isinstance(stats.get("hmc"), dict) else {}
            mem = stats.get("memory") if isinstance(stats.get("memory"), dict) else {}
            att = stats.get("attractors") if isinstance(stats.get("attractors"), dict) else {}

            alive, cont_pid = continuous_owner_alive()

            claim = WorkClaim(
                node_id=str(stats.get("version", "metafield") or "metafield")[:64],
                traj=traj,
                acceptance_rate=max(0.0, min(1.0, _finite(hmc.get("acceptance_rate"), 0.0))),
                recent_abs_dh=abs(_finite(hmc.get("recent_abs_dh"), 0.0)),
                health=str(stats.get("health") or "unknown")[:128],
                live=bool(stats.get("live", False)),
                memory_size=max(0, int(mem.get("size") or 0)),
                num_attractors=max(0, int(att.get("num_attractors") or 0)),
                credit=credit,
                reason=reason[:256],
                timestamp=now,
                extras={
                    "schema_version_stats": stats.get("schema_version"),
                    "continuous_pid": cont_pid if alive else None,
                    "stats_age_sec": stats_freshness_seconds(self.stats_path),
                    "mint_regulation": {
                        "min_traj_delta": MIN_TRAJ_DELTA,
                        "min_accept": MIN_ACCEPT,
                        "max_abs_dh": MAX_ABS_DH,
                        "cooldown_sec": COOLDOWN_SEC,
                        "base_credit": BASE_CREDIT,
                        "max_credit_per_claim": MAX_CREDIT_PER_CLAIM,
                        "max_total_credit": MAX_TOTAL_CREDIT,
                        "max_stats_age_sec": MAX_STATS_AGE_SEC,
                    },
                },
            ).seal(token=provided)

            if not claim.verify_mac(provided):
                raise CreditMintError("internal MAC verify failed after seal")

            self._append_claim(claim)
            state["last_traj"] = traj
            state["last_mint_ts"] = now
            state["total_credit"] = round(total + credit, 6)
            state["last_claim_id"] = claim.claim_id
            self._save_state(state)
            return claim
        finally:
            lock.release()

    def total_credit(self) -> float:
        return _finite(self._load_state().get("total_credit"), 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description="MetaField token-gated swarm credit mint")
    p.add_argument("--watch", action="store_true", help="poll stats and mint when eligible")
    p.add_argument("--interval", type=float, default=15.0, help="watch poll seconds")
    args = p.parse_args()

    if not control_enabled():
        print(
            "[mint] control surface disabled. "
            "Set METAFIELD_CONTROL_TOKEN in the environment to enable minting."
        )
        raise SystemExit(2)

    interval = args.interval if math.isfinite(args.interval) else 15.0
    interval = max(1.0, min(3600.0, interval))

    mint = CreditMint()
    print(f"[mint] claims → {CLAIMS_PATH}")
    print(f"[mint] stats  ← {STATS_PATH}")
    print(
        f"[mint] residual: live_continuous={REQUIRE_LIVE_CONTINUOUS} "
        f"max_stats_age={MAX_STATS_AGE_SEC}s HMAC=on"
    )
    print(
        f"[mint] regulation: Δtraj≥{MIN_TRAJ_DELTA} accept≥{MIN_ACCEPT} "
        f"|dH|≤{MAX_ABS_DH} cooldown={COOLDOWN_SEC}s base={BASE_CREDIT} "
        f"cap/claim={MAX_CREDIT_PER_CLAIM} cap/total={MAX_TOTAL_CREDIT}"
    )

    def once() -> None:
        try:
            claim = mint.try_mint_from_stats()
        except PermissionError as e:
            print(f"[mint] denied: {e}")
            raise SystemExit(2)
        except CreditMintError as e:
            print(f"[mint] skip: {e}")
            return
        if claim is None:
            print(f"[mint] no claim (regulation). total_credit={mint.total_credit():.4f}")
        else:
            print(
                f"[mint] CLAIM {claim.claim_id} credit={claim.credit:.4f} "
                f"traj={claim.traj} mac={claim.mac[:12]}… "
                f"({claim.reason}) total={mint.total_credit():.4f}"
            )

    if not args.watch:
        once()
        return

    print("[mint] watch mode — Ctrl+C to stop")
    try:
        while True:
            once()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[mint] stopped")


if __name__ == "__main__":
    main()
