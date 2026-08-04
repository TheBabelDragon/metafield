#!/usr/bin/env python3
"""
credit_mint.py

Internal swarm-credit mint for MetaField × Aurora regulation.

Principles
----------
- Fail closed: no METAFIELD_CONTROL_TOKEN → no mint.
- Python route only: evidence comes from live/runtime stats (not Git).
- Local append-only claims log (JSONL) under the runtime dir.
- Credits are *internal* regulation units, not external currency / chain assets.
- Self-regulated caps: min traj delta, health gate, acceptance floor, cooldown.

Usage
-----
  export METAFIELD_CONTROL_TOKEN=your-secret
  python credit_mint.py                  # one-shot from stats.json
  python credit_mint.py --watch          # poll while continuous runs

From code:
  from credit_mint import CreditMint
  mint = CreditMint()
  claim = mint.try_mint_from_stats()
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from security import (
    STATS_PATH,
    get_control_token,
    require_control_token,
    control_enabled,
    _runtime_dir,
    read_local_stats,
)
from schemas.work_claim import WorkClaim


CLAIMS_PATH = _runtime_dir() / "work_claims.jsonl"
MINT_STATE_PATH = _runtime_dir() / "mint_state.json"

# Regulation defaults (Aurora can tighten later via env)
MIN_TRAJ_DELTA = int(os.environ.get("METAFIELD_MINT_MIN_TRAJ_DELTA", "10"))
MIN_ACCEPT = float(os.environ.get("METAFIELD_MINT_MIN_ACCEPT", "0.35"))
MAX_ABS_DH = float(os.environ.get("METAFIELD_MINT_MAX_ABS_DH", "3.0"))
COOLDOWN_SEC = float(os.environ.get("METAFIELD_MINT_COOLDOWN_SEC", "60"))
BASE_CREDIT = float(os.environ.get("METAFIELD_MINT_BASE_CREDIT", "1.0"))


class CreditMintError(RuntimeError):
    pass


class CreditMint:
    """
    Token-gated mint. Reads stats.json, applies regulation, appends WorkClaim.
    """

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
            return json.loads(self.state_path.read_text())
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
        line = json.dumps(claim.to_dict(), default=str) + "\n"
        with open(self.claims_path, "a") as f:
            f.write(line)
        try:
            os.chmod(self.claims_path, 0o600)
        except Exception:
            pass

    def _score_credit(self, stats: Dict[str, Any]) -> tuple[float, str]:
        """
        Simple regulated score from observed work quality.
        Higher accept + saner |dH| + live health → more credit.
        """
        hmc = stats.get("hmc") or {}
        accept = float(hmc.get("acceptance_rate") or 0.0)
        abs_dh = float(hmc.get("recent_abs_dh") or 0.0)
        health = str(stats.get("health") or "unknown")

        if not stats.get("live", False):
            return 0.0, "not_live"
        if health in ("stopped", "no_export"):
            return 0.0, f"health={health}"
        if accept < MIN_ACCEPT:
            return 0.0, f"accept={accept:.2f}<{MIN_ACCEPT}"
        if abs_dh > MAX_ABS_DH:
            return 0.0, f"|dH|={abs_dh:.2f}>{MAX_ABS_DH}"

        # Quality multiplier: prefer accept near 0.6 and low |dH|
        quality = min(1.0, accept / 0.65) * max(0.2, 1.0 - (abs_dh / max(MAX_ABS_DH, 1e-6)))
        credit = BASE_CREDIT * quality
        reason = f"accept={accept:.2f} |dH|={abs_dh:.2f} health={health} q={quality:.2f}"
        return round(credit, 4), reason

    def try_mint_from_stats(
        self,
        token: Optional[str] = None,
        force_token_from_env: bool = True,
    ) -> Optional[WorkClaim]:
        """
        Attempt one mint from current stats.json.

        Raises PermissionError if control token missing/invalid.
        Returns None if regulation says "not yet" (cooldown / traj / quality).
        Returns sealed WorkClaim if minted.
        """
        provided = token
        if provided is None and force_token_from_env:
            provided = get_control_token()
        require_control_token(provided)

        stats = read_local_stats(self.stats_path)
        if not stats:
            raise CreditMintError(f"no stats at {self.stats_path} (run with --export-stats)")

        state = self._load_state()
        traj = int(stats.get("traj") or 0)
        now = time.time()

        if traj - int(state.get("last_traj", -1)) < MIN_TRAJ_DELTA:
            return None  # not enough new work
        if now - float(state.get("last_mint_ts", 0.0)) < COOLDOWN_SEC:
            return None  # cooldown

        credit, reason = self._score_credit(stats)
        if credit <= 0.0:
            return None

        hmc = stats.get("hmc") or {}
        mem = stats.get("memory") or {}
        att = stats.get("attractors") or {}

        claim = WorkClaim(
            node_id=str(stats.get("version", "metafield") or "metafield"),
            traj=traj,
            acceptance_rate=float(hmc.get("acceptance_rate") or 0.0),
            recent_abs_dh=float(hmc.get("recent_abs_dh") or 0.0),
            health=str(stats.get("health") or "unknown"),
            live=bool(stats.get("live", False)),
            memory_size=int(mem.get("size") or 0),
            num_attractors=int(att.get("num_attractors") or 0),
            credit=credit,
            reason=reason,
            timestamp=now,
            extras={
                "schema_version_stats": stats.get("schema_version"),
                "mint_regulation": {
                    "min_traj_delta": MIN_TRAJ_DELTA,
                    "min_accept": MIN_ACCEPT,
                    "max_abs_dh": MAX_ABS_DH,
                    "cooldown_sec": COOLDOWN_SEC,
                    "base_credit": BASE_CREDIT,
                },
            },
        ).seal()

        self._append_claim(claim)
        state["last_traj"] = traj
        state["last_mint_ts"] = now
        state["total_credit"] = float(state.get("total_credit", 0.0)) + credit
        state["last_claim_id"] = claim.claim_id
        self._save_state(state)
        return claim

    def total_credit(self) -> float:
        return float(self._load_state().get("total_credit", 0.0))


def main() -> None:
    p = argparse.ArgumentParser(description="MetaField token-gated swarm credit mint")
    p.add_argument("--watch", action="store_true", help="poll stats and mint when eligible")
    p.add_argument("--interval", type=float, default=15.0, help="watch poll seconds")
    p.add_argument("--token", type=str, default=None, help="override control token (else env)")
    args = p.parse_args()

    if not control_enabled() and args.token is None:
        print(
            "[mint] control surface disabled. "
            "Set METAFIELD_CONTROL_TOKEN to enable minting."
        )
        raise SystemExit(2)

    mint = CreditMint()
    print(f"[mint] claims → {CLAIMS_PATH}")
    print(f"[mint] stats  ← {STATS_PATH}")
    print(f"[mint] regulation: Δtraj≥{MIN_TRAJ_DELTA} accept≥{MIN_ACCEPT} "
          f"|dH|≤{MAX_ABS_DH} cooldown={COOLDOWN_SEC}s base={BASE_CREDIT}")

    def once() -> None:
        try:
            claim = mint.try_mint_from_stats(token=args.token)
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
                f"traj={claim.traj} ({claim.reason}) "
                f"total={mint.total_credit():.4f}"
            )

    if not args.watch:
        once()
        return

    print("[mint] watch mode — Ctrl+C to stop")
    try:
        while True:
            once()
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n[mint] stopped")


if __name__ == "__main__":
    main()
