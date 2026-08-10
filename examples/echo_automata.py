#!/usr/bin/env python3
"""
echo_automata.py — Multi-level field automata + memory link

Level 0 atoms → Level 1 ANDs → Level 2 compositions

On each fired gate set:
  • write control event → echo_events.jsonl
  • write FieldMemoryEntry with attractor_id → gate memory + FieldMemoryStore

Attractor IDs (priority order):
  gate:FULL_LOCK > gate:ACTIVE_FIELD > gate:CONFIRMED_TRACK >
  gate:STEALTH_BREAK > gate:<L1 name>

Usage:

  python examples/echo_automata.py --follow --threshold 0.30
  python examples/echo_automata.py --no-follow --verify
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from field_memory_store import FieldMemoryStore
from schemas.field_memory import FieldMemoryEntry

# attractor priority (first match wins)
ATTRACTOR_PRIORITY = [
    "FULL_LOCK",
    "ACTIVE_FIELD",
    "CONFIRMED_TRACK",
    "STEALTH_BREAK",
    "SURPRISE_CONFIRMED",
    "TRACKED_SURPRISE",
    "HIGH_MOTION_SURPRISE",
    "QUIET_ANOMALY",
    "SURPRISE_RAW",
]


def _normalize_argv(argv: List[str]) -> List[str]:
    out = []
    for a in argv:
        for ch in ("\u2014", "\u2013", "\u2212"):
            a = a.replace(ch, "-")
        out.append(a)
    return out


def _load_predictor():
    path = ROOT / "examples" / "echo_field_predictor.py"
    spec = importlib.util.spec_from_file_location("echo_field_predictor", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_model(path: Path, pred_mod):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = pred_mod.TinyFieldHead(in_dim=int(ckpt["in_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    window = int(ckpt.get("window", 8))
    print(f"[automata] head {path}  window={window}")
    return model, window


def follow_jsonl(path: Path, poll_s: float = 0.2):
    print(f"[automata] waiting for {path} …")
    while not path.exists():
        time.sleep(poll_s)
    print(f"[automata] following {path}")
    with path.open(encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if not line:
                time.sleep(poll_s)
                try:
                    if path.stat().st_size < fh.tell():
                        fh.seek(0)
                except OSError:
                    pass
                continue
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                continue


def region_map(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for r in obj.get("field_regions") or []:
        out[str(r.get("region", ""))] = r
    return out


def eval_atoms(
    abs_r: float,
    threshold: float,
    motion: float,
    fuse_agreed: bool,
    n_tracks: int,
) -> Dict[str, bool]:
    return {
        "residual_high": abs_r >= threshold,
        "fuse_agreed": bool(fuse_agreed),
        "has_tracks": n_tracks >= 1,
        "high_motion": motion >= 0.6,
        "quiet": motion < 0.25,
    }


def eval_l1(atoms: Dict[str, bool]) -> List[str]:
    if not atoms["residual_high"]:
        return []
    gates = []
    if atoms["fuse_agreed"]:
        gates.append("SURPRISE_CONFIRMED")
    if atoms["has_tracks"]:
        gates.append("TRACKED_SURPRISE")
    if atoms["high_motion"]:
        gates.append("HIGH_MOTION_SURPRISE")
    if atoms["quiet"]:
        gates.append("QUIET_ANOMALY")
    if not gates:
        gates.append("SURPRISE_RAW")
    return gates


def eval_l2(l1: Set[str], atoms: Dict[str, bool]) -> List[str]:
    gates = []
    if "SURPRISE_CONFIRMED" in l1 and "TRACKED_SURPRISE" in l1:
        gates.append("CONFIRMED_TRACK")
    if "HIGH_MOTION_SURPRISE" in l1 and (
        "SURPRISE_CONFIRMED" in l1 or "TRACKED_SURPRISE" in l1
    ):
        gates.append("ACTIVE_FIELD")
    if "QUIET_ANOMALY" in l1 and atoms["has_tracks"]:
        gates.append("STEALTH_BREAK")
    if "CONFIRMED_TRACK" in gates and "HIGH_MOTION_SURPRISE" in l1:
        gates.append("FULL_LOCK")
    if "FULL_LOCK" in gates:
        for g in ("CONFIRMED_TRACK", "ACTIVE_FIELD"):
            if g not in gates:
                gates.append(g)
    return gates


def primary_attractor(gates: List[str], level: int) -> str:
    gate_set = set(gates)
    for name in ATTRACTOR_PRIORITY:
        if name in gate_set:
            return f"gate:{name}"
    return f"gate:L{level}"


def memory_entry_from_gate(
    obj: Dict[str, Any],
    *,
    attractor_id: str,
    level: int,
    gates: List[str],
    l1: List[str],
    l2: List[str],
    atoms: Dict[str, bool],
    abs_r: float,
    pred_m: float,
    actual_m: float,
    fuse_agreed: bool,
    n_tracks: int,
) -> FieldMemoryEntry:
    entry = FieldMemoryEntry.from_observation(obj, attractor_id=attractor_id)
    # residual is the anomaly signal for gate memory
    entry.anomaly = min(1.0, float(abs_r))
    # confidence: prefer fuse conf from regions if present, else high when agreed
    if fuse_agreed:
        entry.confidence = max(entry.confidence, 0.75)
    entry.extras = dict(entry.extras or {})
    entry.extras.update({
        "source": "field_automata",
        "level": level,
        "gates": gates,
        "l1": l1,
        "l2": l2,
        "atoms": atoms,
        "pred_motion": pred_m,
        "actual_motion": actual_m,
        "abs_residual_motion": abs_r,
        "fuse_agreed": fuse_agreed,
        "n_tracks": n_tracks,
    })
    return entry


def verify_store(store: FieldMemoryStore, mem_path: Path) -> bool:
    """Smoke checks for memory link."""
    ok = True
    stats = store.get_stats()
    print("\n=== verify FieldMemoryStore (gate link) ===")
    print(f"  size={stats['size']}  total_added={stats['total_added']}")
    print(f"  avg_anomaly={stats['avg_anomaly']:.3f}  avg_priority={stats['avg_priority']:.3f}")

    if stats["size"] == 0:
        print("  FAIL: store empty")
        return False

    with_attr = 0
    by_attr: Dict[str, int] = {}
    for item in store.buffer:
        e = item["entry"]
        if e.attractor_id:
            with_attr += 1
            by_attr[e.attractor_id] = by_attr.get(e.attractor_id, 0) + 1

    print(f"  with attractor_id={with_attr}/{stats['size']}")
    for k, v in sorted(by_attr.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {k}: {v}")

    if with_attr != stats["size"]:
        print("  FAIL: some entries missing attractor_id")
        ok = False
    else:
        print("  OK: all entries have attractor_id")

    if mem_path.exists() and mem_path.stat().st_size > 0:
        n_lines = sum(1 for line in mem_path.open() if line.strip().startswith("{"))
        print(f"  jsonl lines={n_lines}  path={mem_path}")
        if n_lines == 0:
            print("  FAIL: memory jsonl empty")
            ok = False
        else:
            print("  OK: memory jsonl non-empty")
    else:
        print(f"  FAIL: missing memory jsonl {mem_path}")
        ok = False

    sampled = store.sample(n=min(5, stats["size"]))
    print("  sample:")
    for e in sampled:
        print(
            f"    attractor={e.attractor_id}  anom={e.anomaly:.3f}  "
            f"conf={e.confidence:.2f}  gates={(e.extras or {}).get('gates', [])[:3]}"
        )

    print("=== verify", "PASS" if ok else "FAIL", "===")
    return ok


def main() -> None:
    sys.argv = [sys.argv[0]] + _normalize_argv(sys.argv[1:])

    p = argparse.ArgumentParser(description="Multi-level Echo field automata + memory link")
    p.add_argument("--file", type=Path, default=Path("/tmp/metafield/echo.jsonl"))
    p.add_argument("--model", type=Path, default=Path("/tmp/metafield/echo_head.pt"))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--events", type=Path, default=Path("/tmp/metafield/echo_events.jsonl"))
    p.add_argument(
        "--memory",
        type=Path,
        default=Path("/tmp/metafield/echo_gate_memory.jsonl"),
        help="Append FieldMemoryEntry JSONL (gate attractors)",
    )
    p.add_argument(
        "--save-store",
        type=Path,
        default=Path("/tmp/metafield/echo_gate_store.jsonl"),
        help="FieldMemoryStore checkpoint",
    )
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--capacity", type=int, default=4096)
    p.add_argument("--min-level", type=int, default=1, choices=(1, 2))
    p.add_argument("--follow", action="store_true", default=True)
    p.add_argument("--no-follow", action="store_true")
    p.add_argument(
        "--verify",
        action="store_true",
        help="After run, assert store entries have attractor_id",
    )
    p.add_argument(
        "--memory-min-level",
        type=int,
        default=1,
        choices=(1, 2),
        help="Only link memory for gates at this level or above (default 1)",
    )
    args = p.parse_args()
    follow = not args.no_follow

    if not args.model.exists():
        print(f"[automata] missing model: {args.model}", file=sys.stderr)
        sys.exit(1)

    pred_mod = _load_predictor()
    model, window = load_model(args.model, pred_mod)
    history: Deque[List[float]] = deque(maxlen=window)

    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.memory.parent.mkdir(parents=True, exist_ok=True)
    if not args.events.exists():
        args.events.touch()

    store = FieldMemoryStore(soft_capacity=args.capacity)
    if args.save_store.exists():
        n = store.load_jsonl(args.save_store)
        if n:
            print(f"[automata] warm-loaded store {n} from {args.save_store}")

    ev_fh = args.events.open("a", encoding="utf-8")
    mem_fh = args.memory.open("a", encoding="utf-8")

    frames = 0
    events = 0
    mem_writes = 0

    print(
        f"[automata] threshold={args.threshold:.3f}  min_level={args.min_level}  "
        f"memory_min_level={args.memory_min_level}"
    )
    print(f"[automata] events→{args.events}")
    print(f"[automata] memory→{args.memory}  store→{args.save_store}")
    print("[automata] L1: SURPRISE_CONFIRMED | TRACKED_SURPRISE | HIGH_MOTION_SURPRISE | QUIET_ANOMALY")
    print("[automata] L2: CONFIRMED_TRACK | ACTIVE_FIELD | STEALTH_BREAK | FULL_LOCK")

    def handle(obj: Dict[str, Any]) -> None:
        nonlocal frames, events, mem_writes
        feats = pred_mod.features_from_observation(obj)
        if feats is None:
            return

        if len(history) < window:
            history.append(feats)
            frames += 1
            return

        x = torch.tensor([[v for row in history for v in row]], dtype=torch.float32)
        with torch.no_grad():
            pred = model(x)[0]
        pred_m = float(pred[0])
        actual_m = float(feats[0])
        abs_r = abs(actual_m - pred_m)
        history.append(feats)
        frames += 1

        by = region_map(obj)
        fuse = by.get("fuse") or {}
        fuse_agreed = bool((fuse.get("extras") or {}).get("agreed", False))
        n_tracks = sum(1 for k in by if k.startswith("track_"))
        motion = actual_m

        atoms = eval_atoms(abs_r, args.threshold, motion, fuse_agreed, n_tracks)
        l1 = eval_l1(atoms)
        l2 = eval_l2(set(l1), atoms)

        if args.min_level >= 2 and not l2:
            if frames % 50 == 0:
                print(f"[automata] frames={frames} events={events} mem={mem_writes} |r|={abs_r:.3f} (no L2)")
            return
        if args.min_level < 2 and not l1:
            if frames % 50 == 0:
                print(f"[automata] frames={frames} events={events} mem={mem_writes} |r|={abs_r:.3f}")
            return

        level = 2 if l2 else 1
        gates = list(dict.fromkeys(l2 + l1))
        attractor_id = primary_attractor(gates, level)

        event = {
            "schema_version": 2,
            "kind": "field_automata_event",
            "level": level,
            "gates": gates,
            "l1": l1,
            "l2": l2,
            "atoms": atoms,
            "attractor_id": attractor_id,
            "body_id": obj.get("body_id", "echo-grid-01"),
            "abs_residual_motion": abs_r,
            "pred_motion": pred_m,
            "actual_motion": actual_m,
            "motion": motion,
            "fuse_agreed": fuse_agreed,
            "n_tracks": n_tracks,
            "threshold": args.threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "excitation_id": obj.get("excitation_id"),
        }
        ev_fh.write(json.dumps(event) + "\n")
        ev_fh.flush()
        events += 1

        # --- memory link ---
        if level >= args.memory_min_level:
            entry = memory_entry_from_gate(
                obj,
                attractor_id=attractor_id,
                level=level,
                gates=gates,
                l1=l1,
                l2=l2,
                atoms=atoms,
                abs_r=abs_r,
                pred_m=pred_m,
                actual_m=actual_m,
                fuse_agreed=fuse_agreed,
                n_tracks=n_tracks,
            )
            store.add(entry)
            mem_fh.write(entry.to_json() + "\n")
            mem_fh.flush()
            mem_writes += 1

            if mem_writes % max(1, args.checkpoint_every) == 0:
                n = store.save_jsonl(args.save_store)
                print(f"[automata] checkpoint store ({n}) → {args.save_store}")

        shown = l2 if l2 else l1
        print(
            f"[GATE L{level}] #{events}  {','.join(shown)}  "
            f"attr={attractor_id}  |r|={abs_r:.3f}  motion={motion:.2f}  "
            f"fuse={'Y' if fuse_agreed else 'n'}  tracks={n_tracks}"
        )

    try:
        if not follow:
            if not args.file.exists():
                print(f"[automata] missing {args.file}", file=sys.stderr)
                sys.exit(1)
            with args.file.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            handle(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        else:
            for obj in follow_jsonl(args.file):
                handle(obj)
    except KeyboardInterrupt:
        print("\n[automata] stopped")
    finally:
        ev_fh.close()
        mem_fh.close()
        n = store.save_jsonl(args.save_store)
        print(f"[automata] frames={frames}  events={events}  mem={mem_writes}")
        print(f"[automata] final store ({n}) → {args.save_store}")
        print(f"[automata] memory jsonl → {args.memory}")

        if args.verify:
            passed = verify_store(store, args.memory)
            sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
