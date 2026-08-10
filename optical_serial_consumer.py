#!/usr/bin/env python3
"""
optical_serial_consumer.py

Host-side bridge: read FieldObservation JSON lines from the ESP32-S3
(optical-body-s3) over serial (or from a JSONL file) and turn them into
FieldMemoryEntry objects for MetaField episodic memory.

Works for any body that emits the FieldObservation contract — optical,
Echo Grid ultrasonic, ZVS, etc.

Usage:

  # from a live board
  python optical_serial_consumer.py --port /dev/ttyUSB0

  # from a recorded log
  python optical_serial_consumer.py --file /tmp/metafield/echo.jsonl

  # live follow (wait for file, then tail new lines)
  python optical_serial_consumer.py --file /tmp/metafield/echo.jsonl --follow \
    --save /tmp/metafield/field_memory.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

from schemas.field_memory import FieldMemoryEntry
from schemas.field_observation import FieldObservation, validate_observation


def iter_json_lines_from_file(path: Path) -> Iterator[dict]:
    """One-shot read of an existing JSONL file."""
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[consumer] file line {line_no}: {e}", file=sys.stderr)


def iter_json_lines_follow(path: Path, poll_s: float = 0.25) -> Iterator[dict]:
    """
    Wait for path to appear, then tail new lines forever (like tail -f).
    Survives the producer starting after the consumer.
    """
    print(f"[consumer] waiting for {path} …")
    while not path.exists():
        time.sleep(poll_s)
    print(f"[consumer] following {path}")

    # Start at beginning so we also pick up whatever is already there
    with path.open(encoding="utf-8") as fh:
        line_no = 0
        while True:
            line = fh.readline()
            if not line:
                time.sleep(poll_s)
                # handle truncation / rewrite
                try:
                    if path.stat().st_size < fh.tell():
                        fh.seek(0)
                        line_no = 0
                except OSError:
                    time.sleep(poll_s)
                continue
            line_no += 1
            text = line.strip()
            if not text or not text.startswith("{"):
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as e:
                print(f"[consumer] file line {line_no}: {e}", file=sys.stderr)


def iter_json_lines_from_serial(port: str, baud: int = 115200) -> Iterator[dict]:
    try:
        import serial  # type: ignore
    except ImportError:
        print("[consumer] pyserial not installed.  pip install pyserial", file=sys.stderr)
        sys.exit(1)

    ser = serial.Serial(port, baud, timeout=1.0)
    print(f"[consumer] listening on {port} @ {baud}")
    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            try:
                text = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if not text.startswith("{"):
                print(f"[body] {text}")
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError:
                print(f"[consumer] bad JSON: {text[:80]}…", file=sys.stderr)
    finally:
        ser.close()


def process_packet(data: dict, attractor_id: Optional[str] = None) -> Optional[FieldMemoryEntry]:
    """Validate observation and promote to FieldMemoryEntry."""
    try:
        obs = FieldObservation.from_dict(data)
    except Exception as e:
        print(f"[consumer] could not parse observation: {e}", file=sys.stderr)
        return None

    problems = validate_observation(obs)
    if problems:
        print(f"[consumer] validation: {problems}", file=sys.stderr)
        # still accept partial packets in Phase 0

    entry = FieldMemoryEntry.from_observation(obs.to_dict(), attractor_id=attractor_id)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FieldObservation → FieldMemoryEntry consumer (optical / echo / any body)"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="Serial port of an ESP32 body")
    src.add_argument("--file", type=Path, help="JSONL file of FieldObservation packets")
    parser.add_argument(
        "--follow", action="store_true",
        help="Wait for --file to appear and tail new lines (live Echo / optical log)",
    )
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--max", type=int, default=0, help="Stop after N entries (0 = forever)")
    parser.add_argument("--save", type=Path, help="Append FieldMemoryEntry JSONL to this path")
    parser.add_argument("--attractor", default=None, help="Optional attractor_id to attach")
    args = parser.parse_args()

    if args.file:
        if args.follow:
            stream = iter_json_lines_follow(args.file)
        else:
            if not args.file.exists():
                print(
                    f"[consumer] file not found: {args.file}\n"
                    f"  Start the producer first, or pass --follow to wait/tail.\n"
                    f"  Echo example:\n"
                    f"    python visualization/dashboard.py --csi "
                    f"--metafield-log {args.file}",
                    file=sys.stderr,
                )
                sys.exit(1)
            stream = iter_json_lines_from_file(args.file)
    else:
        stream = iter_json_lines_from_serial(args.port, args.baud)

    out_fh = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.save.open("a", encoding="utf-8")

    count = 0
    try:
        for data in stream:
            entry = process_packet(data, attractor_id=args.attractor)
            if entry is None:
                continue
            count += 1
            summary = (
                f"#{count:04d}  body={entry.body_id}  "
                f"exc={entry.excitation_id}  "
                f"conf={entry.confidence:.2f}  "
                f"anom={entry.anomaly:.3f}  "
                f"obs_n={len(entry.observed_response or [])}"
            )
            print(summary)
            if out_fh:
                out_fh.write(entry.to_json() + "\n")
                out_fh.flush()
            if args.max and count >= args.max:
                break
    except KeyboardInterrupt:
        print("\n[consumer] stopped")
    finally:
        if out_fh:
            out_fh.close()
            print(f"[consumer] wrote {count} FieldMemoryEntry → {args.save}")


if __name__ == "__main__":
    main()
