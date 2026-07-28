#!/usr/bin/env python3
"""
optical_serial_consumer.py

Host-side bridge: read FieldObservation JSON lines from the ESP32-S3
(optical-body-s3) over serial (or from a JSONL file) and turn them into
FieldMemoryEntry objects for MetaField episodic memory.

Phase 0 usage:

  # from a live board
  python optical_serial_consumer.py --port /dev/ttyUSB0

  # from a recorded log (Python stub or SD dump)
  python optical_serial_consumer.py --file /tmp/metafield/optical_phase0.jsonl

Does not require MetaField to be running continuously; it just materializes
the episodic entries so they can be inspected, saved, or fed into memory.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional

from schemas.field_memory import FieldMemoryEntry
from schemas.field_observation import FieldObservation, validate_observation


def iter_json_lines_from_file(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
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
                # non-JSON status lines from the firmware
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
    parser = argparse.ArgumentParser(description="Optical body → FieldMemoryEntry consumer")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="Serial port of the ESP32-S3 body")
    src.add_argument("--file", type=Path, help="JSONL file of FieldObservation packets")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--max", type=int, default=0, help="Stop after N entries (0 = forever)")
    parser.add_argument("--save", type=Path, help="Append FieldMemoryEntry JSONL to this path")
    parser.add_argument("--attractor", default=None, help="Optional attractor_id to attach")
    args = parser.parse_args()

    if args.file:
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
