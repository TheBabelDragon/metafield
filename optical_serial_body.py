#!/usr/bin/env python3
"""
optical_serial_body.py

Hardware optical body with the same surface as OpticalBodySimulator:

  body.excite(laser_id) → detector vector / FieldObservation
  body.estimate_transfer_matrix()

Usage tomorrow:

  PYTHONPATH=../wavebridge:. python examples/optical_hardware_probe.py \\
      --port /dev/ttyUSB0

  PYTHONPATH=../wavebridge:. python examples/optical_hardware_probe.py --dry-run
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class HardwareExcitationRecord:
    laser_id: int
    drive_level: float
    detector_response: np.ndarray
    ambient: float = 0.0
    geometry_state: str = "live"
    observation: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""
    raw_line: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "laser_id": self.laser_id,
            "drive_level": self.drive_level,
            "detector_response": self.detector_response.tolist(),
            "ambient": self.ambient,
            "geometry_state": self.geometry_state,
            "ok": self.ok,
            "error": self.error,
        }


class OpticalSerialBody:
    """Multi-laser optical body over serial (ESP32-S3 optical-body firmware)."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        *,
        baud: int = 115200,
        n_lasers: int = 12,
        n_detectors: int = 20,
        dry_run: bool = False,
        timeout_s: float = 3.0,
    ) -> None:
        self.port = port
        self.baud = baud
        self.n_lasers = int(n_lasers)
        self.n_detectors = int(n_detectors)
        self.dry_run = bool(dry_run)
        self.timeout_s = float(timeout_s)
        self._backend = None
        self._open = False

    def open(self) -> None:
        if self._open:
            return
        try:
            from wavebridge.hardware.serial_backend import (
                ESP32SerialBackend,
                SerialConfig,
            )

            self._backend = ESP32SerialBackend(
                SerialConfig(
                    port=self.port,
                    baud=self.baud,
                    dry_run=self.dry_run,
                    n_detectors=self.n_detectors,
                    read_timeout_s=self.timeout_s,
                )
            )
            self._backend.open()
        except ImportError:
            self._backend = _PyserialFallback(
                self.port, self.baud, self.dry_run, self.n_detectors, self.timeout_s
            )
            self._backend.open()
        self._open = True

    def close(self) -> None:
        if self._backend is not None and hasattr(self._backend, "close"):
            self._backend.close()
        self._open = False

    def __enter__(self) -> "OpticalSerialBody":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def ping(self) -> bool:
        self.open()
        return bool(self._backend.ping())

    def excite(
        self,
        laser_id: int,
        drive_level: float = 1.0,
        *,
        geometry_state: str = "live",
    ) -> HardwareExcitationRecord:
        if not (0 <= laser_id < self.n_lasers):
            raise ValueError(f"laser_id {laser_id} out of range [0, {self.n_lasers})")
        self.open()
        res = self._backend.excite(int(laser_id), drive_level=float(drive_level))
        resp = np.asarray(
            getattr(res, "detector_response", res), dtype=np.float32
        ).reshape(-1)
        if resp.size != self.n_detectors and resp.size > 0:
            self.n_detectors = int(resp.size)
        return HardwareExcitationRecord(
            laser_id=int(laser_id),
            drive_level=float(drive_level),
            detector_response=resp,
            geometry_state=geometry_state,
            observation=getattr(res, "observation", {}) or {},
            ok=bool(getattr(res, "ok", True)),
            error=str(getattr(res, "error", "") or ""),
            raw_line=str(getattr(res, "raw_line", "") or ""),
        )

    def estimate_transfer_matrix(self, drive_level: float = 1.0) -> np.ndarray:
        cols = []
        for L in range(self.n_lasers):
            rec = self.excite(L, drive_level=drive_level)
            cols.append(rec.detector_response)
        n = max(c.size for c in cols)
        mats = []
        for c in cols:
            if c.size < n:
                c = np.pad(c, (0, n - c.size))
            mats.append(c[:n])
        return np.column_stack(mats).astype(np.float32)


class _PyserialFallback:
    def __init__(self, port, baud, dry_run, n_detectors, timeout_s):
        self.port = port
        self.baud = baud
        self.dry_run = dry_run
        self.n_detectors = n_detectors
        self.timeout_s = timeout_s
        self._ser = None

    def open(self):
        if self.dry_run:
            return
        import serial

        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)

    def close(self):
        if self._ser:
            self._ser.close()

    def ping(self):
        if self.dry_run:
            return True
        self._ser.write(b"PING\n")
        line = self._ser.readline().decode("utf-8", errors="replace")
        return "ok" in line.lower() or "pong" in line.lower()

    def excite(self, laser_id, drive_level=1.0):
        from types import SimpleNamespace
        import json

        if self.dry_run:
            idx = np.arange(self.n_detectors, dtype=np.float32)
            base = np.exp(-0.25 * (idx - (laser_id % self.n_detectors)) ** 2)
            resp = np.clip(base * drive_level + 0.02, 0, 1).astype(np.float32)
            return SimpleNamespace(
                detector_response=resp, ok=True, error="", observation={}, raw_line=""
            )
        self._ser.write(f"EXCITE {laser_id}\n".encode())
        line = self._ser.readline().decode("utf-8", errors="replace").strip()
        try:
            obs = json.loads(line[line.find("{") :])
            regions = obs.get("field_regions") or []
            vals = [float(r.get("observed") or 0) for r in regions]
            resp = np.asarray(vals, dtype=np.float32)
        except Exception as exc:
            return SimpleNamespace(
                detector_response=np.zeros(self.n_detectors, dtype=np.float32),
                ok=False,
                error=str(exc),
                observation={},
                raw_line=line,
            )
        return SimpleNamespace(
            detector_response=resp, ok=True, error="", observation=obs, raw_line=line
        )
