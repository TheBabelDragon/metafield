#!/usr/bin/env python3
"""
field_memory_store.py

Prioritized episodic buffer for physical field bodies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from schemas.field_memory import FieldMemoryEntry
from schemas.scarcity_clock import parse_clock


class FieldMemoryStore:
    def __init__(self, soft_capacity: int = 1024, soft_capacity_max: int = 16384, absolute_safety_limit: int = 50000):
        self.buffer: List[Dict[str, Any]] = []
        self.soft_capacity = soft_capacity
        self.soft_capacity_max = soft_capacity_max
        self.absolute_safety_limit = absolute_safety_limit
        self.total_added = 0

    def _priority(self, entry: FieldMemoryEntry, replay_count: int = 0) -> float:
        anom = max(0.0, float(entry.anomaly))
        conf = max(0.0, min(1.0, float(entry.confidence)))
        return max(0.5, (1.0 + 2.0 * anom) * (1.5 - conf) * (1.0 + 0.1 * replay_count))

    def add(self, entry: FieldMemoryEntry) -> None:
        item = {"entry": entry, "priority": self._priority(entry), "replay": 0}
        self.buffer.append(item)
        self.total_added += 1
        if len(self.buffer) > self.soft_capacity:
            overflow = len(self.buffer) - self.soft_capacity
            self.buffer.sort(key=lambda x: x["priority"])
            self.buffer = self.buffer[overflow:]
        if len(self.buffer) > self.absolute_safety_limit:
            overflow = len(self.buffer) - self.absolute_safety_limit
            self.buffer.sort(key=lambda x: x["priority"])
            self.buffer = self.buffer[overflow:]
        if self.total_added % 200 == 0 and self.soft_capacity < self.soft_capacity_max:
            self.soft_capacity = min(self.soft_capacity_max, self.soft_capacity + 128)

    def sample(self, n: int = 16) -> List[FieldMemoryEntry]:
        if not self.buffer:
            return []
        if len(self.buffer) <= n:
            for item in self.buffer:
                item["replay"] += 1
            return [item["entry"] for item in self.buffer]
        import random
        weights = [item["priority"] for item in self.buffer]
        chosen = random.choices(self.buffer, weights=weights, k=n)
        for item in chosen:
            item["replay"] += 1
            item["priority"] = self._priority(item["entry"], item["replay"])
        return [item["entry"] for item in chosen]

    def get_stats(self) -> Dict[str, Any]:
        if not self.buffer:
            return {"size": 0, "soft_capacity": self.soft_capacity, "total_added": self.total_added, "avg_priority": 0.0, "avg_anomaly": 0.0, "avg_confidence": 0.0}
        pri = [x["priority"] for x in self.buffer]
        anom = [x["entry"].anomaly for x in self.buffer]
        conf = [x["entry"].confidence for x in self.buffer]
        return {
            "size": len(self.buffer),
            "soft_capacity": self.soft_capacity,
            "total_added": self.total_added,
            "avg_priority": sum(pri) / len(pri),
            "avg_anomaly": sum(anom) / len(anom),
            "avg_confidence": sum(conf) / len(conf),
        }

    def save_jsonl(self, path: Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for item in self.buffer:
                fh.write(item["entry"].to_json() + "\n")
        return len(self.buffer)

    def load_jsonl(self, path: Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        n = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = FieldMemoryEntry(
                    body_id=data["body_id"],
                    excitation_id=data.get("excitation_id"),
                    location=data.get("location"),
                    expected_response=data.get("expected_response"),
                    observed_response=data.get("observed_response"),
                    confidence=float(data.get("confidence", 0.0)),
                    anomaly=float(data.get("anomaly", 0.0)),
                    attractor_id=data.get("attractor_id"),
                    timestamp=data.get("timestamp", ""),
                    clock=parse_clock(data.get("clock")),
                    extras=data.get("extras") or {},
                )
                self.add(entry)
                n += 1
        return n

    def __len__(self) -> int:
        return len(self.buffer)
