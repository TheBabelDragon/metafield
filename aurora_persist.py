#!/usr/bin/env python3
"""
aurora_persist.py

Phase 14 — content-addressed persistence for physical-field objects.

Stores:
  field observations / memory entries
  channel / optical calibration profiles
  transfer matrices / geometry snapshots
  encoder checkpoints (state dict paths)
  probe histories

Layout (local by default):

  <root>/
    objects/<sha256[:2]>/<sha256>.json
    refs/<name>.json
    index.jsonl

Optional Aurora publish requires AURORA_PERSIST_ENABLED=1 and Redis.
Without Redis the store is fully offline and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.astype(np.float64).reshape(-1).tolist()
            if np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.integer)
            else obj.tolist(),
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _to_jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return str(obj)


def content_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class AuroraObjectStore:
    """Content-addressed object store with optional Redis mirror."""

    def __init__(
        self,
        root: Union[str, Path] = "/tmp/metafield/aurora_objects",
        *,
        redis_url: Optional[str] = None,
        publish: Optional[bool] = None,
    ) -> None:
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.refs_dir = self.root / "refs"
        self.index_path = self.root / "index.jsonl"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)

        if publish is None:
            publish = os.environ.get("AURORA_PERSIST_ENABLED", "").strip() in (
                "1",
                "true",
                "yes",
            )
        self.publish = bool(publish)
        self.redis_url = redis_url or os.environ.get(
            "REDIS_URL", "redis://127.0.0.1:6379/0"
        )
        self._r = None
        if self.publish:
            self._connect_redis()

    def _connect_redis(self) -> None:
        try:
            import redis

            self._r = redis.from_url(
                self.redis_url, decode_responses=True, socket_connect_timeout=1.5
            )
            self._r.ping()
        except Exception:
            self._r = None

    def put(
        self,
        kind: str,
        data: Any,
        *,
        ref: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        body = {
            "kind": str(kind),
            "data": _to_jsonable(data),
            "meta": _to_jsonable(meta or {}),
            "ts": time.time(),
        }
        sha = content_hash(body)
        path = self.objects_dir / sha[:2] / f"{sha}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(body, indent=2), encoding="utf-8")

        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"sha": sha, "kind": kind, "ref": ref, "ts": body["ts"]})
                + "\n"
            )

        if ref:
            self.set_ref(ref, sha, kind=kind)

        if self._r is not None:
            try:
                key = f"aurora:metafield:object:{sha}"
                self._r.set(key, json.dumps(body))
                if ref:
                    self._r.set(f"aurora:metafield:ref:{ref}", sha)
            except Exception:
                pass

        return sha

    def get(self, sha: str) -> Optional[Dict[str, Any]]:
        path = self.objects_dir / sha[:2] / f"{sha}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set_ref(self, name: str, sha: str, *, kind: str = "") -> None:
        safe = name.replace("/", "_").replace("..", "_")
        payload = {"sha": sha, "kind": kind, "name": name, "ts": time.time()}
        (self.refs_dir / f"{safe}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def resolve(self, name: str) -> Optional[Dict[str, Any]]:
        safe = name.replace("/", "_").replace("..", "_")
        ref_path = self.refs_dir / f"{safe}.json"
        if not ref_path.exists():
            return None
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        return self.get(ref["sha"])

    def list_index(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            return []
        lines = self.index_path.read_text(encoding="utf-8").strip().splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def put_memory_entry(self, entry: Any, *, ref: Optional[str] = None) -> str:
        if hasattr(entry, "to_dict"):
            data = entry.to_dict()
        elif is_dataclass(entry):
            data = asdict(entry)
        else:
            data = entry
        return self.put("memory_entry", data, ref=ref)

    def put_transfer_matrix(
        self,
        matrix: np.ndarray,
        *,
        body_id: str = "optical",
        ref: Optional[str] = None,
    ) -> str:
        return self.put(
            "transfer_matrix",
            {"matrix": matrix, "body_id": body_id},
            ref=ref or f"{body_id}/transfer_matrix",
            meta={"body_id": body_id, "shape": list(np.asarray(matrix).shape)},
        )

    def put_probe_history(
        self, history: List[Dict[str, Any]], *, ref: Optional[str] = None
    ) -> str:
        return self.put("probe_history", history, ref=ref)

    def put_encoder_stats(self, stats: Any, *, ref: Optional[str] = None) -> str:
        return self.put("encoder_stats", stats, ref=ref)
