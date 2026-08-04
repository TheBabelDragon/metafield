#!/usr/bin/env python3
"""meta_field_distributed.py v1.60 — in-tree compressed body (no network)."""
from __future__ import annotations
import base64, gzip, pathlib, runpy, sys
from _v160_blob_a import BLOB_A
from _v160_blob_b import BLOB_B

_CACHE = pathlib.Path(__file__).resolve().parent / ".meta_field_distributed.v160.full.py"

def _ensure_impl() -> pathlib.Path:
    if _CACHE.exists() and b'VERSION = "1.60"' in _CACHE.read_bytes():
        return _CACHE
    data = gzip.decompress(base64.b64decode("".join(BLOB_A + BLOB_B)))
    _CACHE.write_bytes(data)
    return _CACHE

if __name__ == "__main__":
    impl = _ensure_impl()
    sys.argv[0] = str(impl)
    runpy.run_path(str(impl), run_name="__main__")
