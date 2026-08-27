from __future__ import annotations

"""Public API for the Wilson + HMC core.

Known-good body from commit 4588681, split into parts so the file can be
committed through the GitHub contents API. Parts execute in order into
this module; callers still import from meta_field_sim_torch.
"""
from pathlib import Path

_DIR = Path(__file__).resolve().parent
for _name in (
    "_sim_torch_part_a.py",
    "_sim_torch_part_b.py",
    "_sim_torch_part_c.py",
    "_sim_torch_part_d.py",
    "_sim_torch_part_e.py",
):
    _path = _DIR / _name
    exec(compile(_path.read_text(), str(_path), "exec"), globals())
