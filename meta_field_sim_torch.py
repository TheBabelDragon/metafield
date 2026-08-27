from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Any, List

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "meta_field_sim_torch.py requires PyTorch. Install it with:\n"
        "    pip install torch\n"
        "(GPU build if you have CUDA: see https://pytorch.org/get-started/locally/)"
    ) from e
