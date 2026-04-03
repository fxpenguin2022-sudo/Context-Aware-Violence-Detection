from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def runtime_summary() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }
