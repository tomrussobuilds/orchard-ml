"""
Shared Helpers for Detection Adapters.

Internal utilities used by multiple detection adapters.
"""

from __future__ import annotations

from typing import Any

import torch


def to_cpu(d: dict[str, Any]) -> dict[str, Any]:
    """
    Move all tensor values in a dict to CPU.

    Args:
        d: Dict with tensor values (e.g. detection predictions or targets).

    Returns:
        New dict with all tensors moved to CPU; non-tensor values unchanged.
    """
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
