"""Backend factory — pick the right SAM version from a checkpoint filename."""

from __future__ import annotations

import os

from .base import SamBackend, SamBackendError
from .sam1 import Sam1Backend, guess_sam1_model_type
from .sam2 import Sam2Backend, looks_like_sam2

BACKEND_KIND_SAM1 = "sam1"
BACKEND_KIND_SAM2 = "sam2"
BACKEND_KIND_SAM3 = "sam3"  # reserved


def detect_backend_kind(checkpoint_path: str) -> str:
    """Return 'sam1' / 'sam2' / 'sam3' based on filename heuristics.

    Defaults to SAM 1 when nothing else matches — preserves original behavior
    for users with existing sam_vit_*.pth files.
    """
    if not checkpoint_path:
        return BACKEND_KIND_SAM1
    name = os.path.basename(checkpoint_path).lower()
    if looks_like_sam2(name):
        return BACKEND_KIND_SAM2
    if name.startswith("sam3") or "sam_3" in name:
        return BACKEND_KIND_SAM3
    return BACKEND_KIND_SAM1


def create_backend(checkpoint_path: str, *, model_type_hint: str = "") -> SamBackend:
    """Instantiate the appropriate backend for ``checkpoint_path``.

    Args:
        checkpoint_path: path to the .pth/.pt weights file.
        model_type_hint: optional override for SAM 1 (vit_h/vit_l/vit_b).

    Raises:
        SamBackendError if the model family is unsupported.
    """
    kind = detect_backend_kind(checkpoint_path)
    if kind == BACKEND_KIND_SAM2:
        return Sam2Backend()
    if kind == BACKEND_KIND_SAM3:
        raise SamBackendError(
            "SAM 3 暂未支持。请使用 SAM 1 (sam_vit_*.pth) 或 SAM 2 (sam2_*.pt) 权重。"
        )
    # default: SAM 1
    mt = model_type_hint or guess_sam1_model_type(checkpoint_path)
    return Sam1Backend(model_type=mt)
