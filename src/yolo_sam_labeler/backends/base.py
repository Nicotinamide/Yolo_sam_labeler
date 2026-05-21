"""Abstract base for SAM backends.

Concrete backends (sam1, sam2, ...) wrap their respective Python packages
behind this interface so the rest of the app stays version-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class SamBackendError(RuntimeError):
    """Backend-specific error (e.g. package missing, unsupported model)."""


class SamBackend(ABC):
    """Common interface for all SAM versions.

    Lifecycle:
        load(checkpoint, device) → set_image(rgb) → predict_*(prompt) → ...
    Embedding cache:
        snapshot()/restore() let the service keep a per-image cache so revisiting
        an already-encoded image is free.
    """

    name: str = "abstract"           # human-readable backend label
    supports_box: bool = True        # whether this backend supports box prompts

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, checkpoint: str, device: str) -> None:
        """Load weights into memory. Raises SamBackendError on failure."""

    @abstractmethod
    def set_image(self, rgb: np.ndarray) -> None:
        """Encode an HWC RGB uint8 image into the model's internal state."""

    @property
    @abstractmethod
    def is_image_set(self) -> bool: ...

    # ------------------------------------------------------------------
    # Embedding cache (per-key snapshots used by the service-level LRU)
    # ------------------------------------------------------------------

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return the current image-encoding state as a serializable dict."""

    @abstractmethod
    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a previously captured snapshot to skip re-encoding."""

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_point(self, x: int, y: int, multimask: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mask(s) from a single positive point.

        Returns (masks, scores). masks shape (N, H, W) uint8 binary, scores (N,).
        """

    @abstractmethod
    def predict_box(self, x1: int, y1: int, x2: int, y2: int, multimask: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mask(s) from a box prompt."""

    # ------------------------------------------------------------------
    # Optional capabilities
    # ------------------------------------------------------------------

    @property
    def model_type_label(self) -> str:
        """Short human-readable label for the loaded model (e.g. 'vit_h')."""
        return self.name

    def supports_autocast(self, device: str) -> bool:
        """Whether autocast (FP16) is safe on this backend + device."""
        return False
