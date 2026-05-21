"""SAM backend abstraction — unifies SAM 1 / SAM 2 (and future SAM 3) APIs.

Each backend wraps a specific SAM family (SAM 1, SAM 2, ...) behind the same
``SamBackend`` interface so the worker thread doesn't care which version is
running. Backend selection is driven by the checkpoint filename.
"""

from .base import SamBackend, SamBackendError
from .factory import detect_backend_kind, create_backend, BACKEND_KIND_SAM1, BACKEND_KIND_SAM2

__all__ = [
    "SamBackend",
    "SamBackendError",
    "detect_backend_kind",
    "create_backend",
    "BACKEND_KIND_SAM1",
    "BACKEND_KIND_SAM2",
]
