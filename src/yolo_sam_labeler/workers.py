"""SAM inference worker — runs in a background QThread.

Backend-agnostic: delegates the actual SAM ops to a :class:`SamBackend` so
the worker treats SAM 1 / SAM 2 (and future versions) uniformly.

The worker owns an LRU cache of image embeddings keyed by user-supplied
strings (typically absolute file paths plus a ROI suffix). Switching the
*active* image therefore costs an ``O(1)`` state restore instead of a full
image-encoder forward pass, and neighboring images can be warmed up in
advance via :pyattr:`cmd_prefetch`.
"""

import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .backends import SamBackend, SamBackendError, create_backend
from .io_utils import load_image_rgb


# Override with environment variable: ``SAM_EMBEDDING_CACHE=4`` etc.
_DEFAULT_CAPACITY = max(1, int(os.environ.get("SAM_EMBEDDING_CACHE", "16")))


class SamInferenceWorker(QObject):
    """Background worker that owns the SAM backend and an embedding cache."""

    # --- commands (main → worker) ---
    cmd_load = pyqtSignal(str, str, str)        # ckpt, model_type_hint, device_str
    cmd_encode = pyqtSignal(int, str, object)   # gen, key, rgb (HWC uint8)
    cmd_prefetch = pyqtSignal(int, str, str)    # gen, key, image_path
    cmd_predict = pyqtSignal(int, int, int, object)  # gen, x, y, crop_info | None
    cmd_drop_cache = pyqtSignal()

    # --- responses (worker → main) ---
    sig_model_ready = pyqtSignal(str)             # backend label e.g. "SAM 1 vit_h"
    sig_load_failed = pyqtSignal(str)
    sig_encode_done = pyqtSignal(int, str)        # gen, key
    sig_encode_failed = pyqtSignal(str)
    sig_prefetch_done = pyqtSignal(int, str)      # gen, key
    sig_prefetch_failed = pyqtSignal(str)
    sig_predict_done = pyqtSignal(object, int)
    sig_predict_failed = pyqtSignal(str)

    def __init__(self, capacity: int = _DEFAULT_CAPACITY,
                 priority_provider=None):
        super().__init__()
        self._backend: Optional[SamBackend] = None
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._capacity: int = max(1, int(capacity))
        self._active_key: Optional[str] = None
        self._device: Optional[torch.device] = None
        self._priority_provider = priority_provider

    def _is_stale(self, gen: int, key: str, *, prefetch: bool) -> bool:
        """Decide whether a queued slot is still worth running."""
        if self._priority_provider is None:
            return False
        try:
            pri_gen, pri_key = self._priority_provider()
        except Exception:
            return False
        if gen != pri_gen:
            return True
        if not prefetch and pri_key and pri_key != key:
            return True
        return False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @pyqtSlot(str, str, str)
    def do_load(self, ckpt: str, model_type_hint: str, device_str: str):
        try:
            self._backend = create_backend(ckpt, model_type_hint=model_type_hint)
            self._device = torch.device(device_str)
            self._backend.load(ckpt, device_str)
            self._cache.clear()
            self._active_key = None
            self.sig_model_ready.emit(self._backend.model_type_label)
        except SamBackendError as exc:
            self._backend = None
            self.sig_load_failed.emit(str(exc))
        except Exception as exc:
            self._backend = None
            self.sig_load_failed.emit(f"模型加载失败: {exc}")

    @pyqtSlot()
    def do_drop_cache(self):
        self._cache.clear()
        self._active_key = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @pyqtSlot(int, str, object)
    def do_encode(self, gen: int, key: str, rgb: np.ndarray):
        if self._backend is None:
            self.sig_encode_failed.emit("SAM model not loaded")
            return
        if self._is_stale(gen, key, prefetch=False):
            self.sig_encode_done.emit(gen, key)
            return
        try:
            cache_key = key or "<anon>"
            if cache_key in self._cache:
                self._restore(cache_key)
                self.sig_encode_done.emit(gen, cache_key)
                return
            self._encode_to_cache(cache_key, np.asarray(rgb))
            self._restore(cache_key)
            self.sig_encode_done.emit(gen, cache_key)
        except SamBackendError as exc:
            self.sig_encode_failed.emit(str(exc))
        except Exception as exc:
            self.sig_encode_failed.emit(str(exc))

    @pyqtSlot(int, str, str)
    def do_prefetch(self, gen: int, key: str, path: str):
        if self._backend is None:
            self.sig_prefetch_failed.emit("SAM model not loaded")
            return
        if not key:
            self.sig_prefetch_failed.emit("prefetch key is empty")
            return
        if self._is_stale(gen, key, prefetch=True):
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            self.sig_prefetch_done.emit(gen, key)
            return
        try:
            rgb = load_image_rgb(path) if path else None
            if rgb is None:
                self.sig_prefetch_failed.emit(f"prefetch failed to load {path}")
                return
            previous_active = self._active_key
            self._encode_to_cache(key, rgb)
            if previous_active and previous_active in self._cache:
                self._restore(previous_active)
            self.sig_prefetch_done.emit(gen, key)
        except RuntimeError as exc:
            self.sig_prefetch_failed.emit(str(exc))
        except Exception as exc:
            self.sig_prefetch_failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @pyqtSlot(int, int, int, object)
    def do_predict(self, gen: int, x: int, y: int, crop_info):
        if self._backend is None:
            self.sig_predict_failed.emit("SAM model not loaded")
            return
        if self._active_key is None or not self._backend.is_image_set:
            self.sig_predict_failed.emit("SAM image not encoded")
            return
        try:
            ox = int(crop_info.get("x", 0)) if isinstance(crop_info, dict) else 0
            oy = int(crop_info.get("y", 0)) if isinstance(crop_info, dict) else 0

            if isinstance(crop_info, dict) and "box" in crop_info:
                bx1, by1, bx2, by2 = crop_info["box"]
                masks, scores = self._backend.predict_box(
                    bx1 - ox, by1 - oy, bx2 - ox, by2 - oy, multimask=True
                )
            else:
                masks, scores = self._backend.predict_point(
                    x - ox, y - oy, multimask=True
                )

            best = self._best_mask(masks, scores)
            best = self._lift_to_full(best, crop_info)
            self.sig_predict_done.emit(best, gen)
        except SamBackendError as exc:
            self.sig_predict_failed.emit(str(exc))
        except Exception as exc:
            self.sig_predict_failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _encode_to_cache(self, key: str, rgb: np.ndarray):
        rgb = np.ascontiguousarray(np.asarray(rgb))
        self._run_encoder(rgb)
        self._cache[key] = self._backend.snapshot()
        self._cache.move_to_end(key)
        self._evict_to_capacity()
        self._active_key = key

    def _run_encoder(self, rgb: np.ndarray, *, _retried: bool = False):
        """Run the image encoder with NVML retry (Jetson workaround)."""
        try:
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
                torch.cuda.empty_cache()
            self._backend.set_image(rgb)
        except RuntimeError as exc:
            if "NVML_SUCCESS" in str(exc) and not _retried:
                if self._device is not None and self._device.type == "cuda":
                    try:
                        torch.cuda.synchronize(self._device)
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                self._run_encoder(rgb, _retried=True)
            else:
                raise

    def _evict_to_capacity(self):
        while len(self._cache) > self._capacity:
            evicted = False
            for k in list(self._cache.keys()):
                if k != self._active_key:
                    self._cache.pop(k, None)
                    evicted = True
                    break
            if not evicted:
                break

    def _restore(self, key: str):
        if key not in self._cache:
            return
        snap = self._cache[key]
        self._cache.move_to_end(key)
        self._backend.restore(snap)
        self._active_key = key

    @staticmethod
    def _best_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
        best = masks[int(np.argmax(scores))]
        best = np.asarray(best)
        if best.ndim == 3:
            best = np.squeeze(best, axis=0)
        return (best > 0.5).astype(np.uint8)

    @staticmethod
    def _lift_to_full(mask: np.ndarray, crop_info) -> np.ndarray:
        if not isinstance(crop_info, dict):
            return mask
        h_full = int(crop_info.get("h_full") or 0)
        w_full = int(crop_info.get("w_full") or 0)
        if not (h_full and w_full):
            return mask
        ox = int(crop_info.get("x", 0))
        oy = int(crop_info.get("y", 0))
        full = np.zeros((h_full, w_full), dtype=np.uint8)
        ch, cw = mask.shape[:2]
        full[oy:oy + ch, ox:ox + cw] = mask
        return full
