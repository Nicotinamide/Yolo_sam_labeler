"""SAM inference worker — runs in a background QThread.

The worker owns an LRU cache of image embeddings keyed by user-supplied
strings (typically absolute file paths plus a ROI suffix).  Switching the
*active* image therefore costs an ``O(1)`` state restore instead of a full
image-encoder forward pass, and neighboring images can be warmed up in
advance via :pyattr:`cmd_prefetch`.
"""

import os
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from segment_anything import sam_model_registry, SamPredictor

from .io_utils import load_image_rgb


# Override with environment variable: ``SAM_EMBEDDING_CACHE=4`` etc.
# Default 16: covers a typical "browse back and forth across the last
# ~16 images" workflow without re-encoding. ViT-H embedding is ~4 MB so the
# cache stays under 70 MB even at this size.
_DEFAULT_CAPACITY = max(1, int(os.environ.get("SAM_EMBEDDING_CACHE", "16")))


class SamInferenceWorker(QObject):
    """Background worker that owns the SAM model and an embedding cache."""

    # --- commands (main → worker) ---
    cmd_load = pyqtSignal(str, str, str)        # ckpt, model_type, device_str
    cmd_encode = pyqtSignal(int, str, object)   # gen, key, rgb (HWC uint8)
    cmd_prefetch = pyqtSignal(int, str, str)    # gen, key, image_path
    cmd_predict = pyqtSignal(int, int, int, object)  # gen, x, y, crop_info | None
    cmd_drop_cache = pyqtSignal()

    # --- responses (worker → main) ---
    sig_model_ready = pyqtSignal()
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
        self._predictor: Optional[SamPredictor] = None
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._capacity: int = max(1, int(capacity))
        self._active_key: Optional[str] = None
        self._device: Optional[torch.device] = None
        self._use_autocast: bool = False
        # Optional callback returning (priority_gen, priority_key) — set by
        # the service so the worker can skip stale prefetch requests.
        self._priority_provider = priority_provider

    def _is_stale(self, gen: int, key: str, *, prefetch: bool) -> bool:
        """Decide whether a queued slot is still worth running.

        Rules:
        - Generation mismatch → always stale (model reload, dir switch).
        - For active *encode* slots: if the service has set a non-empty
          priority key and it differs from ours, treat as stale.
        - Prefetch slots are allowed to lag behind the active key — they
          only get dropped on generation mismatch.
        """
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
    def do_load(self, ckpt: str, model_type: str, device_str: str):
        try:
            device = torch.device(device_str)
            sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
            sam.eval()  # disable dropout/BN training mode for inference
            self._predictor = SamPredictor(sam)
            self._device = device
            # FP16 autocast on CUDA cuts ViT-H forward time roughly in half
            # with no measurable quality loss. CPU fp16 is usually slower
            # than fp32 in stock PyTorch wheels, so we keep CPU at fp32.
            # Set ``SAM_FP16=0`` to disable.
            self._use_autocast = (
                device.type == "cuda"
                and os.environ.get("SAM_FP16", "1") != "0"
            )
            self._cache.clear()
            self._active_key = None
            self.sig_model_ready.emit()
        except Exception as exc:
            self.sig_load_failed.emit(str(exc))

    @pyqtSlot()
    def do_drop_cache(self):
        self._cache.clear()
        self._active_key = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @pyqtSlot(int, str, object)
    def do_encode(self, gen: int, key: str, rgb: np.ndarray):
        if self._predictor is None:
            self.sig_encode_failed.emit("SAM model not loaded")
            return
        # Drop stale encodes early — generation mismatch or the user has
        # moved on to another image while this slot sat in the queue.
        if self._is_stale(gen, key, prefetch=False):
            # Still emit done so the service can release its busy flag and
            # the UI can clear "encoding…" status. The service compares
            # ``gen`` against its current generation and will treat this as
            # a stale callback automatically.
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
        except Exception as exc:
            self.sig_encode_failed.emit(str(exc))

    @pyqtSlot(int, str, str)
    def do_prefetch(self, gen: int, key: str, path: str):
        if self._predictor is None:
            self.sig_prefetch_failed.emit("SAM model not loaded")
            return
        if not key:
            self.sig_prefetch_failed.emit("prefetch key is empty")
            return
        # Stale prefetch (model reloaded etc.) — silently drop.
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
            # Restore the user-facing image so subsequent predicts hit it.
            if previous_active and previous_active in self._cache:
                self._restore(previous_active)
            self.sig_prefetch_done.emit(gen, key)
        except RuntimeError as exc:
            # On Jetson / certain drivers, NVML assertions can persist even
            # after retry.  Prefetch is best-effort — don't crash the worker.
            self.sig_prefetch_failed.emit(str(exc))
        except Exception as exc:
            self.sig_prefetch_failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @pyqtSlot(int, int, int, object)
    def do_predict(self, gen: int, x: int, y: int, crop_info):
        if self._predictor is None:
            self.sig_predict_failed.emit("SAM model not loaded")
            return
        if self._active_key is None or not self._predictor.is_image_set:
            self.sig_predict_failed.emit("SAM image not encoded")
            return
        try:
            ox = int(crop_info.get("x", 0)) if isinstance(crop_info, dict) else 0
            oy = int(crop_info.get("y", 0)) if isinstance(crop_info, dict) else 0

            with torch.inference_mode():
                if self._use_autocast and self._device is not None:
                    autocast_ctx = torch.autocast(
                        device_type=self._device.type, dtype=torch.float16
                    )
                else:
                    autocast_ctx = nullcontext()
                with autocast_ctx:
                    if isinstance(crop_info, dict) and "box" in crop_info:
                        bx1, by1, bx2, by2 = crop_info["box"]
                        box = np.array(
                            [bx1 - ox, by1 - oy, bx2 - ox, by2 - oy], dtype=np.float32
                        )
                        masks, scores, _ = self._predictor.predict(
                            box=box,
                            multimask_output=True,
                        )
                    else:
                        pts = np.array([[x - ox, y - oy]], dtype=np.float32)
                        masks, scores, _ = self._predictor.predict(
                            point_coords=pts,
                            point_labels=np.array([1]),
                            multimask_output=True,
                        )

            best = self._best_mask(masks, scores)
            best = self._lift_to_full(best, crop_info)
            self.sig_predict_done.emit(best, gen)
        except Exception as exc:
            self.sig_predict_failed.emit(str(exc))

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _encode_to_cache(self, key: str, rgb: np.ndarray):
        rgb = np.ascontiguousarray(np.asarray(rgb))
        # Inference-only: skip autograd bookkeeping. ``set_image`` runs the
        # ViT image encoder which is the dominant cost; without ``no_grad``
        # PyTorch stashes intermediate activations for a backward pass that
        # never happens.
        self._run_encoder(rgb)
        self._cache[key] = self._snapshot()
        self._cache.move_to_end(key)
        self._evict_to_capacity()
        self._active_key = key

    def _run_encoder(self, rgb: np.ndarray, *, _retried: bool = False):
        """Run the ViT image encoder with retry on NVML assertion failures.

        On Jetson / certain CUDA driver versions, concurrent NVML calls from
        the CUDACachingAllocator can trigger an internal assertion
        (``NVML_SUCCESS == r``).  A single retry after clearing the CUDA cache
        and synchronizing typically resolves the transient failure.
        """
        try:
            # Proactively release cached CUDA memory to reduce the chance of
            # the allocator needing to call NVML during the forward pass.
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
                torch.cuda.empty_cache()

            with torch.inference_mode():
                if self._use_autocast and self._device is not None:
                    with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                        self._predictor.set_image(rgb)
                else:
                    self._predictor.set_image(rgb)
        except RuntimeError as exc:
            if "NVML_SUCCESS" in str(exc) and not _retried:
                # Transient NVML assertion — sync, clear cache, and retry once.
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
                break  # only the active entry is left

    def _snapshot(self) -> dict:
        p = self._predictor
        return {
            "original_size": p.original_size,
            "input_size": p.input_size,
            "features": p.features,
            "is_image_set": True,
        }

    def _restore(self, key: str):
        if key not in self._cache:
            return
        snap = self._cache[key]
        self._cache.move_to_end(key)
        p = self._predictor
        p.original_size = snap["original_size"]
        p.input_size = snap["input_size"]
        p.features = snap["features"]
        p.is_image_set = True
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
