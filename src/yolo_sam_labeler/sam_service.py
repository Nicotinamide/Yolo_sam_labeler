"""SAM service: model lifecycle, async predict, embedding cache, prefetch."""

import os
import urllib.request
from typing import Optional

import numpy as np
import torch
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel,
    QProgressBar, QPushButton, QMessageBox,
)

from .workers import SamInferenceWorker

# ---------------------------------------------------------------------------
# Auto-download config
# ---------------------------------------------------------------------------

# SAM 1 (original)
SAM_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
SAM_MODEL_FILES = {k: os.path.basename(v) for k, v in SAM_MODEL_URLS.items()}
SAM_FILE_SIZES = {"vit_h": 2564550879, "vit_l": 1251542702, "vit_b": 375042383}

# SAM 2.1 (recommended over 2.0 — better in low light, low res)
SAM2_MODEL_URLS = {
    "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}
SAM2_MODEL_FILES = {k: os.path.basename(v) for k, v in SAM2_MODEL_URLS.items()}
# Approximate sizes (bytes)
SAM2_FILE_SIZES = {
    "sam2.1_hiera_tiny": 156_000_000,
    "sam2.1_hiera_small": 184_000_000,
    "sam2.1_hiera_base_plus": 323_000_000,
    "sam2.1_hiera_large": 898_000_000,
}


# ---------------------------------------------------------------------------
# SAM service
# ---------------------------------------------------------------------------


class SamService(QObject):
    """Manages SAM lifecycle, async predicts, embedding cache and prefetch.

    Signals:
        model_ready()                       — SAM loaded
        load_failed(msg)                    — load error
        encode_started(key)                 — encode/restore submitted
        encode_done(gen, key)               — active image is ready
        prefetch_done(gen, key)             — neighboring image cached
        prediction_ready(mask, gen)         — prediction result
        prediction_failed(msg)
        error(msg)                          — uniform error stream
    """

    model_ready = pyqtSignal(str)            # backend label e.g. "SAM 1 vit_h"
    load_failed = pyqtSignal(str)
    encode_started = pyqtSignal(str)
    encode_done = pyqtSignal(int, str)
    prefetch_done = pyqtSignal(int, str)
    prediction_ready = pyqtSignal(object, int)
    prediction_failed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: SamInferenceWorker | None = None
        self._thread: QThread | None = None
        self._ready = False
        self._gen = 0
        self._is_busy = False
        self._encode_in_flight: Optional[str] = None
        self._loaded_config: tuple[str, str, str] | None = None
        self._pending_load_config: tuple[str, str, str] | None = None
        self._backend_label: str = ""

        # Active image state
        self._active_key: Optional[str] = None
        self._cached_keys: set[str] = set()
        self._crop_info: Optional[dict] = None
        self._pending_prompt: Optional[tuple[str, tuple]] = None

        # Cross-thread "what does the user actually want right now". The
        # worker reads this just before each prefetch/encode slot to bail
        # out of stale requests early.
        from threading import RLock
        self._priority_lock = RLock()
        self._priority_gen = 0
        self._priority_key: Optional[str] = None

    # ---- properties ----

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_encoded(self) -> bool:
        return self._active_key is not None and self._active_key in self._cached_keys

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def loaded_config(self) -> tuple[str, str, str] | None:
        return self._loaded_config

    @property
    def backend_label(self) -> str:
        """Human-readable label of the loaded backend (e.g. 'SAM 1 vit_h')."""
        return self._backend_label

    @property
    def active_key(self) -> Optional[str]:
        return self._active_key

    @property
    def crop_info(self) -> Optional[dict]:
        return self._crop_info

    def has_cached(self, key: str) -> bool:
        return bool(key) and key in self._cached_keys

    # ---- model lifecycle ----

    def load(self, checkpoint: str, model_type_hint: str = "",
             device_str: str | None = None) -> bool:
        """Load a SAM checkpoint. Backend (SAM 1/2/3) is auto-detected from filename.

        Args:
            checkpoint: path to .pth/.pt file
            model_type_hint: optional override for SAM 1 (vit_h/vit_l/vit_b).
                Ignored for SAM 2 (config inferred from filename).
            device_str: 'cuda', 'cpu', or None for auto.
        """
        if device_str is None or device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        config = (os.path.abspath(checkpoint), model_type_hint, device_str)
        if self._ready and self._loaded_config == config:
            self.model_ready.emit(self._backend_label)
            return False
        self._ready = False
        self._loaded_config = None
        self._cached_keys.clear()
        self._active_key = None
        self._is_busy = True
        self._encode_in_flight = None
        self._pending_prompt = None
        self._ensure_thread()
        self._pending_load_config = config
        self._worker.cmd_load.emit(checkpoint, model_type_hint, device_str)
        return True

    def shutdown(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)

    # ---- active image management ----

    def invalidate_image(self):
        """Bump generation; in-flight callbacks are dropped, cache survives."""
        self._gen += 1
        self._is_busy = False
        self._encode_in_flight = None
        self._pending_prompt = None
        self._active_key = None
        self._crop_info = None
        # Clear priority — anything still queued is now stale by definition.
        with self._priority_lock:
            self._priority_gen = self._gen
            self._priority_key = None

    def drop_cache(self):
        """Drop every cached embedding (called when ROI mode toggles)."""
        self._cached_keys.clear()
        self._active_key = None
        self._crop_info = None
        if self._worker is not None:
            self._worker.cmd_drop_cache.emit()

    # ---- encoding ----

    def encode(self, rgb: np.ndarray, key: str,
               crop_info: Optional[dict] = None) -> bool:
        """Encode (or restore from cache) the active image identified by ``key``."""
        if not self.is_ready or not key:
            return False
        if rgb is None or rgb.size == 0:
            return False
        self._active_key = key
        self._crop_info = dict(crop_info) if crop_info else None
        self._set_priority(key)

        # Always hand the RGB to the worker; it skips re-encoding on cache hit.
        self._is_busy = True
        self._encode_in_flight = key
        self.encode_started.emit(key)
        self._worker.cmd_encode.emit(self._gen, key, np.ascontiguousarray(rgb))
        return True

    def prefetch(self, key: str, image_path: str) -> bool:
        """Warm up an embedding for a neighboring image.

        Prefetch deliberately does **not** raise priority; the worker will
        skip prefetch slots if the active key has changed.
        """
        if not self.is_ready or not key or not image_path:
            return False
        if key in self._cached_keys:
            return False
        self._worker.cmd_prefetch.emit(self._gen, key, image_path)
        return True

    # ---- prediction ----

    def predict_async(self, x: int, y: int) -> bool:
        if not self.is_ready:
            return False
        if not self.is_encoded:
            self._pending_prompt = ("point", (x, y))
            return False
        if self._is_busy:
            # Busy with encode/prefetch — queue the click for when it completes.
            self._pending_prompt = ("point", (x, y))
            return False
        self._is_busy = True
        self._worker.cmd_predict.emit(self._gen, int(x), int(y), self._crop_info)
        return True

    def predict_box_async(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        if not self.is_ready:
            return False
        box = (int(x1), int(y1), int(x2), int(y2))
        if not self.is_encoded:
            self._pending_prompt = ("box", box)
            return False
        if self._is_busy:
            self._pending_prompt = ("box", box)
            return False
        self._is_busy = True
        info = dict(self._crop_info) if self._crop_info else {}
        info["box"] = box
        self._worker.cmd_predict.emit(self._gen, -1, -1, info)
        return True

    # ---- priority shared with worker ----

    def _set_priority(self, key: str):
        with self._priority_lock:
            self._priority_gen = self._gen
            self._priority_key = key

    def _priority_snapshot(self) -> tuple[int, Optional[str]]:
        with self._priority_lock:
            return self._priority_gen, self._priority_key

    # ---- internals ----

    def _ensure_thread(self):
        if self._thread is not None:
            return
        self._thread = QThread(self)
        self._worker = SamInferenceWorker(priority_provider=self._priority_snapshot)
        self._worker.moveToThread(self._thread)

        # Cross-thread: service (main) → worker (background)
        self._worker.cmd_load.connect(self._worker.do_load)
        self._worker.cmd_encode.connect(self._worker.do_encode)
        self._worker.cmd_prefetch.connect(self._worker.do_prefetch)
        self._worker.cmd_predict.connect(self._worker.do_predict)
        self._worker.cmd_drop_cache.connect(self._worker.do_drop_cache)

        # Cross-thread: worker (background) → service (main)
        self._worker.sig_model_ready.connect(self._on_model_ready)
        self._worker.sig_load_failed.connect(self._on_load_failed)
        self._worker.sig_encode_done.connect(self._on_encode_done)
        self._worker.sig_encode_failed.connect(self._on_encode_failed)
        self._worker.sig_prefetch_done.connect(self._on_prefetch_done)
        self._worker.sig_prefetch_failed.connect(self._on_prefetch_failed)
        self._worker.sig_predict_done.connect(self._on_predict_done)
        self._worker.sig_predict_failed.connect(self._on_predict_failed)

        self._thread.start()

    def _on_model_ready(self, label: str):
        self._ready = True
        self._is_busy = False
        self._loaded_config = self._pending_load_config
        self._backend_label = label
        self.model_ready.emit(label)

    def _on_load_failed(self, msg: str):
        self._ready = False
        self._is_busy = False
        self._loaded_config = None
        self.load_failed.emit(msg)
        self.error.emit(msg)

    def _on_encode_done(self, gen: int, key: str):
        # Always release the busy flag and reset the in-flight slot, even
        # when the result is stale — otherwise UI status sticks forever.
        self._is_busy = False
        if self._encode_in_flight == key:
            self._encode_in_flight = None

        if gen != self._gen:
            # Stale: still emit so the UI can clear "encoding…" if it cares.
            self.encode_done.emit(gen, key)
            return

        self._cached_keys.add(key)
        if self._active_key is None:
            self._active_key = key
        self.encode_done.emit(gen, key)

        # Drain any pending click that was queued while encoding.
        if self._pending_prompt and self._active_key == key:
            kind, payload = self._pending_prompt
            self._pending_prompt = None
            if kind == "point":
                px, py = payload
                self.predict_async(px, py)
            elif kind == "box":
                x1, y1, x2, y2 = payload
                self.predict_box_async(x1, y1, x2, y2)

    def _on_encode_failed(self, msg: str):
        self._is_busy = False
        self._encode_in_flight = None
        self.error.emit(msg)

    def _on_prefetch_done(self, gen: int, key: str):
        if gen != self._gen:
            return
        self._cached_keys.add(key)
        self.prefetch_done.emit(gen, key)

    def _on_prefetch_failed(self, msg: str):
        self.error.emit(f"SAM 预取失败: {msg}")

    def _on_predict_done(self, mask, gen: int):
        if gen != self._gen:
            return
        self._is_busy = False
        self.prediction_ready.emit(mask, gen)
        # Drain queued click that arrived while this prediction was running.
        self._drain_pending()

    def _on_predict_failed(self, msg: str):
        self._is_busy = False
        self.prediction_failed.emit(msg)
        self.error.emit(msg)
        self._drain_pending()

    def _drain_pending(self):
        """Execute a queued point/box prompt if the service is free."""
        if self._pending_prompt and not self._is_busy and self.is_encoded:
            kind, payload = self._pending_prompt
            self._pending_prompt = None
            if kind == "point":
                self.predict_async(*payload)
            elif kind == "box":
                self.predict_box_async(*payload)


# ---------------------------------------------------------------------------
# SAM checkpoint downloader (unchanged)
# ---------------------------------------------------------------------------


def download_sam_checkpoint(parent, model_type: str, save_path: str) -> bool:
    """Download SAM checkpoint with progress dialog. Returns True on success.

    Resolves URL from both SAM 1 (vit_h/l/b) and SAM 2 (sam2.1_*) registries.
    """
    url = SAM_MODEL_URLS.get(model_type) or SAM2_MODEL_URLS.get(model_type)
    if not url:
        QMessageBox.critical(parent, "下载失败", f"不支持的模型类型: {model_type}")
        return False

    expected = SAM_FILE_SIZES.get(model_type) or SAM2_FILE_SIZES.get(model_type, 0)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    dlg = QDialog(parent)
    dlg.setWindowTitle("下载 SAM 权重")
    dlg.setFixedSize(480, 120)
    layout = QVBoxLayout(dlg)
    label = QLabel(f"正在从 Meta 官方下载 SAM {model_type} 权重…")
    label.setWordWrap(True)
    layout.addWidget(label)
    bar = QProgressBar()
    bar.setMaximum(100)
    layout.addWidget(bar)

    cancelled = [False]

    def on_cancel():
        cancelled[0] = True
        dlg.reject()

    btn = QPushButton("取消")
    btn.clicked.connect(on_cancel)
    layout.addWidget(btn)
    dlg.show()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = resp.length or expected
            downloaded = 0
            part_path = save_path + ".part"
            with open(part_path, "wb") as f:
                while True:
                    QApplication.processEvents()
                    if cancelled[0]:
                        if os.path.exists(part_path):
                            os.remove(part_path)
                        return False
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(downloaded * 100 / total)
                        bar.setValue(pct)
                        label.setText(
                            f"下载中… {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
                        )
            os.rename(part_path, save_path)
            return True
    except Exception as e:
        QMessageBox.critical(parent, "下载失败", f"无法下载 SAM 权重:\n{e}")
        for p in (save_path, save_path + ".part"):
            if os.path.exists(p):
                os.remove(p)
        return False
    finally:
        dlg.close()
