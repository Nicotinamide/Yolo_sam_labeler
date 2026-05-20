"""YOLO prediction service — async wrapper around ultralytics.

Inference runs in a dedicated ``QThread`` so the UI never freezes on large
images or slow GPUs.  Both segmentation and pure-detection weights are
supported: a detection-only model returns boxes, a segmentation model returns
masks plus boxes.
"""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


# ---------------------------------------------------------------------------
# Result payload
# ---------------------------------------------------------------------------


@dataclass
class YoloPrediction:
    """Output of a single YOLO inference call."""

    masks: list[np.ndarray]
    mask_class_ids: list[int]
    boxes: list[tuple[int, int, int, int]]  # (x1, y1, x2, y2) in original pixels
    box_class_ids: list[int]
    replace: bool

    @property
    def has_masks(self) -> bool:
        return bool(self.masks)

    @property
    def has_boxes(self) -> bool:
        return bool(self.boxes)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class _YoloWorker(QObject):
    """Lives in a dedicated QThread and owns the ultralytics model."""

    cmd_load = pyqtSignal(str)
    cmd_predict = pyqtSignal(object, float, bool)

    sig_load_done = pyqtSignal(str)
    sig_load_failed = pyqtSignal(str)
    sig_predict_done = pyqtSignal(object)
    sig_predict_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._model = None
        self._weights = ""

    @pyqtSlot(str)
    def do_load(self, weights_path: str):
        try:
            from ultralytics import YOLO  # heavy import deferred to worker thread
        except ImportError:
            self.sig_load_failed.emit("未安装 ultralytics。请执行: pip install ultralytics")
            return
        try:
            self._model = YOLO(weights_path)
            self._weights = weights_path
            self.sig_load_done.emit(weights_path)
        except Exception as exc:
            self._model = None
            self._weights = ""
            self.sig_load_failed.emit(f"YOLO 模型加载失败: {exc}")

    @pyqtSlot(object, float, bool)
    def do_predict(self, image_bgr: np.ndarray, conf: float, replace: bool):
        if self._model is None:
            self.sig_predict_failed.emit("YOLO 模型未加载。")
            return
        try:
            results = self._model.predict(
                source=np.ascontiguousarray(image_bgr),
                conf=float(conf),
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            self.sig_predict_failed.emit(f"YOLO 推理失败: {exc}")
            return
        if not results:
            self.sig_predict_done.emit(
                YoloPrediction([], [], [], [], replace)
            )
            return
        prediction = _build_prediction(results[0], image_bgr.shape[:2], replace)
        self.sig_predict_done.emit(prediction)


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------


class YoloService(QObject):
    """Async YOLO inference controller.

    Signals:
        load_done(path)         — model successfully loaded
        load_failed(msg)        — load error
        predict_done(payload)   — :class:`YoloPrediction` ready
        busy_changed(is_busy)   — convenience for UI button enablement
        error(msg)              — uniform error stream
    """

    load_done = pyqtSignal(str)
    load_failed = pyqtSignal(str)
    predict_done = pyqtSignal(object)
    busy_changed = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_YoloWorker] = None
        self._weights_path: str = ""
        self._is_loaded = False
        self._is_busy = False

    @property
    def weights_path(self) -> str:
        return self._weights_path

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    # ---- lifecycle ----

    def load(self, weights_path: str):
        if not weights_path:
            self.error.emit("YOLO 权重路径为空。")
            return
        if not os.path.isfile(weights_path):
            self.error.emit(f"未找到 YOLO 权重: {weights_path}")
            return
        self._is_loaded = False
        self._weights_path = weights_path
        self._ensure_thread()
        self._set_busy(True)
        self._worker.cmd_load.emit(weights_path)

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25,
                replace: bool = True):
        if not self._is_loaded or self._worker is None:
            self.error.emit("YOLO 模型未加载。")
            return
        if self._is_busy:
            self.error.emit("YOLO 正在推理，请稍候。")
            return
        self._set_busy(True)
        # The worker runs in a different thread; ndarray is shared but only
        # read inside the worker, so a contiguous copy is enough.
        self._worker.cmd_predict.emit(np.ascontiguousarray(image_bgr), float(conf), bool(replace))

    def shutdown(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)

    # ---- internals ----

    def _ensure_thread(self):
        if self._thread is not None:
            return
        self._thread = QThread(self)
        self._worker = _YoloWorker()
        self._worker.moveToThread(self._thread)

        self._worker.cmd_load.connect(self._worker.do_load)
        self._worker.cmd_predict.connect(self._worker.do_predict)

        self._worker.sig_load_done.connect(self._on_load_done)
        self._worker.sig_load_failed.connect(self._on_load_failed)
        self._worker.sig_predict_done.connect(self._on_predict_done)
        self._worker.sig_predict_failed.connect(self._on_predict_failed)

        self._thread.start()

    def _set_busy(self, busy: bool):
        if self._is_busy == busy:
            return
        self._is_busy = busy
        self.busy_changed.emit(busy)

    def _on_load_done(self, path: str):
        self._is_loaded = True
        self._set_busy(False)
        self.load_done.emit(path)

    def _on_load_failed(self, msg: str):
        self._is_loaded = False
        self._set_busy(False)
        self.load_failed.emit(msg)
        self.error.emit(msg)

    def _on_predict_done(self, payload: YoloPrediction):
        self._set_busy(False)
        self.predict_done.emit(payload)

    def _on_predict_failed(self, msg: str):
        self._set_busy(False)
        self.error.emit(msg)


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------


def _build_prediction(result, img_shape: tuple, replace: bool) -> YoloPrediction:
    """Convert ultralytics single-image ``Result`` to :class:`YoloPrediction`."""
    h_img, w_img = img_shape
    masks_out: list[np.ndarray] = []
    mask_cls: list[int] = []
    boxes_out: list[tuple[int, int, int, int]] = []
    box_cls: list[int] = []

    boxes = getattr(result, "boxes", None)
    obb = getattr(result, "obb", None)
    if (boxes is None or len(boxes) == 0) and (obb is None or len(obb) == 0):
        return YoloPrediction([], [], [], [], replace)

    if boxes is not None and len(boxes) > 0:
        cls_arr = _to_numpy(boxes.cls).astype(int)
        xyxy = _to_numpy(boxes.xyxy)  # (N, 4) in original pixels
    else:
        # OBB is not a native annotation type in this app.  Ultralytics exposes
        # an axis-aligned xyxy approximation, which matches our detection boxes.
        cls_arr = _to_numpy(obb.cls).astype(int)
        xyxy = _to_numpy(obb.xyxy)

    seg = getattr(result, "masks", None)

    n = xyxy.shape[0]
    for i in range(n):
        cid = int(cls_arr[i]) if i < len(cls_arr) else 0
        x1, y1, x2, y2 = xyxy[i]
        x1i, y1i, x2i, y2i = _clamp_box(x1, y1, x2, y2, w_img, h_img)
        if x2i - x1i < 3 or y2i - y1i < 3:
            continue

        added_mask = False
        if seg is not None:
            m = _mask_from_yolo_result(seg, i, h_img, w_img)
            if m is not None:
                masks_out.append(m)
                mask_cls.append(cid)
                added_mask = True

        # Detection-only models (or tiny mask) fall back to a clean box.
        if not added_mask:
            boxes_out.append((x1i, y1i, x2i, y2i))
            box_cls.append(cid)

    return YoloPrediction(masks_out, mask_cls, boxes_out, box_cls, replace)


def _to_numpy(value) -> np.ndarray:
    """Convert torch/Ultralytics tensors or numpy arrays to a CPU ndarray."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _clamp_box(x1, y1, x2, y2, w_img: int, h_img: int) -> tuple[int, int, int, int]:
    x1i = max(0, min(w_img - 1, int(round(float(x1)))))
    y1i = max(0, min(h_img - 1, int(round(float(y1)))))
    x2i = max(0, min(w_img - 1, int(round(float(x2)))))
    y2i = max(0, min(h_img - 1, int(round(float(y2)))))
    if x2i < x1i:
        x1i, x2i = x2i, x1i
    if y2i < y1i:
        y1i, y2i = y2i, y1i
    return x1i, y1i, x2i, y2i


def _mask_from_yolo_result(seg, index: int, h_img: int, w_img: int) -> Optional[np.ndarray]:
    """Return a binary mask in original-image coordinates for one YOLO result.

    Ultralytics ``masks.data`` is the raw float prototype (typically 160×160
    or model input size).  We prefer it over ``masks.xy`` because upsampling
    the continuous float values with bilinear interpolation produces smoother
    edges than the polygon path (which uses CHAIN_APPROX_SIMPLE and loses
    curvature).  No post-hoc smoothing is applied — the mask reflects the
    model's actual output fidelity.
    """
    # --- preferred path: raw float prototype (continuous, bilinear upsample) ---
    data_mask = _mask_from_data(seg, index, h_img, w_img)
    if data_mask is not None:
        return data_mask

    # --- fallback: polygon contour ---
    mask = _mask_from_segments(_segments_at(seg, "xy", index), h_img, w_img)
    if mask is not None:
        return mask

    mask = _mask_from_normalized_segments(_segments_at(seg, "xyn", index), h_img, w_img)
    return mask


def _mask_from_data(seg, index: int, h_img: int, w_img: int) -> Optional[np.ndarray]:
    """Upsample the raw float mask prototype back to original image coords.

    Bilinear interpolation on the continuous float values preserves whatever
    curvature the model learned — this is the most faithful representation
    of what the model actually predicted.
    """
    data = getattr(seg, "data", None)
    if data is None:
        return None
    seg_data = _to_numpy(data)
    if seg_data.size == 0 or index >= seg_data.shape[0]:
        return None
    mi = np.asarray(seg_data[index], dtype=np.float32)
    while mi.ndim > 2 and mi.shape[0] == 1:
        mi = mi[0]
    if mi.ndim != 2:
        return None

    mh, mw = mi.shape
    if (mh, mw) == (h_img, w_img):
        upsampled = mi
    else:
        upsampled = cv2.resize(
            mi, (w_img, h_img), interpolation=cv2.INTER_LINEAR
        )

    m = (upsampled > 0.5).astype(np.uint8)
    return m if int(m.sum()) >= 30 else None


def _segments_at(seg, attr: str, index: int):
    try:
        segments = getattr(seg, attr)
    except Exception:
        return None
    if segments is None or index >= len(segments):
        return None
    return segments[index]


def _mask_from_normalized_segments(segments, h_img: int, w_img: int) -> Optional[np.ndarray]:
    if segments is None:
        return None
    try:
        arr = np.asarray(segments, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    scaled = arr.copy()
    scaled[:, 0] *= w_img
    scaled[:, 1] *= h_img
    return _mask_from_segments(scaled, h_img, w_img)


def _mask_from_segments(segments, h_img: int, w_img: int) -> Optional[np.ndarray]:
    if segments is None:
        return None
    try:
        arr = np.asarray(segments, dtype=np.float32).copy()
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        return None
    arr[:, 0] = np.clip(arr[:, 0], 0, w_img - 1)
    arr[:, 1] = np.clip(arr[:, 1], 0, h_img - 1)
    pts = np.rint(arr).astype(np.int32)
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask if int(mask.sum()) >= 30 else None
