"""SAM 1 backend — wraps facebookresearch/segment-anything."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from .base import SamBackend, SamBackendError


# Map filename hash to vit_* model type (SAM 1 official checkpoint names)
_FILENAME_HINTS: dict[str, str] = {
    "sam_vit_h_4b8939": "vit_h",
    "sam_vit_l_0b3195": "vit_l",
    "sam_vit_b_01ec64": "vit_b",
}


def guess_sam1_model_type(checkpoint_path: str) -> str:
    """Guess SAM 1 model type from filename. Defaults to vit_h."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem in _FILENAME_HINTS:
        return _FILENAME_HINTS[stem]
    if "vit_h" in stem:
        return "vit_h"
    if "vit_l" in stem:
        return "vit_l"
    if "vit_b" in stem:
        return "vit_b"
    return "vit_h"


class Sam1Backend(SamBackend):
    """SAM 1 (segment-anything) backend.

    Uses ``segment_anything.SamPredictor`` directly. Snapshot captures the
    predictor's per-image state (features, sizes) so the service-level LRU
    cache can swap images at O(1) cost.
    """

    name = "sam1"
    supports_box = True

    def __init__(self, model_type: Optional[str] = None):
        self._predictor = None
        self._model_type = model_type or "vit_h"
        self._device: Optional[torch.device] = None

    @property
    def model_type_label(self) -> str:
        return f"SAM 1 {self._model_type}"

    def supports_autocast(self, device: str) -> bool:
        # Only on CUDA; CPU FP16 is slower than FP32 in stock wheels.
        return str(device).startswith("cuda") and os.environ.get("SAM_FP16", "1") != "0"

    def load(self, checkpoint: str, device: str) -> None:
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as exc:
            raise SamBackendError(
                "未安装 segment-anything 包。请执行: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        if self._model_type not in sam_model_registry:
            raise SamBackendError(f"未知 SAM 1 模型类型: {self._model_type}")
        try:
            torch_device = torch.device(device)
            sam = sam_model_registry[self._model_type](checkpoint=checkpoint).to(torch_device)
            sam.eval()
            self._predictor = SamPredictor(sam)
            self._device = torch_device
        except Exception as exc:
            raise SamBackendError(f"SAM 1 模型加载失败: {exc}") from exc

    @property
    def is_image_set(self) -> bool:
        return self._predictor is not None and self._predictor.is_image_set

    def set_image(self, rgb: np.ndarray) -> None:
        if self._predictor is None:
            raise SamBackendError("SAM 1 模型未加载")
        rgb = np.ascontiguousarray(np.asarray(rgb))
        with torch.inference_mode():
            if self.supports_autocast(str(self._device)):
                with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                    self._predictor.set_image(rgb)
            else:
                self._predictor.set_image(rgb)

    def snapshot(self) -> dict[str, Any]:
        if self._predictor is None:
            raise SamBackendError("SAM 1 模型未加载")
        p = self._predictor
        return {
            "original_size": p.original_size,
            "input_size": p.input_size,
            "features": p.features,
            "is_image_set": True,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if self._predictor is None:
            raise SamBackendError("SAM 1 模型未加载")
        p = self._predictor
        p.original_size = snapshot["original_size"]
        p.input_size = snapshot["input_size"]
        p.features = snapshot["features"]
        p.is_image_set = True

    def _predict_inference(self, *, point_coords=None, point_labels=None, box=None,
                           multimask: bool = True):
        with torch.inference_mode():
            if self.supports_autocast(str(self._device)):
                with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                    return self._predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        multimask_output=multimask,
                    )
            return self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask,
            )

    def predict_point(self, x: int, y: int, multimask: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
        if self._predictor is None or not self._predictor.is_image_set:
            raise SamBackendError("SAM 图像未编码")
        pts = np.array([[x, y]], dtype=np.float32)
        masks, scores, _ = self._predict_inference(
            point_coords=pts,
            point_labels=np.array([1]),
            multimask=multimask,
        )
        return np.asarray(masks), np.asarray(scores)

    def predict_box(self, x1: int, y1: int, x2: int, y2: int, multimask: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
        if self._predictor is None or not self._predictor.is_image_set:
            raise SamBackendError("SAM 图像未编码")
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        masks, scores, _ = self._predict_inference(box=box, multimask=multimask)
        return np.asarray(masks), np.asarray(scores)
