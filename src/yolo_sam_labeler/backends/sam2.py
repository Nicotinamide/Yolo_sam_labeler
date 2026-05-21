"""SAM 2 backend — wraps facebookresearch/sam2."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from .base import SamBackend, SamBackendError


# Map SAM 2 checkpoint filenames → (config_name, label).
# SAM 2 ships with config files; we hardcode the standard mapping here.
_SAM2_CONFIGS: dict[str, tuple[str, str]] = {
    # SAM 2.1 (recommended)
    "sam2.1_hiera_tiny":       ("configs/sam2.1/sam2.1_hiera_t.yaml", "hiera_tiny"),
    "sam2.1_hiera_small":      ("configs/sam2.1/sam2.1_hiera_s.yaml", "hiera_small"),
    "sam2.1_hiera_base_plus":  ("configs/sam2.1/sam2.1_hiera_b+.yaml", "hiera_base+"),
    "sam2.1_hiera_large":      ("configs/sam2.1/sam2.1_hiera_l.yaml", "hiera_large"),
    # SAM 2.0 (older)
    "sam2_hiera_tiny":         ("configs/sam2/sam2_hiera_t.yaml", "hiera_tiny"),
    "sam2_hiera_small":        ("configs/sam2/sam2_hiera_s.yaml", "hiera_small"),
    "sam2_hiera_base_plus":    ("configs/sam2/sam2_hiera_b+.yaml", "hiera_base+"),
    "sam2_hiera_large":        ("configs/sam2/sam2_hiera_l.yaml", "hiera_large"),
}


def looks_like_sam2(checkpoint_path: str) -> bool:
    """Return True if the filename looks like a SAM 2 checkpoint."""
    name = os.path.basename(checkpoint_path).lower()
    return name.startswith("sam2") or name.startswith("sam_2")


def guess_sam2_config(checkpoint_path: str) -> Optional[str]:
    """Guess the SAM 2 config name from filename. Returns None if unknown."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem in _SAM2_CONFIGS:
        return _SAM2_CONFIGS[stem][0]
    return None


def guess_sam2_label(checkpoint_path: str) -> str:
    """Short human-readable label like 'hiera_large'."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem in _SAM2_CONFIGS:
        return _SAM2_CONFIGS[stem][1]
    return stem


class Sam2Backend(SamBackend):
    """SAM 2 backend using ``sam2.SAM2ImagePredictor``.

    SAM 2's ``predict()`` API mirrors SAM 1: point_coords/point_labels/box,
    multimask_output, returns (masks, scores, low_res_masks). The predictor
    state ``_features`` and ``_orig_hw`` plays the role of SAM 1's features.
    """

    name = "sam2"
    supports_box = True

    def __init__(self):
        self._predictor = None
        self._device: Optional[torch.device] = None
        self._config_label: str = ""

    @property
    def model_type_label(self) -> str:
        return f"SAM 2 {self._config_label}" if self._config_label else "SAM 2"

    def supports_autocast(self, device: str) -> bool:
        return str(device).startswith("cuda") and os.environ.get("SAM_FP16", "1") != "0"

    def load(self, checkpoint: str, device: str) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise SamBackendError(
                "未安装 sam2 包。请执行: "
                "pip install git+https://github.com/facebookresearch/sam2.git"
            ) from exc

        config = guess_sam2_config(checkpoint)
        if config is None:
            raise SamBackendError(
                f"无法识别 SAM 2 配置: {os.path.basename(checkpoint)}\n"
                f"支持的文件名: {', '.join(_SAM2_CONFIGS.keys())}"
            )

        try:
            torch_device = torch.device(device)
            sam2_model = build_sam2(config, checkpoint, device=torch_device)
            self._predictor = SAM2ImagePredictor(sam2_model)
            self._device = torch_device
            self._config_label = guess_sam2_label(checkpoint)
        except Exception as exc:
            raise SamBackendError(f"SAM 2 模型加载失败: {exc}") from exc

    @property
    def is_image_set(self) -> bool:
        return self._predictor is not None and getattr(self._predictor, "_is_image_set", False)

    def set_image(self, rgb: np.ndarray) -> None:
        if self._predictor is None:
            raise SamBackendError("SAM 2 模型未加载")
        rgb = np.ascontiguousarray(np.asarray(rgb))
        with torch.inference_mode():
            if self.supports_autocast(str(self._device)):
                with torch.autocast(device_type=self._device.type, dtype=torch.float16):
                    self._predictor.set_image(rgb)
            else:
                self._predictor.set_image(rgb)

    def snapshot(self) -> dict[str, Any]:
        if self._predictor is None:
            raise SamBackendError("SAM 2 模型未加载")
        p = self._predictor
        # SAM 2 predictor state — these attribute names come from sam2 source
        return {
            "_features": p._features,
            "_orig_hw": p._orig_hw,
            "_is_image_set": True,
            "_is_batch": getattr(p, "_is_batch", False),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if self._predictor is None:
            raise SamBackendError("SAM 2 模型未加载")
        p = self._predictor
        p._features = snapshot["_features"]
        p._orig_hw = snapshot["_orig_hw"]
        p._is_image_set = True
        if "_is_batch" in snapshot:
            p._is_batch = snapshot["_is_batch"]

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
        if not self.is_image_set:
            raise SamBackendError("SAM 2 图像未编码")
        pts = np.array([[x, y]], dtype=np.float32)
        masks, scores, _ = self._predict_inference(
            point_coords=pts,
            point_labels=np.array([1]),
            multimask=multimask,
        )
        return np.asarray(masks), np.asarray(scores)

    def predict_box(self, x1: int, y1: int, x2: int, y2: int, multimask: bool = True
                    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_image_set:
            raise SamBackendError("SAM 2 图像未编码")
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        masks, scores, _ = self._predict_inference(box=box, multimask=multimask)
        return np.asarray(masks), np.asarray(scores)
