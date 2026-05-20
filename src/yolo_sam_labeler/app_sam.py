"""SAM and ROI controller logic (mixin for MainWindow)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from .sam_service import download_sam_checkpoint, SAM_MODEL_FILES

if TYPE_CHECKING:
    from .sam_service import SamService
    from .sidebar import Sidebar


class SamControllerMixin:
    """Mixin that provides SAM loading, encoding, prediction, and ROI methods.

    Expects the host class to expose:
        sam: SamService
        image_rgb, image_bgr, image_shape, image_paths, index
        coords, store, classes, sidebar, current_class_id
        sam_checkpoint, model_type, device
        roi_mode, roi_pts, roi_mask
        _encode_debounce: QTimer
        _prefetch_debounce: QTimer
        _encode_reason: str
        hover_kind, hover_idx
        _log(msg, level)
        _refresh_canvas()
        _update_header()
        _remember_paths()
        _encode_key_for(path) -> str
        lbl_filename: QLabel
    """

    # ------------------------------------------------------------------
    # SAM loading
    # ------------------------------------------------------------------

    def _load_sam(self):
        self.model_type = self.sidebar.combo_model.currentText()
        ckpt = self.sam_checkpoint
        mt = self.model_type
        if not ckpt:
            ckpt = self._fallback_sam_checkpoint(mt)
            self.sam_checkpoint = ckpt
            self.sidebar.set_checkpoint_label(ckpt)
        if not ckpt:
            QMessageBox.warning(self, "无法加载", "请先选择 SAM 权重文件。")
            return
        if not os.path.isfile(ckpt):
            default_name = SAM_MODEL_FILES.get(mt)
            if os.path.basename(ckpt) != default_name:
                QMessageBox.warning(self, "无法加载", f"未找到权重文件:\n{ckpt}")
                return
            reply = QMessageBox.question(
                self, "权重文件缺失",
                f"未找到 SAM 权重:\n{ckpt}\n\n是否自动从 Meta 官方下载？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes or not download_sam_checkpoint(self, mt, ckpt):
                return
        config = (os.path.abspath(ckpt), mt, str(self.device))
        if self.sam.is_ready and self.sam.loaded_config == config:
            self._log("SAM 已加载，直接使用当前图像。", "info")
            if not self.sam.is_encoded:
                self._encode_debounce.stop()
                self._encode_current_image("当前图像")
            return
        self._log(f"正在后台加载 SAM ({mt}, {self.device})…", "info")
        self.sam.load(ckpt, mt, str(self.device))
        self._remember_paths()

    def _auto_load_sam(self):
        if self.sam.is_ready:
            return
        if not self.sam_checkpoint:
            self.sam_checkpoint = self._fallback_sam_checkpoint(self.model_type)
            self.sidebar.set_checkpoint_label(self.sam_checkpoint)
        if self.sam_checkpoint and os.path.isfile(self.sam_checkpoint):
            self._log("自动加载 SAM 权重。", "info")
            self._load_sam()
        elif self.sam_checkpoint:
            self._log(f"SAM 权重不存在，未自动加载: {self.sam_checkpoint}", "warn")

    def _on_load_sam(self, ckpt: str, model_type: str):
        if ckpt:
            self.sam_checkpoint = ckpt
        self.model_type = model_type
        self.sidebar.set_checkpoint_label(self.sam_checkpoint)
        self._load_sam()

    def _on_sam_ready(self):
        device_str = "CUDA" if torch.cuda.is_available() else "CPU"
        budget = {
            ("CUDA", "vit_h"): "≈0.3s",
            ("CUDA", "vit_l"): "≈0.2s",
            ("CUDA", "vit_b"): "≈0.1s",
            ("CPU", "vit_h"): "≈8–15s",
            ("CPU", "vit_l"): "≈4–8s",
            ("CPU", "vit_b"): "≈1–2s",
        }.get((device_str, self.model_type), "")
        suffix = f"，单图编码 {budget}" if budget else ""
        self._log(f"SAM 模型已加载 ({self.model_type} on {device_str}){suffix}", "ok")
        if device_str == "CPU" and self.model_type == "vit_h":
            self._log("提示：CPU 跑 vit_h 较慢，可在侧栏切换到 vit_b 提速。", "info")
        self._encode_debounce.stop()
        self._encode_current_image("当前图像")
        self._schedule_prefetch()

    def _on_sam_encode_started(self, key: str):
        if self.image_paths and 0 <= self.index < len(self.image_paths):
            current = os.path.abspath(self.image_paths[self.index])
            if key.startswith(current):
                self.lbl_filename.setText(
                    f"{os.path.basename(self.image_paths[self.index])}  · SAM 编码中…"
                )

    def _on_sam_encode_done(self, gen: int, key: str):
        self._update_header()
        self._schedule_prefetch()

    def _on_sam_prefetch_done(self, gen: int, key: str):
        pass

    def _on_sam_error(self, msg: str):
        QMessageBox.critical(self, "SAM 错误", msg)

    # ------------------------------------------------------------------
    # SAM encoding & prefetch
    # ------------------------------------------------------------------

    def _encode_current_image(self, reason: str = "") -> bool:
        """Synchronous encode dispatch. Use ``_schedule_encode`` for debounced version."""
        if self.image_rgb is None or not self.sam.is_ready:
            return False
        if not self.image_paths:
            return False
        path = self.image_paths[self.index]
        rgb, crop_info = self._build_encode_input(self.image_rgb)
        key = self._encode_key_for(path)
        cache_hit = self.sam.has_cached(key)
        queued = self.sam.encode(rgb, key, crop_info)
        if queued:
            prefix = f"{reason}：" if reason else ""
            if cache_hit:
                self._log(f"{prefix}命中 SAM 缓存。", "info")
            else:
                self._log(f"{prefix}已排队 SAM 编码。", "info")
        return queued

    def _schedule_encode(self, reason: str = "", delay_ms: int = 350):
        """Debounced encode: wait until the user settles on this image."""
        if not self.sam.is_ready or self.image_rgb is None or not self.image_paths:
            self._encode_debounce.stop()
            return
        path = self.image_paths[self.index]
        key = self._encode_key_for(path)
        if self.sam.has_cached(key):
            self._encode_debounce.stop()
            self._encode_current_image(reason or "缓存恢复")
            return
        self._encode_reason = reason
        self._encode_debounce.start(max(0, int(delay_ms)))
        if 0 <= self.index < len(self.image_paths):
            self.lbl_filename.setText(
                f"{os.path.basename(self.image_paths[self.index])}  · 等待停留后编码…"
            )

    def _encode_current_image_debounced(self):
        if self.image_rgb is None or not self.image_paths:
            return
        self._encode_current_image(self._encode_reason or "切换图像")
        self._encode_reason = ""

    def _schedule_prefetch(self, delay_ms: int = 900):
        """Debounced prefetch of neighbor embeddings."""
        if not self.sam.is_ready or not self.image_paths:
            self._prefetch_debounce.stop()
            return
        if self.roi_mode == "polygon" and self.sidebar.chk_roi_crop.isChecked():
            self._prefetch_debounce.stop()
            return
        self._prefetch_debounce.start(max(0, int(delay_ms)))

    def _encode_key_for(self, image_path: str) -> str:
        """Return the cache key for ``image_path``, folding ROI bbox if active."""
        ap = os.path.abspath(image_path)
        if (
            self.roi_mode == "polygon"
            and self.sidebar.chk_roi_crop.isChecked()
            and self.roi_mask is not None
        ):
            box = self._roi_bbox()
            if box is not None:
                return f"{ap}|crop:{box[0]},{box[1]},{box[2]},{box[3]}"
        return ap

    def _build_encode_input(self, rgb_full):
        """Return ``(rgb_to_encode, crop_info)`` honoring ROI crop preference."""
        if (
            self.roi_mode == "polygon"
            and self.sidebar.chk_roi_crop.isChecked()
            and self.roi_mask is not None
        ):
            box = self._roi_bbox()
            if box is not None:
                x1, y1, x2, y2 = box
                crop = np.ascontiguousarray(rgb_full[y1:y2, x1:x2])
                if crop.size > 0:
                    h_full, w_full = rgb_full.shape[:2]
                    return crop, {
                        "x": x1,
                        "y": y1,
                        "h_full": h_full,
                        "w_full": w_full,
                        "crop_h": y2 - y1,
                        "crop_w": x2 - x1,
                    }
        return rgb_full, None

    def _roi_bbox(self) -> tuple[int, int, int, int] | None:
        if self.roi_mask is None:
            return None
        ys, xs = np.where(self.roi_mask > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        h, w = self.roi_mask.shape[:2]
        pad = 8
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(w, int(xs.max()) + 1 + pad)
        y2 = min(h, int(ys.max()) + 1 + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return x1, y1, x2, y2

    def _prefetch_neighbors(self):
        """Warm SAM embeddings for neighboring images in the background."""
        if os.environ.get("SAM_PREFETCH", "1") == "0":
            return
        if not self.sam.is_ready or not self.image_paths:
            return
        if self.roi_mode == "polygon" and self.sidebar.chk_roi_crop.isChecked():
            return
        radius = 2
        n = len(self.image_paths)
        anchor = self.index
        for offset in (1, -1, 2, -2)[: 2 * radius]:
            j = anchor + offset
            if 0 <= j < n and j != anchor:
                path = self.image_paths[j]
                self.sam.prefetch(self._encode_key_for(path), path)

    # ------------------------------------------------------------------
    # SAM prediction
    # ------------------------------------------------------------------

    def _sam_predict(self, x: int, y: int):
        if not self.sam.is_ready:
            return
        if self.image_bgr is None:
            return
        self._sam_result_class_id = None
        self._sam_result_replace_box = None
        h, w = self.image_shape
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        # ROI check
        if self.roi_mode == "polygon" and self.roi_mask is not None:
            if self.roi_mask[y, x] == 0:
                if self.sidebar.chk_roi_auto.isChecked():
                    self._roi_reset()
                else:
                    self._log("请在 ROI 多边形内点击。", "warn")
                    return
        # Lazy encode
        if not self.sam.is_encoded:
            self._encode_debounce.stop()
            if self._encode_current_image("首次分割"):
                self._log("编码完成后会自动执行本次点击分割。", "info")
            self.sam.predict_async(x, y)
            return
        self.sam.predict_async(x, y)

    def _on_sam_prediction(self, mask_2d, gen):
        h, w = self.image_shape
        if mask_2d.shape != (h, w):
            self._log("SAM 返回尺寸与当前图像不一致，已忽略旧结果。", "warn")
            self._sam_result_class_id = None
            self._sam_result_replace_box = None
            return
        # ROI intersection
        if self.roi_mode == "polygon" and self.roi_mask is not None:
            mask_2d = (mask_2d & (self.roi_mask // 255)).astype(np.uint8)
        if int(mask_2d.sum()) < 30:
            self._sam_result_class_id = None
            self._sam_result_replace_box = None
            return
        class_id = self._sam_result_class_id
        replace_box = self._sam_result_replace_box
        self._sam_result_class_id = None
        self._sam_result_replace_box = None
        if class_id is None:
            class_id = self.current_class_id
        self.classes.ensure(class_id)
        if replace_box is not None:
            idx, snapshot = replace_box
            if self.store.replace_box_with_mask(idx, snapshot, mask_2d, class_id):
                self.hover_kind = ""
                self.hover_idx = -1
                self._log("已由检测框生成 SAM mask，并删除原框。", "ok")
            else:
                self._log("原检测框已变化，已忽略框转 mask 结果。", "warn")
            return
        self.store.add_mask(mask_2d, class_id)

    # ------------------------------------------------------------------
    # Annotation conversion (mask <-> box)
    # ------------------------------------------------------------------

    def _selected_mask_index(self) -> int:
        if self.hover_kind == "mask" and 0 <= self.hover_idx < len(self.store.masks):
            return self.hover_idx
        return len(self.store.masks) - 1

    def _selected_box_index(self) -> int:
        if self.hover_kind == "box" and 0 <= self.hover_idx < len(self.store.boxes):
            return self.hover_idx
        return len(self.store.boxes) - 1

    def _convert_hovered_annotation(self):
        if self.hover_kind == "mask" and 0 <= self.hover_idx < len(self.store.masks):
            self._mask_to_box()
            return
        if self.hover_kind == "box" and 0 <= self.hover_idx < len(self.store.boxes):
            self._box_to_sam_mask()
            return
        if self.store.last_kind == "mask" and self.store.masks:
            self._mask_to_box()
            return
        if self.store.last_kind == "box" and self.store.boxes:
            self._box_to_sam_mask()
            return
        if self.store.masks and not self.store.boxes:
            self._mask_to_box()
            return
        if self.store.boxes and not self.store.masks:
            self._box_to_sam_mask()
            return
        if self.store.masks and self.store.boxes:
            self._log("请把鼠标移到要转换的 mask 或检测框上再按 T。", "warn")
            return
        self._log("当前图没有可转换的 mask 或检测框。", "warn")

    def _mask_to_box(self):
        idx = self._selected_mask_index()
        if idx < 0:
            self._log("当前图没有可转换的 mask。", "warn")
            return
        mask = self.store.masks[idx]
        ys, xs = np.where(mask.data > 0)
        if len(xs) == 0 or len(ys) == 0:
            self._log("当前 mask 为空，无法生成框。", "warn")
            return
        if self.store.replace_mask_with_box(
            idx, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        ):
            self.hover_kind = ""
            self.hover_idx = -1
            self._log("已由 mask 生成检测框，并删除原 mask。", "ok")

    def _box_to_sam_mask(self):
        idx = self._selected_box_index()
        if idx < 0:
            self._log("当前图没有可转换的检测框。", "warn")
            return
        if not self.sam.is_ready:
            self._log("请先加载 SAM 模型。", "warn")
            return
        box = self.store.boxes[idx]
        self._sam_result_class_id = box.class_id
        self._sam_result_replace_box = (
            idx,
            (box.class_id, box.x1, box.y1, box.x2, box.y2),
        )
        if not self.sam.is_encoded:
            self._encode_debounce.stop()
            if self._encode_current_image("框转 Mask"):
                self._log("编码完成后会自动执行框转 mask。", "info")
            self.sam.predict_box_async(box.x1, box.y1, box.x2, box.y2)
            return
        if self.sam.predict_box_async(box.x1, box.y1, box.x2, box.y2):
            self._log("已提交框转 SAM mask。", "info")
        else:
            self._sam_result_class_id = None
            self._sam_result_replace_box = None
            self._log("SAM 正忙，暂不能框转 mask。", "warn")

    # ------------------------------------------------------------------
    # ROI state machine
    # ------------------------------------------------------------------

    def _roi_start_draw(self):
        self.roi_mode = "drawing"
        self.roi_pts.clear()
        self.roi_mask = None
        self._log("ROI: 请左键点击添加多边形顶点，右键撤销顶点。", "info")

    def _roi_close(self):
        if len(self.roi_pts) < 3:
            self._log("ROI: 至少需要 3 个顶点。", "warn")
            return
        h, w = self.image_shape
        arr = np.array(self.roi_pts, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [arr], 255)
        self.roi_mask = mask
        self.roi_mode = "polygon"
        self._log("ROI 已闭合。", "ok")
        self._refresh_canvas()
        if self.sidebar.chk_roi_crop.isChecked() and self.sam.is_ready:
            self._encode_debounce.stop()
            self._encode_current_image("ROI 闭合")

    def _roi_pop(self):
        if self.roi_pts:
            self.roi_pts.pop()
            self._refresh_canvas()

    def _roi_reset(self):
        was_cropped = (
            self.roi_mode == "polygon"
            and self.sidebar.chk_roi_crop.isChecked()
        )
        self.roi_mode = "full"
        self.roi_pts.clear()
        h, w = self.image_shape
        self.roi_mask = np.ones((h, w), dtype=np.uint8) * 255
        self._log("ROI: 已恢复全图模式。", "info")
        self._refresh_canvas()
        if was_cropped and self.sam.is_ready:
            self._encode_debounce.stop()
            self._encode_current_image("恢复全图")
