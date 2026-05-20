"""MainWindow — thin controller that wires model, view, and service layers."""

import json
import os
from typing import Optional

import cv2
import numpy as np
import torch
from PyQt5.QtCore import Qt, QPoint, QTimer, QSettings
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFrame, QLabel, QAction,
    QFileDialog, QMessageBox, QTextEdit,
)

from .models import AnnotationStore, ClassRegistry, DEFAULT_CLASS_NAMES
from .io_utils import (
    scan_images, load_image_bgr,
    save_labels, load_labels_for_image,
    load_class_names, save_class_names,
)
from .canvas import (
    ImageCanvas, CoordTransformer, DrawState,
    render_composite, composite_to_pixmap,
)
from .sidebar import Sidebar
from .right_panel import RightPanel
from .sam_service import SamService, SAM_MODEL_FILES
from .yolo_service import YoloService
from .app_sam import SamControllerMixin
from .app_input import InputHandlerMixin


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(SamControllerMixin, InputHandlerMixin, QMainWindow):
    """Top-level window.  Thin — delegates to sub-modules and mixins."""

    def __init__(self, image_dir: str = "", label_dir: str = "",
                 sam_checkpoint: str = "", model_type: str = "vit_h",
                 yolo_weights: str = ""):
        super().__init__()
        self.setWindowTitle("YOLO SAM Labeler")
        self.resize(1480, 920)
        self.settings = QSettings("yolo-sam-labeler", "yolo-sam-labeler")

        stored_image_dir = self._setting_str("paths/image_dir")
        stored_label_dir = self._setting_str("paths/label_dir")
        stored_sam_checkpoint = self._setting_str("paths/sam_checkpoint")
        image_dir_arg = image_dir
        label_dir_arg = label_dir
        image_dir = image_dir_arg or stored_image_dir
        if label_dir_arg:
            label_dir = label_dir_arg
        elif image_dir_arg:
            label_dir = self._auto_label_dir_for(image_dir_arg)
        else:
            label_dir = stored_label_dir
        sam_checkpoint = sam_checkpoint or stored_sam_checkpoint
        if stored_image_dir and image_dir and not os.path.isdir(image_dir) and os.path.isdir(stored_image_dir):
            image_dir = stored_image_dir
        if (
            stored_sam_checkpoint
            and sam_checkpoint
            and not os.path.isfile(sam_checkpoint)
            and os.path.isfile(stored_sam_checkpoint)
        ):
            sam_checkpoint = stored_sam_checkpoint
        model_type = model_type or self._setting_str("model/sam_type", "vit_h")
        yolo_weights = yolo_weights or self._setting_str("paths/yolo_weights")
        if image_dir and not label_dir:
            label_dir = self._auto_label_dir_for(image_dir)

        # --- data model ---
        self.image_dir = image_dir or ""
        self.classes = ClassRegistry(self._load_startup_classes(image_dir, label_dir))
        self.store = AnnotationStore(self.classes, label_dir)

        # --- services ---
        self.sam = SamService(self)
        self.yolo = YoloService(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_weights_path = yolo_weights or ""

        # --- view state ---
        self.image_paths: list[str] = []
        self.index: int = 0
        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_shape: tuple = (0, 0)
        self.coords: Optional[CoordTransformer] = None
        self.draw_state = DrawState()
        self.hover_kind: str = ""
        self.hover_idx: int = -1
        ids = self.classes.sorted_ids()
        self.current_class_id: int = ids[0] if ids else 0
        self._loading_image = False
        self._autosave_warned = False
        self._sam_result_class_id: int | None = None
        self._sam_result_replace_box: tuple[int, tuple[int, int, int, int, int]] | None = None

        # pan
        self._panning = False
        self._pan_start: Optional[QPoint] = None
        self._pan_view_start: tuple[float, float] = (0.0, 0.0)

        # ROI
        self.roi_mode: str = "full"
        self.roi_pts: list[tuple[int, int]] = []
        self.roi_mask: Optional[np.ndarray] = None

        # SAM state
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type

        # SAM dispatch debounce
        self._encode_debounce = QTimer(self)
        self._encode_debounce.setSingleShot(True)
        self._encode_debounce.timeout.connect(self._encode_current_image_debounced)
        self._prefetch_debounce = QTimer(self)
        self._prefetch_debounce.setSingleShot(True)
        self._prefetch_debounce.timeout.connect(self._prefetch_neighbors)
        self._encode_reason: str = ""

        # --- build UI ---
        self._build_ui()
        self._connect_signals()
        self._sync_sidebar_state()
        self._refresh_class_list()
        self._set_current_class(self.current_class_id)

        # --- load data ---
        if image_dir:
            self._load_directory(image_dir)
        if yolo_weights:
            self.yolo.load(yolo_weights)
        QTimer.singleShot(0, self._auto_load_sam)

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- menu bar ---
        mb = self.menuBar()
        file_menu = mb.addMenu("文件")
        act_open = QAction("打开图片目录…", self)
        act_open.triggered.connect(self._pick_image_dir)
        file_menu.addAction(act_open)
        act_label_dir = QAction("选择标签目录…", self)
        act_label_dir.triggered.connect(self._pick_label_dir)
        file_menu.addAction(act_label_dir)
        file_menu.addSeparator()
        act_save = QAction("保存当前图", self)
        act_save.setShortcuts([QKeySequence("S"), QKeySequence.Save])
        act_save.triggered.connect(self._save_current)
        file_menu.addAction(act_save)
        act_save_next = QAction("保存并下一张", self)
        act_save_next.setShortcuts([QKeySequence("N"), QKeySequence("Space")])
        act_save_next.triggered.connect(self._save_and_next)
        file_menu.addAction(act_save_next)
        act_save_prev = QAction("保存并上一张", self)
        act_save_prev.setShortcut(QKeySequence("P"))
        act_save_prev.triggered.connect(self._save_and_prev)
        file_menu.addAction(act_save_prev)
        file_menu.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.setShortcuts([QKeySequence("Q"), QKeySequence("E"), QKeySequence("Esc")])
        act_quit.triggered.connect(self._save_and_close)
        file_menu.addAction(act_quit)

        nav_menu = mb.addMenu("导航")
        act_skip = QAction("跳过当前图", self)
        act_skip.setShortcut(QKeySequence("D"))
        act_skip.triggered.connect(self._skip)
        nav_menu.addAction(act_skip)
        act_undo = QAction("撤销", self)
        act_undo.setShortcuts([QKeySequence("U"), QKeySequence.Undo])
        act_undo.triggered.connect(self._undo)
        nav_menu.addAction(act_undo)
        act_delete = QAction("删除悬停标注", self)
        act_delete.setShortcuts([QKeySequence("Del"), QKeySequence("Backspace")])
        act_delete.triggered.connect(self._delete_hovered)
        nav_menu.addAction(act_delete)
        act_clear = QAction("清空当前标注", self)
        act_clear.setShortcut(QKeySequence("C"))
        act_clear.triggered.connect(self._clear)
        nav_menu.addAction(act_clear)
        act_reset_zoom = QAction("重置缩放", self)
        act_reset_zoom.setShortcut(QKeySequence("R"))
        act_reset_zoom.triggered.connect(self._reset_zoom)
        nav_menu.addAction(act_reset_zoom)

        tool_menu = mb.addMenu("工具")
        act_convert = QAction("Mask/检测框互转", self)
        act_convert.setShortcut(QKeySequence("T"))
        act_convert.triggered.connect(self._convert_hovered_annotation)
        tool_menu.addAction(act_convert)

        model_menu = mb.addMenu("模型")
        act_sam_ckpt = QAction("选择 SAM 权重…", self)
        act_sam_ckpt.triggered.connect(self._pick_sam_ckpt)
        model_menu.addAction(act_sam_ckpt)
        act_load_sam = QAction("加载 SAM", self)
        act_load_sam.triggered.connect(self._load_sam)
        model_menu.addAction(act_load_sam)
        model_menu.addSeparator()
        act_yolo_w = QAction("选择 YOLO 权重…", self)
        act_yolo_w.triggered.connect(self._pick_yolo_weights)
        model_menu.addAction(act_yolo_w)

        # --- info bar ---
        info = QFrame()
        info.setObjectName("InfoBar")
        info_lay = QHBoxLayout(info)
        info_lay.setContentsMargins(8, 2, 8, 2)
        self.lbl_device = QLabel(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        info_lay.addWidget(self.lbl_device)
        self.lbl_progress = QLabel("0 / 0")
        info_lay.addWidget(self.lbl_progress)
        self.lbl_filename = QLabel("")
        info_lay.addWidget(self.lbl_filename, stretch=1)
        root.addWidget(info)

        # --- workbench (three-panel splitter) ---
        splitter = QSplitter(Qt.Horizontal)

        # Left sidebar
        self.sidebar = Sidebar()
        self.sidebar.setMinimumWidth(160)
        self.sidebar.setMaximumWidth(380)
        splitter.addWidget(self.sidebar)

        # Center canvas
        center_wrap = QWidget()
        center_lay = QVBoxLayout(center_wrap)
        center_lay.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas()
        center_lay.addWidget(self.canvas)
        splitter.addWidget(center_wrap)

        # Right panel
        self.rpanel = RightPanel()
        self.rpanel.setMinimumWidth(140)
        self.rpanel.setMaximumWidth(320)
        splitter.addWidget(self.rpanel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 900, 220])
        root.addWidget(splitter, stretch=1)

        # --- log panel ---
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(120)
        self.log_panel.setPlaceholderText("日志…")
        root.addWidget(self.log_panel)

    # ==================================================================
    # Signal wiring
    # ==================================================================

    def _connect_signals(self):
        # Canvas callbacks
        self.canvas.on_wheel = self._on_wheel
        self.canvas.on_mouse_press = self._on_mouse_press
        self.canvas.on_mouse_move = self._on_mouse_move
        self.canvas.on_mouse_release = self._on_mouse_release
        self.canvas.on_key_press = self._on_key_press
        self.canvas.on_resize = self._on_canvas_resize

        # Annotation store
        self.store.changed.connect(self._on_store_changed)
        self.classes.classes_changed.connect(self._on_classes_changed)

        # Sidebar signals
        self.sidebar.load_sam_requested.connect(self._on_load_sam)
        self.sidebar.yolo_predict_requested.connect(self._on_yolo_predict)
        self.sidebar.roi_draw_requested.connect(self._roi_start_draw)
        self.sidebar.roi_close_requested.connect(self._roi_close)
        self.sidebar.roi_pop_requested.connect(self._roi_pop)
        self.sidebar.roi_full_requested.connect(self._roi_reset)

        # Right panel signals
        self.rpanel.class_selected.connect(self._on_class_selected)
        self.rpanel.class_add_requested.connect(self._on_class_add)
        self.rpanel.class_delete_requested.connect(self._on_class_delete)
        self.rpanel.class_rename_requested.connect(self._on_class_rename)
        self.rpanel.save_current.connect(self._save_current)
        self.rpanel.save_and_next.connect(self._save_and_next)
        self.rpanel.save_and_prev.connect(self._save_and_prev)
        self.rpanel.skip.connect(self._skip)
        self.rpanel.undo.connect(self._undo)
        self.rpanel.clear.connect(self._clear)
        self.rpanel.convert_annotation.connect(self._convert_hovered_annotation)

        # SAM signals
        self.sam.model_ready.connect(self._on_sam_ready)
        self.sam.load_failed.connect(self._on_sam_error)
        self.sam.prediction_ready.connect(self._on_sam_prediction)
        self.sam.encode_started.connect(self._on_sam_encode_started)
        self.sam.encode_done.connect(self._on_sam_encode_done)
        self.sam.prefetch_done.connect(self._on_sam_prefetch_done)
        self.sam.error.connect(lambda msg: self._log(msg, "err"))

        # YOLO signals
        self.yolo.predict_done.connect(self._on_yolo_predict_done)
        self.yolo.busy_changed.connect(self._on_yolo_busy_changed)
        self.yolo.load_done.connect(lambda p: self._log(f"YOLO 已加载: {os.path.basename(p)}", "ok"))
        self.yolo.error.connect(lambda msg: self._log(msg, "err"))

        # Timer for refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_canvas)

    # ==================================================================
    # Settings & class files
    # ==================================================================

    def _setting_str(self, key: str, default: str = "") -> str:
        value = self.settings.value(key, default)
        return default if value is None else str(value)

    def _class_file_candidates(self, image_dir: str = "", label_dir: str = "") -> list[str]:
        candidates: list[str] = []
        image_dir = image_dir or getattr(self, "image_dir", "")
        if not label_dir and hasattr(self, "store"):
            label_dir = self.store.label_dir
        if image_dir:
            candidates.append(os.path.join(image_dir, "classes.txt"))
        if label_dir:
            candidates.append(os.path.join(label_dir, "classes.txt"))
        deduped: list[str] = []
        seen: set[str] = set()
        for path in candidates:
            ap = os.path.abspath(path)
            if ap not in seen:
                seen.add(ap)
                deduped.append(path)
        return deduped

    def _load_classes_from_settings(self) -> dict[int, str]:
        raw = self._setting_str("classes/json")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[int, str] = {}
        for key, value in data.items():
            try:
                cid = int(key)
            except (TypeError, ValueError):
                continue
            name = str(value).strip()
            if name:
                out[cid] = name
        return out

    def _load_startup_classes(self, image_dir: str, label_dir: str) -> dict[int, str]:
        for path in self._class_file_candidates(image_dir, label_dir):
            classes = load_class_names(path)
            if classes:
                return classes
        return self._load_classes_from_settings() or dict(DEFAULT_CLASS_NAMES)

    def _load_classes_for_current_dirs(self):
        for path in self._class_file_candidates():
            classes = load_class_names(path)
            if classes:
                self.classes.set_names(classes)
                self._log(f"已载入类别: {path}", "ok")
                return
        if len(self.classes) == 0:
            self.classes.set_names(dict(DEFAULT_CLASS_NAMES))
            self._log("未找到 classes.txt，请在右侧面板添加类别。", "warn")

    def _persist_classes(self):
        data = self.classes.to_names()
        self.settings.setValue(
            "classes/json",
            json.dumps({str(k): v for k, v in data.items()}, ensure_ascii=False),
        )
        targets = self._class_file_candidates()
        if not targets:
            return
        try:
            save_class_names(targets[0], data)
        except OSError as exc:
            self._log(f"类别文件保存失败: {exc}", "warn")

    def _sync_sidebar_state(self):
        self.sidebar.set_checkpoint_label(self.sam_checkpoint)
        self.sidebar.set_model_type(self.model_type)
        self.sidebar.set_yolo_weights_label(self.yolo_weights_path)

    def _remember_paths(self):
        self.settings.setValue("paths/image_dir", self.image_dir)
        self.settings.setValue("paths/label_dir", self.store.label_dir)
        self.settings.setValue("paths/sam_checkpoint", self.sam_checkpoint)
        self.settings.setValue("paths/yolo_weights", self.yolo_weights_path)
        self.settings.setValue("model/sam_type", self.model_type)

    @staticmethod
    def _same_path(a: str, b: str) -> bool:
        if not a or not b:
            return False
        return os.path.abspath(a) == os.path.abspath(b)

    @staticmethod
    def _is_subpath(path: str, parent: str) -> bool:
        if not path or not parent:
            return False
        try:
            return (
                os.path.commonpath([os.path.abspath(path), os.path.abspath(parent)])
                == os.path.abspath(parent)
            )
        except ValueError:
            return False

    @staticmethod
    def _auto_label_dir_for(image_dir: str) -> str:
        return os.path.join(image_dir, "labels") if image_dir else ""

    @staticmethod
    def _fallback_sam_checkpoint(model_type: str) -> str:
        filename = SAM_MODEL_FILES.get(model_type, "sam_vit_h_4b8939.pth")
        candidates = [
            os.path.abspath(filename),
            os.path.join(os.getcwd(), filename),
            os.path.join(os.path.expanduser("~"), "yolo_seg_label_sam", filename),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]

    # ==================================================================
    # Image directory & navigation
    # ==================================================================

    def _load_directory(self, path: str):
        old_image_dir = self.image_dir
        old_auto_label_dir = self._auto_label_dir_for(old_image_dir)
        self.image_dir = path
        switching_dirs = bool(old_image_dir) and not self._same_path(old_image_dir, path)
        label_was_tied_to_old_dir = switching_dirs and self._is_subpath(self.store.label_dir, old_image_dir)
        if (
            not self.store.label_dir
            or self._same_path(self.store.label_dir, old_auto_label_dir)
            or label_was_tied_to_old_dir
        ):
            self.store.label_dir = self._auto_label_dir_for(path)
        self._load_classes_for_current_dirs()
        self.image_paths = scan_images(path)
        self.index = 0
        self._remember_paths()
        if self.image_paths:
            self._load_current_image()
        else:
            self.image_bgr = None
            self.image_rgb = None
            self._encode_debounce.stop()
            self._prefetch_debounce.stop()
            self.sam.invalidate_image()
            self._log(f"当前目录未找到图像: {path}", "warn")
        self._update_header()

    def _load_current_image(self):
        if not self.image_paths or self.index < 0:
            return
        if self.index >= len(self.image_paths):
            self.index = len(self.image_paths) - 1
        path = self.image_paths[self.index]
        bgr = load_image_bgr(path)
        if bgr is None:
            self._log(f"无法加载图像: {path}", "err")
            return
        self._encode_debounce.stop()
        self._prefetch_debounce.stop()
        self.sam.invalidate_image()
        self.image_bgr = bgr
        self.image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        self.image_shape = (h, w)
        self.store.image_height = h
        self.store.image_width = w

        # Reset viewport
        cw, ch = self.canvas.width(), self.canvas.height()
        if cw <= 0:
            cw = 800
        if ch <= 0:
            ch = 600
        self.coords = CoordTransformer(cw, ch, w, h)
        self.draw_state = DrawState()
        self.hover_kind = ""
        self.hover_idx = -1
        self._sam_result_class_id = None
        self._sam_result_replace_box = None

        # Load annotations from disk without treating it as an edit.
        self._loading_image = True
        try:
            load_labels_for_image(self.store, path, w, h)
        finally:
            self._loading_image = False

        # SAM encode (if model ready and not lazy)
        if self.sam.is_ready and not self.sidebar.chk_lazy.isChecked():
            self._schedule_encode("切换图像")

        self._schedule_prefetch()
        self._refresh_canvas()
        QTimer.singleShot(0, self._refresh_canvas)
        self._update_header()
        self._log(f"已载入: {os.path.basename(path)} ({w}×{h})", "ok")

    # ==================================================================
    # Save & navigation actions
    # ==================================================================

    def _save_current(self, silent: bool = False) -> bool:
        if self.image_bgr is None:
            return False
        if not self.store.label_dir:
            if silent:
                if not self._autosave_warned:
                    self._log("自动保存失败：请先选择标签保存目录。", "warn")
                    self._autosave_warned = True
            else:
                QMessageBox.warning(self, "无法保存", "请先选择标签保存目录。")
            return False
        path = self.image_paths[self.index]
        stem = os.path.splitext(os.path.basename(path))[0]
        h, w = self.image_shape
        self._persist_classes()
        save_labels(self.store, stem, w, h)
        self._autosave_warned = False
        if not silent:
            self._log(f"已保存: {stem}.txt", "ok")
        return True

    def _autosave_current(self):
        self._save_current(silent=True)

    def _save_and_next(self):
        if self._save_current():
            self._next_image()

    def _save_and_prev(self):
        if self._save_current():
            self._prev()

    def _next_image(self):
        if not self.image_paths:
            return
        old = self.index
        self.index = min(self.index + 1, len(self.image_paths) - 1)
        if self.index == old:
            self._log("已是最后一张。", "warn")
        self._load_current_image()

    def _skip(self):
        self._next_image()

    def _prev(self):
        if not self.image_paths:
            return
        old = self.index
        self.index = max(0, self.index - 1)
        if self.index == old:
            self._log("已是第一张。", "warn")
        self._load_current_image()

    def _clear(self):
        self.store.clear()

    def _undo(self):
        self.store.undo_last()

    def _delete_hovered(self):
        if self.hover_kind == "box" and 0 <= self.hover_idx < len(self.store.boxes):
            del self.store.boxes[self.hover_idx]
        elif self.hover_kind == "mask" and 0 <= self.hover_idx < len(self.store.masks):
            del self.store.masks[self.hover_idx]
        else:
            self._log("没有可删除的悬停标注。", "warn")
            return
        self.store._refresh_last_kind()
        self.hover_kind = ""
        self.hover_idx = -1
        self.store.changed.emit()

    def _reset_zoom(self):
        if self.image_bgr is None:
            return
        h, w = self.image_shape
        cw, ch = self.canvas.width(), self.canvas.height()
        if cw <= 0:
            cw = 800
        if ch <= 0:
            ch = 600
        self.coords = CoordTransformer(cw, ch, w, h)
        self._refresh_canvas()

    def _save_and_close(self):
        if self.image_bgr is None or self._save_current():
            self.close()

    # ==================================================================
    # Class management
    # ==================================================================

    def _on_classes_changed(self):
        ids = self.classes.sorted_ids()
        if not ids:
            self.current_class_id = 0
        elif self.current_class_id not in self.classes:
            self.current_class_id = ids[0]
        self._refresh_class_list()
        if ids:
            self.rpanel.select_class(self.current_class_id)
        self._persist_classes()
        self._refresh_canvas()

    def _set_current_class(self, cid: int):
        if cid in self.classes:
            self.current_class_id = cid
            self.rpanel.select_class(cid)
            self._log(f"当前类别: {cid} — {self.classes.name(cid)}", "info")

    def _apply_class_key(self, cid: int):
        if cid not in self.classes:
            return
        # Hover relabel
        if self.hover_kind == "box" and self.hover_idx >= 0:
            if self.store.relabel("box", self.hover_idx, cid):
                self._log(f"改框类别: → {self.classes.name(cid)}", "ok")
                self._refresh_canvas()
            return
        if self.hover_kind == "mask" and self.hover_idx >= 0:
            if self.store.relabel("mask", self.hover_idx, cid):
                self._log(f"改 mask 类别: → {self.classes.name(cid)}", "ok")
                self._refresh_canvas()
            return
        # No hover → switch current class
        self._set_current_class(cid)

    def _on_class_selected(self, cid: int):
        if cid >= 0 and cid in self.classes:
            self.current_class_id = cid

    def _on_class_add(self, name: str):
        new_id = self.classes.add(name)
        self.current_class_id = new_id
        self.rpanel.select_class(new_id)

    def _on_class_delete(self, cid: int):
        if cid in self.classes:
            if len(self.classes) <= 1:
                self._log("至少保留一个类别。", "warn")
                return
            # Determine which class to select after deletion:
            # prefer the next one in sorted order; if deleting the last, pick the new last.
            ids = self.classes.sorted_ids()
            idx = ids.index(cid) if cid in ids else 0
            # Compute the future id list (without cid)
            future_ids = [i for i in ids if i != cid]
            if future_ids:
                next_idx = min(idx, len(future_ids) - 1)
                self.current_class_id = future_ids[next_idx]
            # Now remove — classes_changed signal will use the updated current_class_id
            self.classes.remove(cid)

    def _on_class_rename(self, cid: int, name: str):
        if cid in self.classes and self.classes.rename(cid, name):
            self._log(f"类别已重命名: {cid} — {name}", "ok")

    def _refresh_class_list(self):
        self.rpanel.set_classes(self.classes.to_names())

    # ==================================================================
    # YOLO interaction
    # ==================================================================

    def _on_yolo_predict(self, conf: float, replace: bool):
        if self.image_bgr is None:
            return
        self.yolo.predict(self.image_bgr, conf, replace)

    def _on_yolo_predict_done(self, payload):
        self.classes.ensure_ids(payload.mask_class_ids)
        self.classes.ensure_ids(payload.box_class_ids)
        self.store.apply_yolo_predictions(
            masks=payload.masks,
            mask_class_ids=payload.mask_class_ids,
            boxes=payload.boxes,
            box_class_ids=payload.box_class_ids,
            replace=payload.replace,
        )
        n_mask = len(payload.masks)
        n_box = len(payload.boxes)
        if n_mask and n_box:
            self._log(f"YOLO 预测完成: {n_mask} 个 mask + {n_box} 个检测框", "ok")
        elif n_mask:
            self._log(f"YOLO 预测完成: {n_mask} 个 mask", "ok")
        elif n_box:
            self._log(f"YOLO 预测完成: {n_box} 个检测框", "ok")
        else:
            self._log("YOLO 没有检测到目标。", "warn")

    def _on_yolo_busy_changed(self, busy: bool):
        if hasattr(self.sidebar, "btn_yolo"):
            self.sidebar.btn_yolo.setEnabled(not busy)

    # ==================================================================
    # Refresh & helpers
    # ==================================================================

    def _on_store_changed(self):
        self._refresh_canvas()
        if not self._loading_image:
            self._autosave_current()

    def _on_canvas_resize(self):
        if hasattr(self, "_refresh_timer"):
            self._refresh_timer.start(0)
        else:
            self._refresh_canvas()

    def _refresh_canvas(self):
        if self.image_bgr is None or self.coords is None:
            return
        vis = render_composite(
            self.image_bgr, self.store,
            hover_kind=self.hover_kind, hover_idx=self.hover_idx,
            draw_state=self.draw_state,
            roi_pts=self.roi_pts if self.roi_mode == "drawing" else None,
            roi_mask=self.roi_mask if self.roi_mode == "polygon" else None,
            roi_mode=self.roi_mode,
        )
        cw, ch = self.canvas.width(), self.canvas.height()
        if cw <= 0:
            cw = 800
        if ch <= 0:
            ch = 600
        self.coords.update_canvas_size(cw, ch)
        pix = composite_to_pixmap(vis, self.coords, cw, ch)
        self.canvas.setPixmap(pix)

    def _update_header(self):
        total = len(self.image_paths)
        self.lbl_progress.setText(f"{self.index + 1} / {total}" if total else "0 / 0")
        if self.image_paths and self.index < total:
            self.lbl_filename.setText(os.path.basename(self.image_paths[self.index]))

    def _log(self, msg: str, level: str = "info"):
        prefixes = {"ok": "  ✓ ", "err": "  ✗ ", "warn": "  ⚠ ", "info": "  → "}
        prefix = prefixes.get(level, "  → ")
        self.log_panel.append(prefix + msg)
        if self.log_panel.document().blockCount() > 400:
            self.log_panel.clear()

    # ==================================================================
    # File pickers
    # ==================================================================

    def _pick_image_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片目录", self.image_dir or ".")
        if d:
            self._load_directory(d)

    def _pick_label_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择标签保存目录", self.store.label_dir or ".")
        if d:
            self.store.label_dir = d
            self._load_classes_for_current_dirs()
            self._remember_paths()
            self._log(f"标签目录: {d}", "ok")
            if self.image_bgr is not None:
                self._load_current_image()

    def _pick_sam_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 SAM 权重", os.path.dirname(self.sam_checkpoint) or ".",
            "PyTorch (*.pth);;All (*.*)"
        )
        if path:
            self.sam_checkpoint = path
            self.sidebar.set_checkpoint_label(path)
            self._remember_paths()
            self._log(f"SAM 权重: {path}", "ok")

    def _pick_yolo_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重", os.path.dirname(self.yolo_weights_path) or ".",
            "PyTorch (*.pt);;All (*.*)"
        )
        if path:
            self.yolo_weights_path = path
            self.yolo.load(path)
            self.sidebar.set_yolo_weights_label(path)
            self._remember_paths()
            self._log(f"YOLO 权重: {path}", "ok")

    # ==================================================================
    # Window management
    # ==================================================================

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_canvas()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._refresh_canvas)

    def closeEvent(self, event):
        self._remember_paths()
        self._persist_classes()
        self.sam.shutdown()
        if hasattr(self.yolo, "shutdown"):
            self.yolo.shutdown()
        super().closeEvent(event)
