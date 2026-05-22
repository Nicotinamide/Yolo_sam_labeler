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
    scan_images, discover_image_dir, load_image_bgr,
    save_labels, load_labels_for_image,
    load_class_names, save_class_names,
    inspect_label_dir_format,
    split_mixed_label_dir,
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
from .weight_manager import open_weight_manager


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
        stored_legacy_label_dir = self._setting_str("paths/label_dir")
        stored_seg_dir = self._setting_str("paths/seg_dir")
        stored_detect_dir = self._setting_str("paths/detect_dir")
        stored_sam_checkpoint = self._setting_str("paths/sam_checkpoint")
        image_dir_arg = image_dir
        label_dir_arg = label_dir  # CLI / constructor only
        image_dir = image_dir_arg or stored_image_dir
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

        # --- data model ---
        self.image_dir = image_dir or ""
        # Resolve a hint for class-file lookup. Real seg/detect dirs are set
        # by _seed_label_dirs below.
        startup_label_hint = (
            label_dir_arg
            or stored_seg_dir
            or stored_detect_dir
            or stored_legacy_label_dir
            or (self._auto_label_dir_for(image_dir) if image_dir else "")
        )
        self.classes = ClassRegistry(
            self._load_startup_classes(image_dir, startup_label_hint)
        )
        # Construct an empty store. The two label-dir fields are populated
        # below from explicit args / stored settings / auto-discovery.
        self.store = AnnotationStore(self.classes, "")
        self._seed_label_dirs(
            label_dir_arg=label_dir_arg,
            stored_seg_dir=stored_seg_dir,
            stored_detect_dir=stored_detect_dir,
            stored_legacy_label_dir=stored_legacy_label_dir,
        )

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
        act_class_file = QAction("载入类别文件…", self)
        act_class_file.triggered.connect(self._pick_class_file)
        file_menu.addAction(act_class_file)
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
        act_split_dir = QAction("整理标签目录…", self)
        act_split_dir.triggered.connect(self._run_split_wizard)
        tool_menu.addAction(act_split_dir)

        model_menu = mb.addMenu("模型")
        act_sam_ckpt = QAction("加载 SAM 权重…", self)
        act_sam_ckpt.triggered.connect(self._pick_sam_ckpt)
        model_menu.addAction(act_sam_ckpt)
        act_weight_mgr = QAction("SAM 权重管理…", self)
        act_weight_mgr.triggered.connect(self._open_weight_manager)
        model_menu.addAction(act_weight_mgr)
        model_menu.addSeparator()
        act_yolo_w = QAction("选择 YOLO 权重…", self)
        act_yolo_w.triggered.connect(self._pick_yolo_weights)
        model_menu.addAction(act_yolo_w)

        adv_menu = mb.addMenu("高级")
        act_seg_dir = QAction("单独指定分割目录…", self)
        act_seg_dir.triggered.connect(self._pick_seg_dir)
        adv_menu.addAction(act_seg_dir)
        act_detect_dir = QAction("单独指定检测目录…", self)
        act_detect_dir.triggered.connect(self._pick_detect_dir)
        adv_menu.addAction(act_detect_dir)

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
        self.sidebar.weight_manager_requested.connect(self._open_weight_manager)
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
        """Return candidate paths for classes.txt.

        Search locations (in priority order):
        1. image_dir/classes.txt   (most common: classes.txt next to images)
        2. label_dir/classes.txt   (some YOLO tools put it in labels/)
        """
        candidates: list[str] = []
        image_dir = image_dir or getattr(self, "image_dir", "")
        if not label_dir and hasattr(self, "store"):
            label_dir = self.store.label_dir

        if image_dir:
            candidates.append(os.path.join(image_dir, "classes.txt"))
        if label_dir:
            candidates.append(os.path.join(label_dir, "classes.txt"))

        # Deduplicate by resolved path
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
        # No classes.txt found — reset to default
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
        if self.sam_checkpoint and os.path.isfile(self.sam_checkpoint):
            self.sidebar.set_sam_status(os.path.basename(self.sam_checkpoint))
        else:
            self.sidebar.set_sam_status("未加载")
        self.sidebar.set_yolo_weights_label(self.yolo_weights_path)

    def _remember_paths(self):
        self.settings.setValue("paths/image_dir", self.image_dir)
        self.settings.setValue("paths/seg_dir", self.store.seg_dir)
        self.settings.setValue("paths/detect_dir", self.store.detect_dir)
        # ``paths/label_dir`` is read-only legacy: migrated once by
        # _seed_label_dirs at startup and never written again.
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
    def _discover_label_dir(root_dir: str, image_dir: str = "") -> str:
        """Find the label directory by matching .txt filenames against image stems.

        Args:
            root_dir: the directory the user selected (to scan subdirs)
            image_dir: where images actually are (may be a subdir of root_dir)

        Strategy:
        1. Check root_dir/labels/ (standard YOLO layout)
        2. Check image_dir/labels/ (if image_dir != root_dir)
        3. Scan subdirs of root_dir — pick the one with most .txt stem matches
        4. Fall back to root_dir/labels/
        """
        if not root_dir or not os.path.isdir(root_dir):
            return os.path.join(root_dir, "labels") if root_dir else ""

        image_dir = image_dir or root_dir

        # Collect image stems for matching
        image_stems: set[str] = set()
        if os.path.isdir(image_dir):
            for name in os.listdir(image_dir):
                ext = os.path.splitext(name)[1].lower()
                if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                    image_stems.add(os.path.splitext(name)[0])

        if not image_stems:
            return os.path.join(root_dir, "labels")

        # 1. Check standard "labels/" under root
        labels_dir = os.path.join(root_dir, "labels")
        if os.path.isdir(labels_dir):
            return labels_dir

        # 2. Check "labels/" under image_dir (if different from root)
        if image_dir != root_dir:
            img_labels = os.path.join(image_dir, "labels")
            if os.path.isdir(img_labels):
                return img_labels

        # 3. Scan subdirectories of root for matching .txt files
        best_dir = ""
        best_matches = 0
        try:
            for entry in os.scandir(root_dir):
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                # Skip the image directory itself
                if os.path.abspath(entry.path) == os.path.abspath(image_dir):
                    continue
                txt_stems: set[str] = set()
                try:
                    for f in os.scandir(entry.path):
                        if f.is_file() and f.name.endswith(".txt") and f.name != "classes.txt":
                            txt_stems.add(os.path.splitext(f.name)[0])
                except PermissionError:
                    continue
                matches = len(txt_stems & image_stems)
                if matches > best_matches:
                    best_matches = matches
                    best_dir = entry.path
        except PermissionError:
            pass

        if best_dir and best_matches > 0:
            return best_dir

        # 4. Default
        return labels_dir

    @staticmethod
    def _fallback_sam_checkpoint(model_type: str) -> str:
        from .sam_service import SAM2_MODEL_FILES
        filename = SAM_MODEL_FILES.get(model_type) or SAM2_MODEL_FILES.get(model_type, "sam_vit_h_4b8939.pth")
        # Project root = parent of src/
        pkg_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/yolo_sam_labeler/
        src_dir = os.path.dirname(pkg_dir)
        project_root = os.path.dirname(src_dir)
        candidates = [
            os.path.join(project_root, "weights", "sam", filename),  # weights/sam/
            os.path.join(os.getcwd(), "weights", "sam", filename),    # CWD/weights/sam/
            os.path.join(project_root, "weights", filename),          # legacy weights/
            os.path.abspath(filename),                                 # just the filename
            os.path.join(os.getcwd(), filename),                       # CWD/filename
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
        # Clear old annotations immediately so they never bleed into the new directory
        self.store.masks.clear()
        self.store.boxes.clear()
        self.store.last_kind = ""
        self.image_dir = path
        switching_dirs = bool(old_image_dir) and not self._same_path(old_image_dir, path)
        # Drop seg/detect dirs that lived inside the old project; keep dirs
        # outside it (the user explicitly pinned them somewhere stable).
        if switching_dirs:
            if self.store.seg_dir and self._is_subpath(self.store.seg_dir, old_image_dir):
                self.store.seg_dir = ""
            if self.store.detect_dir and self._is_subpath(self.store.detect_dir, old_image_dir):
                self.store.detect_dir = ""
        # Discover actual image location (might be a subdirectory)
        actual_image_dir = discover_image_dir(path)
        if actual_image_dir != path:
            self._log(f"图片目录: {actual_image_dir}", "info")
        self.image_paths = scan_images(actual_image_dir)
        # Auto-discover the label directory if neither seg nor detect dir is set.
        if not self.store.seg_dir and not self.store.detect_dir:
            discovered = self._discover_label_dir(path, actual_image_dir)
            if discovered:
                self._handle_picked_dir(discovered)
                self._maybe_offer_sibling_pair(discovered)
                self._log_label_dirs(picked=discovered, log_zh="标签目录")
        else:
            # Keep showing where labels go even when they were preserved.
            picked = self.store.seg_dir or self.store.detect_dir
            self._log_label_dirs(picked=picked, log_zh="标签目录 (沿用)")
        self._load_classes_for_current_dirs()
        self.index = 0
        # Reset ROI — old ROI mask dimensions won't match new images
        self.roi_mode = "full"
        self.roi_pts.clear()
        self.roi_mask = None
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
        # Final refresh — ensure right panel always reflects current state
        self._refresh_class_list()
        ids = self.classes.sorted_ids()
        if ids:
            if self.current_class_id not in self.classes:
                self.current_class_id = ids[0]
            self.rpanel.select_class(self.current_class_id)
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

        # SAM encode (if model ready)
        if self.sam.is_ready:
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
        if not self.store.seg_dir and not self.store.detect_dir:
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
        report = save_labels(self.store, stem, w, h)
        self._finalize_shared_after_save(report)
        self._autosave_warned = False
        self._log_save_report(report, stem)
        if not silent and not (
            report.refused_seg or report.refused_detect or report.conflict_shared
        ):
            self._log(f"已保存: {stem}.txt", "ok")
        return True

    def _log_save_report(self, report, stem: str):
        """Translate a SaveReport into user-facing log lines."""
        if report.refused_seg:
            self._log(
                f"拒写保护: {stem} 的分割文件实际是检测格式，未覆盖。", "warn"
            )
        if report.refused_detect:
            self._log(
                f"拒写保护: {stem} 的检测文件实际是分割格式，未覆盖。", "warn"
            )
        if report.conflict_shared:
            self._log(
                "共用目录冲突: 同图同时含分割和检测，已只写其中一种。", "warn"
            )
        if report.cleared_seg:
            self._log(f"分割文件已清空: {stem}.txt", "info")
        if report.cleared_detect:
            self._log(f"检测文件已清空: {stem}.txt", "info")
        if report.skipped_no_dir:
            kinds = "/".join(report.skipped_no_dir)
            self._log(f"未配置 {kinds} 目录，对应类型未保存。", "info")

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
        d = QFileDialog.getExistingDirectory(self, "选择标签保存目录", self._first_label_dir() or ".")
        if d:
            self._apply_label_dir_choice(d, kind=None)
            self._remember_paths()
            if self.image_bgr is not None:
                self._load_current_image()

    def _pick_seg_dir(self):
        start = self.store.seg_dir or self._first_label_dir() or "."
        d = QFileDialog.getExistingDirectory(self, "选择分割标签目录", start)
        if not d:
            return
        self._apply_label_dir_choice(d, kind="seg")
        self._remember_paths()
        if self.image_bgr is not None:
            self._load_current_image()

    def _pick_detect_dir(self):
        start = self.store.detect_dir or self._first_label_dir() or "."
        d = QFileDialog.getExistingDirectory(self, "选择检测标签目录", start)
        if not d:
            return
        self._apply_label_dir_choice(d, kind="detect")
        self._remember_paths()
        if self.image_bgr is not None:
            self._load_current_image()

    # ------------------------------------------------------------------
    # Label directory resolution (single source of truth: store.seg_dir/.detect_dir)
    # ------------------------------------------------------------------

    def _first_label_dir(self) -> str:
        """Return the most representative directory for dialogs and pickers."""
        return self.store.seg_dir or self.store.detect_dir or ""

    @staticmethod
    def _sibling_label_dir(path: str, want_kind: str) -> str:
        """Return a sibling directory path for ``want_kind`` next to ``path``.

        Substitutes ``_seg``/``-seg``/``_detect``/``-detect`` suffixes when
        present, else appends ``_<want_kind>``. Never creates the directory.
        """
        if not path:
            return ""
        path = os.path.normpath(path)
        base = os.path.basename(path)
        parent = os.path.dirname(path)
        other = "detect" if want_kind == "seg" else "seg"
        for sep in ("_", "-"):
            suffix = sep + other
            if base.endswith(suffix):
                stem = base[: -len(suffix)]
                return os.path.join(parent, stem + sep + want_kind)
        return path + "_" + want_kind

    def _seed_label_dirs(self, label_dir_arg: str, stored_seg_dir: str,
                         stored_detect_dir: str, stored_legacy_label_dir: str = ""):
        """Initialize ``store.seg_dir`` / ``store.detect_dir`` at startup.

        Decision tree (highest priority first):
            1. ``label_dir_arg`` (CLI / explicit constructor): treat as a
               picked directory and run :meth:`_handle_picked_dir`.
            2. Persisted new keys (``paths/seg_dir`` / ``paths/detect_dir``):
               restore them, seeding a sibling on the empty side if needed.
            3. Legacy ``paths/label_dir`` migration: treat as case 1 and
               clear the legacy key.
            4. Otherwise leave both fields empty; they get set later by
               :meth:`_load_directory` once an image dir is opened.
        """
        # Branch 1: explicit argument wins.
        if label_dir_arg:
            self.store.seg_dir = ""
            self.store.detect_dir = ""
            self._handle_picked_dir(label_dir_arg)
            return

        # Branch 2: validated new keys.
        used_persisted = False
        if stored_seg_dir and os.path.isdir(stored_seg_dir):
            self.store.seg_dir = stored_seg_dir
            used_persisted = True
        else:
            self.store.seg_dir = ""
        if stored_detect_dir and os.path.isdir(stored_detect_dir):
            self.store.detect_dir = stored_detect_dir
            used_persisted = True
        else:
            self.store.detect_dir = ""

        if used_persisted:
            # Auto-fill the missing side so both kinds have a target.
            if self.store.seg_dir and not self.store.detect_dir:
                self.store.detect_dir = self._sibling_label_dir(
                    self.store.seg_dir, "detect"
                )
            if self.store.detect_dir and not self.store.seg_dir:
                self.store.seg_dir = self._sibling_label_dir(
                    self.store.detect_dir, "seg"
                )
            return

        # Branch 3: legacy migration.
        if stored_legacy_label_dir and os.path.isdir(stored_legacy_label_dir):
            self._handle_picked_dir(stored_legacy_label_dir)
            self.settings.remove("paths/label_dir")
            return

        # Branch 4: nothing — caller (_load_directory) handles auto-discovery.

    def _apply_label_dir_choice(self, path: str, kind: str | None):
        """Handle a directory the user picked (single or kind-specific).

        ``kind`` ∈ {``None``, ``"seg"``, ``"detect"``}.
        """
        if kind == "seg":
            self.store.seg_dir = path
            if not self.store.detect_dir:
                self.store.detect_dir = self._sibling_label_dir(path, "detect")
            log_zh = "分割标签目录"
        elif kind == "detect":
            self.store.detect_dir = path
            if not self.store.seg_dir:
                self.store.seg_dir = self._sibling_label_dir(path, "seg")
            log_zh = "检测标签目录"
        else:
            # Unified picker — wipe and let the sniffer decide.
            self.store.seg_dir = ""
            self.store.detect_dir = ""
            self._handle_picked_dir(path)
            self._maybe_offer_sibling_pair(path)
            log_zh = "标签目录"
        self._load_classes_for_current_dirs()
        self._log_label_dirs(picked=path, log_zh=log_zh)

    def _handle_picked_dir(self, path: str):
        """Sniff ``path`` and route to seg_dir/detect_dir accordingly."""
        if not path:
            return
        kind, stats = inspect_label_dir_format(path)
        # Always surface what we saw so the user can sanity-check.
        self._log(
            f"目录嗅探: {kind} (seg={stats['seg']}, detect={stats['detect']}, "
            f"empty={stats['empty']}, scanned={stats['scanned']}/{stats['total']})",
            "info",
        )
        if kind == "seg":
            self.store.seg_dir = path
            if not self.store.detect_dir:
                self.store.detect_dir = self._sibling_label_dir(path, "detect")
        elif kind == "detect":
            self.store.detect_dir = path
            if not self.store.seg_dir:
                self.store.seg_dir = self._sibling_label_dir(path, "seg")
        elif kind == "mixed":
            self._handle_mixed_label_dir(path)
        else:  # "empty" or unknown
            # Shared seed: both fields point to the same path. The first save
            # will collapse this into a single-kind layout via
            # :meth:`_finalize_shared_after_save`.
            self.store.seg_dir = path
            self.store.detect_dir = path

    def _handle_mixed_label_dir(self, path: str):
        """Offer to split a mixed seg+detect directory into two siblings."""
        kind, stats = inspect_label_dir_format(path)
        # ``kind`` is "mixed" when we get here.
        seg_n = stats["seg"]
        det_n = stats["detect"]
        # Default destinations: keep the majority kind in place, move the
        # minority into a sibling.
        if seg_n >= det_n:
            seg_dst = path
            detect_dst = path + "_detect"
            keep_label = "分割"
            move_label = f"检测 → {os.path.basename(detect_dst)}"
        else:
            seg_dst = path + "_seg"
            detect_dst = path
            keep_label = "检测"
            move_label = f"分割 → {os.path.basename(seg_dst)}"

        preview = split_mixed_label_dir(path, seg_dst, detect_dst, dry_run=True)
        moved = preview["moved_seg"] + preview["moved_detect"]
        conflicts = preview["conflicts"]

        box = QMessageBox(self)
        box.setWindowTitle("发现混合标签目录")
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"目录里同时存在两种格式的 YOLO 标签:\n  {path}\n\n"
            f"  分割 (seg): {seg_n} 个\n"
            f"  检测 (detect): {det_n} 个"
        )
        body = (
            f"建议拆为两个目录，保留 {keep_label} 在原处，{move_label}。\n"
            f"将移动 {moved} 个文件。"
        )
        if conflicts:
            body += f"\n注意: {conflicts} 个目标路径已存在，会被跳过。"
        box.setInformativeText(body)
        split_btn = box.addButton("拆分", QMessageBox.AcceptRole)
        share_btn = box.addButton("共用此目录", QMessageBox.NoRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(split_btn)
        box.exec_()
        clicked = box.clickedButton()

        if clicked is split_btn:
            result = split_mixed_label_dir(path, seg_dst, detect_dst, dry_run=False)
            self.store.seg_dir = result["seg_dst"]
            self.store.detect_dir = result["detect_dst"]
            self._log(
                "已拆分: 移动 "
                f"{result['moved_seg']} seg + {result['moved_detect']} detect"
                f"，保留 {result['kept_seg']} seg + {result['kept_detect']} detect"
                f"，跳过 {result['skipped_unknown']} 未识别 / "
                f"{result['skipped_empty']} 空文件"
                + (f" / {result['conflicts']} 冲突" if result['conflicts'] else ""),
                "ok",
            )
        elif clicked is share_btn:
            self.store.seg_dir = path
            self.store.detect_dir = path
            self._log("混合目录: 共用单一目录 (按文件内容分流)", "info")
        else:
            self._log("取消混合目录处理，标签目录未变更。", "warn")

    def _run_split_wizard(self):
        """Manual entry point for "工具 → 整理标签目录…"."""
        path = self._first_label_dir()
        if not path or not os.path.isdir(path):
            QMessageBox.information(self, "整理标签目录", "请先选择一个标签目录。")
            return
        kind, _stats = inspect_label_dir_format(path)
        if kind == "mixed":
            self._handle_mixed_label_dir(path)
            self._remember_paths()
            return
        QMessageBox.information(
            self, "整理标签目录",
            f"当前标签目录格式: {kind}\n无需拆分。"
        )

    def _log_label_dirs(self, picked: str, log_zh: str):
        seg = self.store.seg_dir
        det = self.store.detect_dir
        self._log(f"{log_zh}: {picked}", "ok")
        if seg and det and os.path.abspath(seg) == os.path.abspath(det):
            self._log("分割与检测共用一个目录，按文件内容自动分流。", "info")
        else:
            if seg:
                self._log(f"分割: {seg}", "info")
            if det:
                self._log(f"检测: {det}", "info")

    def _maybe_offer_sibling_pair(self, picked: str):
        """If the picked dir is a clean seg-or-detect, look for a sibling pair."""
        if not picked or not os.path.isdir(picked):
            return
        own_kind, _ = inspect_label_dir_format(picked)
        if own_kind not in ("seg", "detect"):
            return
        other = "detect" if own_kind == "seg" else "seg"
        sibling = self._find_sibling_label_dir(picked, other)
        if not sibling:
            return
        ans = QMessageBox.question(
            self,
            "发现配套标签目录",
            (
                f"在同级目录下找到了 {other} 标签:\n  {sibling}\n\n"
                "是否同时使用两个目录?\n"
                f"  - 分割: {picked if own_kind == 'seg' else sibling}\n"
                f"  - 检测: {picked if own_kind == 'detect' else sibling}"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if ans == QMessageBox.Yes:
            if other == "detect":
                self.store.detect_dir = sibling
            else:
                self.store.seg_dir = sibling

    @staticmethod
    def _find_sibling_label_dir(picked: str, want_kind: str) -> str:
        """Search for a sibling dir whose contents match ``want_kind``."""
        if not picked or not os.path.isdir(picked):
            return ""
        picked = os.path.normpath(picked)
        parent = os.path.dirname(picked)
        own_basename = os.path.basename(picked)
        candidates: list[str] = [picked + "_" + want_kind]
        for suffix in ("_seg", "_detect", "-seg", "-detect"):
            if own_basename.endswith(suffix):
                stem = own_basename[: -len(suffix)]
                candidates.append(os.path.join(parent, stem + "_" + want_kind))
                candidates.append(os.path.join(parent, stem + "-" + want_kind))
                break
        candidates.append(os.path.join(parent, "labels_" + want_kind))
        candidates.append(os.path.join(parent, want_kind + "_labels"))
        candidates.append(os.path.join(parent, want_kind))

        seen = {os.path.abspath(picked)}
        for cand in candidates:
            ap = os.path.abspath(cand)
            if ap in seen or not os.path.isdir(cand):
                continue
            seen.add(ap)
            cand_kind, _ = inspect_label_dir_format(cand)
            if cand_kind == want_kind:
                return cand
        return ""

    def _finalize_shared_after_save(self, report):
        """Collapse a shared (seg_dir == detect_dir) seed after a save.

        Triggered when the saver actually committed annotations. Picks one
        kind to keep at the shared path and re-targets the other to a
        sibling. ``report.conflict_shared`` triggers an interactive prompt.
        """
        if not self.store.seg_dir or not self.store.detect_dir:
            return
        if os.path.abspath(self.store.seg_dir) != os.path.abspath(self.store.detect_dir):
            return

        # Conflict: both kinds present in the shared dir. Ask the user which
        # one should keep the shared path.
        if getattr(report, "conflict_shared", False):
            self._resolve_shared_conflict()
            return

        has_masks = bool(self.store.masks)
        has_boxes = bool(self.store.boxes)
        if has_masks and not has_boxes:
            self.store.detect_dir = self._sibling_label_dir(self.store.seg_dir, "detect")
            self._log("标签格式已自动定为: 分割", "ok")
            self._log(f"检测目录改为: {self.store.detect_dir} (按需创建)", "info")
        elif has_boxes and not has_masks:
            self.store.seg_dir = self._sibling_label_dir(self.store.detect_dir, "seg")
            self._log("标签格式已自动定为: 检测", "ok")
            self._log(f"分割目录改为: {self.store.seg_dir} (按需创建)", "info")
        # both / neither → keep both fields; per-file sniffer handles it.

    def _resolve_shared_conflict(self):
        """Ask the user which kind keeps the shared path on conflict."""
        path = self.store.seg_dir
        box = QMessageBox(self)
        box.setWindowTitle("共用目录冲突")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            f"分割与检测当前共用同一目录:\n  {path}\n\n"
            "本图同时含分割和检测标签，但同一文件不能存两种格式。\n"
            "请选择哪一类留在原目录，另一类将移到 sibling 目录。"
        )
        keep_seg = box.addButton("分割留下", QMessageBox.AcceptRole)
        keep_det = box.addButton("检测留下", QMessageBox.NoRole)
        box.addButton("稍后处理", QMessageBox.RejectRole)
        box.setDefaultButton(keep_seg)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is keep_seg:
            self.store.detect_dir = self._sibling_label_dir(path, "detect")
            self._log(f"分割留在 {path}；检测改写到 {self.store.detect_dir}", "ok")
        elif clicked is keep_det:
            self.store.seg_dir = self._sibling_label_dir(path, "seg")
            self._log(f"检测留在 {path}；分割改写到 {self.store.seg_dir}", "ok")
        else:
            self._log("共用冲突未处理，下次保存仍可能丢失一类。", "warn")

    def _pick_sam_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 SAM 权重", os.path.dirname(self.sam_checkpoint) or ".",
            "PyTorch (*.pth);;All (*.*)"
        )
        if path:
            self.sam_checkpoint = path
            self._remember_paths()
            self._log(f"SAM 权重: {path}", "ok")
            self._load_sam()

    def _open_weight_manager(self):
        """Open the SAM weight manager to download and select checkpoints."""
        weight_dir = os.path.dirname(self.sam_checkpoint) if self.sam_checkpoint else "."
        path, model_type = open_weight_manager(self, weight_dir=weight_dir)
        if path and model_type:
            self.sam_checkpoint = path
            self.model_type = model_type
            self._remember_paths()
            self._log(f"已选择 SAM 权重: {os.path.basename(path)} ({model_type})", "ok")
            self._load_sam()

    def _pick_yolo_weights(self):
        # Default to weights/yolo/ in project root
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(pkg_dir))
        default_yolo_dir = os.path.join(project_root, "weights", "yolo")
        start = (
            os.path.dirname(self.yolo_weights_path)
            or (default_yolo_dir if os.path.isdir(default_yolo_dir) else ".")
        )
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 权重", start,
            "PyTorch (*.pt);;All (*.*)"
        )
        if path:
            self.yolo_weights_path = path
            self.yolo.load(path)
            self.sidebar.set_yolo_weights_label(path)
            self._remember_paths()
            self._log(f"YOLO 权重: {path}", "ok")

    def _pick_class_file(self):
        """Let user manually select a class names file (any .txt)."""
        start_dir = self.image_dir or "."
        path, _ = QFileDialog.getOpenFileName(
            self, "载入类别文件", start_dir,
            "文本文件 (*.txt *.names);;All (*.*)"
        )
        if not path:
            return
        classes = load_class_names(path)
        if not classes:
            self._log(f"类别文件为空或格式不正确: {path}", "warn")
            return
        self.classes.set_names(classes)
        self._refresh_class_list()
        ids = self.classes.sorted_ids()
        if ids:
            self.current_class_id = ids[0]
            self.rpanel.select_class(self.current_class_id)
        self._persist_classes()
        self._log(f"已载入类别文件: {path} ({len(classes)} 个类别)", "ok")

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
