"""Export dialog and packaging worker.

Lets the user pack labels (and optionally images) into a ZIP / TAR.GZ archive
for sharing or training. Pure-Python (zipfile / tarfile, no extra deps).
"""

from __future__ import annotations

import datetime
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Iterable, Optional

from PyQt5.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .io_utils import IMAGE_EXTS

META_DIRNAME = ".meta"
GENERIC_DIR_NAMES = {"images", "imgs", "pics", "pictures", "image", "img",
                     "labels", "label", "labels_seg", "labels_detect"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Resolved settings for one export run."""

    content: str            # "labels" | "dataset" | "images"
    label_source: str       # "seg" | "detect" | "both" | "shared"
    archive_format: str     # "zip" | "targz"
    exclude_meta: bool
    skip_empty_txt: bool
    output_path: str
    image_dir: str
    seg_dir: str
    detect_dir: str
    classes_file: str = ""

    # Convenience
    @property
    def needs_images(self) -> bool:
        return self.content in ("dataset", "images")

    @property
    def needs_labels(self) -> bool:
        return self.content in ("labels", "dataset")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_name(image_dir: str, seg_dir: str, detect_dir: str) -> str:
    """Pick a short project label for filenames."""
    for raw in (image_dir, seg_dir, detect_dir):
        if not raw:
            continue
        path = os.path.abspath(raw)
        base = os.path.basename(path)
        if base.lower() in GENERIC_DIR_NAMES:
            parent = os.path.basename(os.path.dirname(path))
            if parent:
                return _slug(parent)
        if base:
            return _slug(base)
    return "yolo-sam-labels"


def _slug(s: str) -> str:
    """Filesystem-safe shortening of an arbitrary path basename."""
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s) or "labels"


def _content_tag(content: str, label_source: str) -> str:
    if content == "images":
        return "images"
    if content == "dataset":
        if label_source == "seg":
            return "dataset-seg"
        if label_source == "detect":
            return "dataset-detect"
        return "dataset"
    # content == "labels"
    if label_source == "seg":
        return "seg-labels"
    if label_source == "detect":
        return "detect-labels"
    return "labels"


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def _archive_extension(fmt: str) -> str:
    return ".tar.gz" if fmt == "targz" else ".zip"


def suggest_filename(image_dir: str, seg_dir: str, detect_dir: str,
                     content: str, label_source: str, fmt: str) -> str:
    """Build the default basename — without directory."""
    project = _project_name(image_dir, seg_dir, detect_dir)
    tag = _content_tag(content, label_source)
    return f"{project}_{tag}_{_timestamp()}{_archive_extension(fmt)}"


def suggest_directory(image_dir: str, seg_dir: str, detect_dir: str,
                      remembered: str = "") -> str:
    """Pick a sensible default output directory."""
    if remembered and os.path.isdir(remembered):
        return remembered
    for raw in (image_dir, seg_dir, detect_dir):
        if not raw:
            continue
        parent = os.path.dirname(os.path.abspath(raw))
        if parent and os.path.isdir(parent):
            return parent
    return os.path.expanduser("~")


def _list_image_stems(directory: str) -> list[tuple[str, str]]:
    """Return [(stem, abs_path), ...] for images directly under ``directory``."""
    if not directory or not os.path.isdir(directory):
        return []
    out: list[tuple[str, str]] = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            stem = os.path.splitext(name)[0]
            out.append((stem, os.path.join(directory, name)))
    return out


def _list_label_files(directory: str, *, exclude_meta: bool, skip_empty: bool
                      ) -> list[tuple[str, str]]:
    """Return [(rel_arcname, abs_path), ...] for label .txt files.

    ``classes.txt`` is intentionally excluded — it's added separately at the
    archive root by the worker so it lives next to the labels/ folder rather
    than inside it.
    """
    if not directory or not os.path.isdir(directory):
        return []
    out: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isdir(full):
            if exclude_meta and entry == META_DIRNAME:
                continue
            # Don't recurse: YOLO labels are flat by convention.
            continue
        if entry.startswith("."):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext != ".txt":
            continue
        if entry.lower() == "classes.txt":
            # Routed separately by ExportWorker (archive root).
            continue
        if skip_empty:
            try:
                if os.path.getsize(full) == 0:
                    continue
            except OSError:
                continue
        out.append((entry, full))
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class ExportWorker(QObject):
    """Runs the actual file copy/archive in a worker thread."""

    progress = pyqtSignal(int, int, str)  # done, total, current_relpath
    finished = pyqtSignal(bool, str)       # success, message_or_path

    def __init__(self, cfg: ExportConfig):
        super().__init__()
        self.cfg = cfg
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            entries = self._collect_entries()
        except Exception as exc:  # pragma: no cover - defensive
            self.finished.emit(False, f"收集文件失败: {exc}")
            return

        total = len(entries)
        if total == 0:
            self.finished.emit(False, "没有可打包的文件。")
            return

        out_path = self.cfg.output_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        try:
            if self.cfg.archive_format == "zip":
                self._write_zip(out_path, entries)
            else:
                self._write_tar(out_path, entries)
        except Exception as exc:  # pragma: no cover - defensive
            # Best-effort cleanup of partial archive
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except OSError:
                pass
            self.finished.emit(False, f"写入归档失败: {exc}")
            return

        if self._cancelled:
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except OSError:
                pass
            self.finished.emit(False, "已取消")
            return

        self.finished.emit(True, out_path)

    # --------- collection ---------

    def _collect_entries(self) -> list[tuple[str, str]]:
        """Return [(arcname, abs_src_path), ...] in archive order."""
        cfg = self.cfg
        entries: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(arcname: str, src: str):
            if arcname in seen:
                return
            seen.add(arcname)
            entries.append((arcname, src))

        # Labels
        if cfg.needs_labels:
            seg_arc, det_arc = self._label_arc_dirs()
            if cfg.label_source in ("seg", "shared", "both") and cfg.seg_dir:
                for name, src in _list_label_files(
                    cfg.seg_dir,
                    exclude_meta=cfg.exclude_meta,
                    skip_empty=cfg.skip_empty_txt,
                ):
                    add(f"{seg_arc}/{name}", src)
            if cfg.label_source in ("detect", "both") and cfg.detect_dir \
                    and (cfg.label_source != "shared"):
                # Avoid re-adding when seg_dir == detect_dir (shared layout).
                if (cfg.label_source == "both"
                        and cfg.seg_dir
                        and os.path.abspath(cfg.seg_dir) == os.path.abspath(cfg.detect_dir)):
                    pass
                else:
                    for name, src in _list_label_files(
                        cfg.detect_dir,
                        exclude_meta=cfg.exclude_meta,
                        skip_empty=cfg.skip_empty_txt,
                    ):
                        add(f"{det_arc}/{name}", src)

            if cfg.classes_file and os.path.isfile(cfg.classes_file):
                add("classes.txt", cfg.classes_file)

        # Images
        if cfg.needs_images and cfg.image_dir:
            for stem, src in _list_image_stems(cfg.image_dir):
                add(f"images/{os.path.basename(src)}", src)

        return entries

    def _label_arc_dirs(self) -> tuple[str, str]:
        """Return (seg_arcdir, detect_arcdir) inside the archive."""
        cfg = self.cfg
        if cfg.content == "dataset":
            if cfg.label_source == "both":
                return ("labels_seg", "labels_detect")
            return ("labels", "labels")
        # labels-only
        if cfg.label_source == "both":
            return ("labels_seg", "labels_detect")
        return ("labels", "labels")

    # --------- writing ---------

    def _write_zip(self, out_path: str, entries: Iterable[tuple[str, str]]):
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            entries = list(entries)
            total = len(entries)
            for i, (arcname, src) in enumerate(entries, 1):
                if self._cancelled:
                    return
                self.progress.emit(i, total, arcname)
                zf.write(src, arcname)

    def _write_tar(self, out_path: str, entries: Iterable[tuple[str, str]]):
        with tarfile.open(out_path, "w:gz") as tf:
            entries = list(entries)
            total = len(entries)
            for i, (arcname, src) in enumerate(entries, 1):
                if self._cancelled:
                    return
                self.progress.emit(i, total, arcname)
                tf.add(src, arcname=arcname)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class ExportDialog(QDialog):
    """Lets the user configure an export and triggers the worker."""

    SETTINGS_KEY = "export/last_dir"

    def __init__(self, parent: Optional[QWidget] = None, *,
                 image_dir: str, seg_dir: str, detect_dir: str,
                 classes_file: str = "",
                 settings: Optional[QSettings] = None):
        super().__init__(parent)
        self.setWindowTitle("导出数据包")
        self.image_dir = image_dir or ""
        self.seg_dir = seg_dir or ""
        self.detect_dir = detect_dir or ""
        self.classes_file = classes_file or ""
        self.settings = settings or QSettings("yolo-sam-labeler", "yolo-sam-labeler")
        self._has_seg = bool(self.seg_dir and os.path.isdir(self.seg_dir))
        self._has_detect = bool(self.detect_dir and os.path.isdir(self.detect_dir))
        self._shared = (self._has_seg and self._has_detect
                        and os.path.abspath(self.seg_dir) == os.path.abspath(self.detect_dir))
        self._has_image_dir = bool(self.image_dir and os.path.isdir(self.image_dir))

        self._build_ui()
        self._wire_signals()
        self._refresh_label_source_visibility()
        self._refresh_default_filename()
        self._refresh_summary()

    # --------- UI construction ---------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Content group ------------------------------------------------
        content_box = QGroupBox("内容")
        cl = QVBoxLayout(content_box)
        self.rb_labels = QRadioButton("仅标签 (推荐)")
        self.rb_dataset = QRadioButton("图片 + 标签")
        self.rb_images = QRadioButton("仅图片")
        self.rb_labels.setChecked(True)
        if not self._has_image_dir:
            self.rb_dataset.setEnabled(False)
            self.rb_images.setEnabled(False)
        cl.addWidget(self.rb_labels)
        cl.addWidget(self.rb_dataset)
        cl.addWidget(self.rb_images)
        self.content_group = QButtonGroup(self)
        for rb in (self.rb_labels, self.rb_dataset, self.rb_images):
            self.content_group.addButton(rb)
        root.addWidget(content_box)

        # Label source group ------------------------------------------
        self.source_box = QGroupBox("标签来源")
        sl = QVBoxLayout(self.source_box)
        self.lbl_source_summary = QLabel("")
        self.lbl_source_summary.setWordWrap(True)
        sl.addWidget(self.lbl_source_summary)
        self.rb_src_seg = QRadioButton("仅分割")
        self.rb_src_detect = QRadioButton("仅检测")
        self.rb_src_both = QRadioButton("两者都打 (labels_seg/ + labels_detect/)")
        sl.addWidget(self.rb_src_seg)
        sl.addWidget(self.rb_src_detect)
        sl.addWidget(self.rb_src_both)
        self.source_group = QButtonGroup(self)
        for rb in (self.rb_src_seg, self.rb_src_detect, self.rb_src_both):
            self.source_group.addButton(rb)
        # Pick the most reasonable default
        if self._has_seg:
            self.rb_src_seg.setChecked(True)
        elif self._has_detect:
            self.rb_src_detect.setChecked(True)
        self.rb_src_seg.setEnabled(self._has_seg)
        self.rb_src_detect.setEnabled(self._has_detect)
        self.rb_src_both.setEnabled(self._has_seg and self._has_detect and not self._shared)
        root.addWidget(self.source_box)

        # Format group -------------------------------------------------
        fmt_box = QGroupBox("格式")
        fl = QHBoxLayout(fmt_box)
        self.rb_zip = QRadioButton("ZIP")
        self.rb_tar = QRadioButton("TAR.GZ")
        self.rb_zip.setChecked(True)
        fl.addWidget(self.rb_zip)
        fl.addWidget(self.rb_tar)
        fl.addStretch(1)
        self.fmt_group = QButtonGroup(self)
        for rb in (self.rb_zip, self.rb_tar):
            self.fmt_group.addButton(rb)
        root.addWidget(fmt_box)

        # Options ------------------------------------------------------
        opts_box = QGroupBox("选项")
        ol = QVBoxLayout(opts_box)
        self.chk_exclude_meta = QCheckBox("排除工具元数据 (.meta/)")
        self.chk_exclude_meta.setChecked(True)
        self.chk_skip_empty = QCheckBox("跳过空 .txt 文件")
        self.chk_skip_empty.setChecked(True)
        ol.addWidget(self.chk_exclude_meta)
        ol.addWidget(self.chk_skip_empty)
        root.addWidget(opts_box)

        # Output -------------------------------------------------------
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("输出："))
        self.edit_out = QLineEdit()
        out_row.addWidget(self.edit_out, stretch=1)
        self.btn_browse = QPushButton("浏览…")
        out_row.addWidget(self.btn_browse)
        root.addLayout(out_row)

        # Summary
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color: gray;")
        self.lbl_summary.setWordWrap(True)
        root.addWidget(self.lbl_summary)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.button(QDialogButtonBox.Ok).setText("导出")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _wire_signals(self):
        for rb in (self.rb_labels, self.rb_dataset, self.rb_images):
            rb.toggled.connect(self._on_content_changed)
        for rb in (self.rb_src_seg, self.rb_src_detect, self.rb_src_both):
            rb.toggled.connect(self._on_source_changed)
        for rb in (self.rb_zip, self.rb_tar):
            rb.toggled.connect(self._refresh_default_filename)
        self.chk_exclude_meta.toggled.connect(self._refresh_summary)
        self.chk_skip_empty.toggled.connect(self._refresh_summary)
        self.btn_browse.clicked.connect(self._on_browse)

    # --------- reactive helpers ---------

    def _refresh_label_source_visibility(self):
        if self._content() == "images":
            self.source_box.setVisible(False)
            return
        # Hide source box when there is only one option to make.
        if self._shared:
            self.source_box.setVisible(False)
            return
        if self._has_seg and not self._has_detect:
            self.source_box.setVisible(False)
            return
        if self._has_detect and not self._has_seg:
            self.source_box.setVisible(False)
            return
        if not self._has_seg and not self._has_detect:
            self.source_box.setVisible(False)
            return
        self.source_box.setVisible(True)
        bits: list[str] = ["当前配置："]
        if self._has_seg:
            bits.append(f"  分割: {self.seg_dir}")
        if self._has_detect:
            bits.append(f"  检测: {self.detect_dir}")
        self.lbl_source_summary.setText("\n".join(bits))

    def _on_content_changed(self):
        self._refresh_label_source_visibility()
        self._refresh_default_filename()
        self._refresh_summary()

    def _on_source_changed(self):
        self._refresh_default_filename()
        self._refresh_summary()

    def _refresh_default_filename(self):
        out_dir = suggest_directory(
            self.image_dir, self.seg_dir, self.detect_dir,
            remembered=self.settings.value(self.SETTINGS_KEY, "", type=str),
        )
        name = suggest_filename(
            self.image_dir, self.seg_dir, self.detect_dir,
            content=self._content(),
            label_source=self._label_source(),
            fmt=self._format(),
        )
        # Only auto-overwrite when the line is empty or matches the previous suggestion
        current = self.edit_out.text().strip()
        if (not current or current == getattr(self, "_last_suggested", "")):
            self.edit_out.setText(os.path.join(out_dir, name))
        self._last_suggested = os.path.join(out_dir, name)

    def _refresh_summary(self):
        try:
            cfg = self._make_config_for_summary()
        except ValueError as exc:
            self.lbl_summary.setText(f"⚠ {exc}")
            return
        worker = ExportWorker(cfg)
        try:
            entries = worker._collect_entries()  # noqa: SLF001 - intentional preview
        except Exception:
            entries = []
        total_size = 0
        for _, src in entries:
            try:
                total_size += os.path.getsize(src)
            except OSError:
                continue
        self.lbl_summary.setText(
            f"预计：{len(entries)} 个文件，约 {_human_size(total_size)}"
        )

    def _on_browse(self):
        current = self.edit_out.text().strip() or os.path.expanduser("~")
        start_dir = os.path.dirname(current) or os.path.expanduser("~")
        ext_filter = ("ZIP (*.zip)" if self._format() == "zip"
                      else "TAR.GZ (*.tar.gz)")
        chosen, _ = QFileDialog.getSaveFileName(
            self, "选择保存位置", current, ext_filter
        )
        if chosen:
            # Make sure the user-picked file has a sane extension
            if self._format() == "zip" and not chosen.lower().endswith(".zip"):
                chosen += ".zip"
            if self._format() == "targz" and not chosen.lower().endswith(".tar.gz"):
                chosen += ".tar.gz"
            self.edit_out.setText(chosen)
            self._last_suggested = chosen

    # --------- accessors ---------

    def _content(self) -> str:
        if self.rb_dataset.isChecked():
            return "dataset"
        if self.rb_images.isChecked():
            return "images"
        return "labels"

    def _label_source(self) -> str:
        if self._shared:
            return "shared"
        if not self._has_seg and self._has_detect:
            return "detect"
        if not self._has_detect and self._has_seg:
            return "seg"
        if self.rb_src_both.isChecked():
            return "both"
        if self.rb_src_detect.isChecked():
            return "detect"
        return "seg"

    def _format(self) -> str:
        return "targz" if self.rb_tar.isChecked() else "zip"

    # --------- result ---------

    def _make_config_for_summary(self) -> ExportConfig:
        path = self.edit_out.text().strip() or self._last_suggested
        return ExportConfig(
            content=self._content(),
            label_source=self._label_source(),
            archive_format=self._format(),
            exclude_meta=self.chk_exclude_meta.isChecked(),
            skip_empty_txt=self.chk_skip_empty.isChecked(),
            output_path=path,
            image_dir=self.image_dir,
            seg_dir=self.seg_dir,
            detect_dir=self.detect_dir,
            classes_file=self.classes_file,
        )

    def get_config(self) -> ExportConfig:
        cfg = self._make_config_for_summary()
        # Persist last-used directory
        out_dir = os.path.dirname(os.path.abspath(cfg.output_path))
        if out_dir:
            self.settings.setValue(self.SETTINGS_KEY, out_dir)
        return cfg

    # --------- accept guard ---------

    def accept(self):  # noqa: D401 - QDialog override
        path = self.edit_out.text().strip()
        if not path:
            QMessageBox.warning(self, "无效路径", "请填写输出文件路径。")
            return
        cfg = self._make_config_for_summary()
        if cfg.content == "images" and not self._has_image_dir:
            QMessageBox.warning(self, "无图片目录", "尚未设置图片目录。")
            return
        if cfg.needs_labels and not (self._has_seg or self._has_detect):
            QMessageBox.warning(self, "无标签目录", "尚未设置任何标签目录。")
            return
        if os.path.exists(path):
            ans = QMessageBox.question(
                self, "文件已存在",
                f"目标已存在：\n{path}\n\n是否覆盖？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        super().accept()


def _human_size(n_bytes: int) -> str:
    """Format byte counts the way file managers do."""
    n = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n_bytes} B"


# ---------------------------------------------------------------------------
# Convenience runner — wires dialog + progress + worker thread together.
# ---------------------------------------------------------------------------


def run_export(parent: QWidget, *, image_dir: str, seg_dir: str, detect_dir: str,
               classes_file: str = "",
               settings: Optional[QSettings] = None,
               log_fn=None) -> bool:
    """Show the export dialog, run the worker, return True on success."""
    log = log_fn or (lambda msg, level="info": None)

    dlg = ExportDialog(
        parent,
        image_dir=image_dir,
        seg_dir=seg_dir,
        detect_dir=detect_dir,
        classes_file=classes_file,
        settings=settings,
    )
    if dlg.exec_() != QDialog.Accepted:
        return False
    cfg = dlg.get_config()

    progress = QProgressDialog("准备打包…", "取消", 0, 100, parent)
    progress.setWindowTitle("导出数据包")
    progress.setWindowModality(Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setValue(0)

    worker = ExportWorker(cfg)
    thread = QThread(parent)
    worker.moveToThread(thread)

    state = {"success": False, "result": ""}

    def on_progress(done: int, total: int, name: str):
        if total > 0:
            progress.setMaximum(total)
            progress.setValue(done)
        progress.setLabelText(f"{done}/{total}\n{name}")

    def on_finished(success: bool, message: str):
        state["success"] = success
        state["result"] = message
        progress.close()
        thread.quit()

    def on_cancelled():
        worker.cancel()

    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    progress.canceled.connect(on_cancelled)
    thread.started.connect(worker.run)

    thread.start()
    progress.exec_()
    thread.wait(2000)

    if state["success"]:
        log(f"导出完成: {state['result']}", "ok")
        QMessageBox.information(
            parent, "导出完成",
            f"已写入：\n{state['result']}",
        )
        return True
    log(f"导出失败: {state['result']}", "warn")
    if state["result"] and state["result"] != "已取消":
        QMessageBox.warning(parent, "导出失败", state["result"])
    return False
