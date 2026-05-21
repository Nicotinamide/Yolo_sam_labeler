"""SAM weight manager dialog — browse, download, and select SAM checkpoints."""

import os
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QAbstractItemView, QMessageBox, QFileDialog, QApplication,
)

from .sam_service import (
    SAM_MODEL_URLS, SAM_MODEL_FILES, SAM_FILE_SIZES,
    SAM2_MODEL_URLS, SAM2_MODEL_FILES, SAM2_FILE_SIZES,
)

# Default search directory for SAM weights — 'weights/sam/' under project root.
def _default_weight_dir() -> str:
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
    project_root = os.path.dirname(pkg_dir)
    candidate = os.path.join(project_root, "weights", "sam")
    if "site-packages" in pkg_dir:
        candidate = os.path.join(os.getcwd(), "weights", "sam")
    return candidate


_DEFAULT_WEIGHT_DIR = _default_weight_dir()


def _human_size(size_bytes: int) -> str:
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_048_576:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


# Model metadata for display
# Model metadata for display — both SAM 1 and SAM 2
_MODEL_INFO = [
    # --- SAM 1 ---
    {
        "type": "vit_h",
        "name": "SAM 1 ViT-H (最精确)",
        "desc": "原版最高精度，GPU 编码 ~0.3s",
        "size": SAM_FILE_SIZES.get("vit_h", 0),
        "file": SAM_MODEL_FILES.get("vit_h", ""),
        "url_dict": SAM_MODEL_URLS,
    },
    {
        "type": "vit_l",
        "name": "SAM 1 ViT-L (平衡)",
        "desc": "精度与速度平衡，GPU 编码 ~0.2s",
        "size": SAM_FILE_SIZES.get("vit_l", 0),
        "file": SAM_MODEL_FILES.get("vit_l", ""),
        "url_dict": SAM_MODEL_URLS,
    },
    {
        "type": "vit_b",
        "name": "SAM 1 ViT-B (最快)",
        "desc": "速度优先，适合 Jetson/CPU",
        "size": SAM_FILE_SIZES.get("vit_b", 0),
        "file": SAM_MODEL_FILES.get("vit_b", ""),
        "url_dict": SAM_MODEL_URLS,
    },
    # --- SAM 2.1 (recommended for new projects) ---
    {
        "type": "sam2.1_hiera_tiny",
        "name": "SAM 2.1 Hiera Tiny",
        "desc": "SAM 2 最小，约 156 MB，速度最快",
        "size": SAM2_FILE_SIZES.get("sam2.1_hiera_tiny", 0),
        "file": SAM2_MODEL_FILES.get("sam2.1_hiera_tiny", ""),
        "url_dict": SAM2_MODEL_URLS,
    },
    {
        "type": "sam2.1_hiera_small",
        "name": "SAM 2.1 Hiera Small",
        "desc": "SAM 2 小型，约 184 MB",
        "size": SAM2_FILE_SIZES.get("sam2.1_hiera_small", 0),
        "file": SAM2_MODEL_FILES.get("sam2.1_hiera_small", ""),
        "url_dict": SAM2_MODEL_URLS,
    },
    {
        "type": "sam2.1_hiera_base_plus",
        "name": "SAM 2.1 Hiera Base+",
        "desc": "SAM 2 中型，约 323 MB",
        "size": SAM2_FILE_SIZES.get("sam2.1_hiera_base_plus", 0),
        "file": SAM2_MODEL_FILES.get("sam2.1_hiera_base_plus", ""),
        "url_dict": SAM2_MODEL_URLS,
    },
    {
        "type": "sam2.1_hiera_large",
        "name": "SAM 2.1 Hiera Large (最精确)",
        "desc": "SAM 2 最大，约 898 MB，比 SAM 1 ViT-H 精度更好且更小",
        "size": SAM2_FILE_SIZES.get("sam2.1_hiera_large", 0),
        "file": SAM2_MODEL_FILES.get("sam2.1_hiera_large", ""),
        "url_dict": SAM2_MODEL_URLS,
    },
]


class WeightManagerDialog(QDialog):
    """Dialog for managing SAM model weights — view status, download, select."""

    def __init__(self, parent=None, weight_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("SAM 权重管理")
        self.setMinimumSize(600, 320)
        self._weight_dir = weight_dir or _DEFAULT_WEIGHT_DIR
        self._selected_path: Optional[str] = None
        self._selected_type: Optional[str] = None
        self._downloading = False
        self._build_ui()
        self._refresh_status()

    @property
    def selected_path(self) -> Optional[str]:
        """Path to the user-selected checkpoint, or None if cancelled."""
        return self._selected_path

    @property
    def selected_type(self) -> Optional[str]:
        """Model type of the selected checkpoint."""
        return self._selected_type

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QLabel("选择或下载 SAM 模型权重：")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        # Table
        self._table = QTableWidget(len(_MODEL_INFO), 5)
        self._table.setHorizontalHeaderLabels(["模型", "说明", "大小", "状态", "操作"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._table)

        # Progress bar (hidden by default)
        self._progress = QProgressBar()
        self._progress.setMaximum(100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._lbl_status = QLabel("")
        self._lbl_status.setVisible(False)
        layout.addWidget(self._lbl_status)

        # Bottom buttons
        btn_row = QHBoxLayout()
        self._btn_dir = QPushButton("更改保存目录…")
        self._btn_dir.clicked.connect(self._pick_dir)
        btn_row.addWidget(self._btn_dir)

        btn_row.addStretch()

        self._btn_use = QPushButton("使用选中的权重")
        self._btn_use.setEnabled(False)
        self._btn_use.clicked.connect(self._use_selected)
        btn_row.addWidget(self._btn_use)

        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

        # Dir label
        self._lbl_dir = QLabel(f"保存目录: {self._weight_dir}")
        self._lbl_dir.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._lbl_dir)

        # Selection changed
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

    def _refresh_status(self):
        """Update table rows with current download status."""
        for row, info in enumerate(_MODEL_INFO):
            # Col 0: Model name
            self._table.setItem(row, 0, QTableWidgetItem(info["name"]))

            # Col 1: Description
            self._table.setItem(row, 1, QTableWidgetItem(info["desc"]))

            # Col 2: Size
            self._table.setItem(row, 2, QTableWidgetItem(_human_size(info["size"])))

            # Col 3: Status
            path = self._path_for(info["type"])
            if os.path.isfile(path):
                actual_size = os.path.getsize(path)
                if actual_size >= info["size"] * 0.95:
                    status = "✓ 已下载"
                else:
                    status = f"⚠ 不完整 ({_human_size(actual_size)})"
            else:
                status = "未下载"
            item = QTableWidgetItem(status)
            if "✓" in status:
                item.setForeground(Qt.darkGreen)
            elif "⚠" in status:
                item.setForeground(Qt.darkYellow)
            self._table.setItem(row, 3, item)

            # Col 4: Action button
            btn = QPushButton("下载" if "未下载" in status or "⚠" in status else "重新下载")
            btn.setProperty("row", row)
            btn.clicked.connect(lambda checked, r=row: self._download_row(r))
            self._table.setCellWidget(row, 4, btn)

        self._on_selection_changed()

    def _path_for(self, model_type: str) -> str:
        filename = SAM_MODEL_FILES.get(model_type) or SAM2_MODEL_FILES.get(model_type, "")
        return os.path.join(self._weight_dir, filename)

    def _on_selection_changed(self):
        row = self._table.currentRow()
        if row < 0 or row >= len(_MODEL_INFO):
            self._btn_use.setEnabled(False)
            return
        info = _MODEL_INFO[row]
        path = self._path_for(info["type"])
        self._btn_use.setEnabled(os.path.isfile(path))

    def _use_selected(self):
        row = self._table.currentRow()
        if row < 0:
            return
        info = _MODEL_INFO[row]
        path = self._path_for(info["type"])
        if os.path.isfile(path):
            self._selected_path = path
            self._selected_type = info["type"]
            self.accept()

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择权重保存目录", self._weight_dir)
        if d:
            self._weight_dir = d
            self._lbl_dir.setText(f"保存目录: {d}")
            self._refresh_status()

    def _download_row(self, row: int):
        if self._downloading:
            QMessageBox.information(self, "请稍候", "正在下载中，请等待完成。")
            return
        info = _MODEL_INFO[row]
        model_type = info["type"]
        save_path = self._path_for(model_type)
        url = info["url_dict"].get(model_type)
        if not url:
            return

        self._downloading = True
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._lbl_status.setVisible(True)
        self._lbl_status.setText(f"正在下载 {info['name']}…")

        # Disable all download buttons during download
        for r in range(len(_MODEL_INFO)):
            widget = self._table.cellWidget(r, 4)
            if widget:
                widget.setEnabled(False)

        import urllib.request

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        expected = info["size"]
        cancelled = [False]

        # Use a timer to do non-blocking download chunks
        part_path = save_path + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            total = resp.length or expected
            downloaded = [0]
            f = open(part_path, "wb")

            def read_chunk():
                if cancelled[0]:
                    f.close()
                    resp.close()
                    if os.path.exists(part_path):
                        os.remove(part_path)
                    self._finish_download(False)
                    return

                chunk = resp.read(65536)
                if not chunk:
                    f.close()
                    resp.close()
                    os.rename(part_path, save_path)
                    self._finish_download(True)
                    return

                f.write(chunk)
                downloaded[0] += len(chunk)
                if total:
                    pct = int(downloaded[0] * 100 / total)
                    self._progress.setValue(pct)
                    self._lbl_status.setText(
                        f"下载 {info['name']}… "
                        f"{downloaded[0] / 1024 / 1024:.1f} / {total / 1024 / 1024:.0f} MB"
                    )
                QTimer.singleShot(0, read_chunk)

            QTimer.singleShot(0, read_chunk)

        except Exception as e:
            if os.path.exists(part_path):
                os.remove(part_path)
            self._finish_download(False)
            QMessageBox.critical(self, "下载失败", f"无法下载:\n{e}")

    def _finish_download(self, success: bool):
        self._downloading = False
        self._progress.setVisible(False)
        self._lbl_status.setVisible(False)
        # Re-enable download buttons
        for r in range(len(_MODEL_INFO)):
            widget = self._table.cellWidget(r, 4)
            if widget:
                widget.setEnabled(True)
        self._refresh_status()


def open_weight_manager(parent, weight_dir: str = "") -> tuple[Optional[str], Optional[str]]:
    """Open the weight manager dialog.

    Returns (checkpoint_path, model_type) or (None, None) if cancelled.
    """
    dlg = WeightManagerDialog(parent, weight_dir=weight_dir)
    if dlg.exec_() == QDialog.Accepted:
        return dlg.selected_path, dlg.selected_type
    return None, None
