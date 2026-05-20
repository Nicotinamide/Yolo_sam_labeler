"""Right panel: class list."""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QInputDialog,
    QSizePolicy,
)

from .colors import class_colors_for_ids


def _make_compact_btn(text: str, tooltip: str = "") -> QPushButton:
    """Right-panel button that shrinks below its sizeHint instead of
    pushing siblings out of the column at narrow panel widths.
    """
    btn = QPushButton(text)
    btn.setMinimumWidth(0)
    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


class RightPanel(QWidget):
    """Right dock: class list, class management, and compact actions.

    Signals:
        class_selected(class_id)
        class_added(name) -> returns id via return
        class_deleted(class_id)
        class_rename_requested(class_id, new_name)
    """

    class_selected = pyqtSignal(int)
    class_add_requested = pyqtSignal(str)
    class_delete_requested = pyqtSignal(int)
    class_rename_requested = pyqtSignal(int, str)
    save_current = pyqtSignal()
    save_and_next = pyqtSignal()
    save_and_prev = pyqtSignal()
    skip = pyqtSignal()
    undo = pyqtSignal()
    clear = pyqtSignal()
    convert_annotation = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("RightPanel")
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Class list ---
        title = QLabel("类别")
        title.setStyleSheet("font-weight: bold; padding: 0;")
        layout.addWidget(title)

        self.class_list = QListWidget()
        self.class_list.setMinimumWidth(0)
        self.class_list.currentRowChanged.connect(self._on_row_changed)
        self.class_list.itemDoubleClicked.connect(lambda _: self._rename_class())
        layout.addWidget(self.class_list, stretch=1)

        # Add / delete class
        cls_btn_row = QHBoxLayout()
        cls_btn_row.setSpacing(4)
        btn_add = _make_compact_btn("添加", "添加新类别")
        btn_add.clicked.connect(self._add_class)
        cls_btn_row.addWidget(btn_add)
        btn_rename = _make_compact_btn("重命名", "重命名选中类别")
        btn_rename.clicked.connect(self._rename_class)
        cls_btn_row.addWidget(btn_rename)
        btn_del = _make_compact_btn("删除", "删除选中类别")
        btn_del.clicked.connect(self._delete_class)
        cls_btn_row.addWidget(btn_del)
        layout.addLayout(cls_btn_row)

        layout.addWidget(QLabel("操作"))
        save_row = QHBoxLayout()
        save_row.setSpacing(4)
        btn_save = _make_compact_btn("保存", "保存当前图标注 (S)")
        btn_save.clicked.connect(self.save_current.emit)
        save_row.addWidget(btn_save)
        btn_next = _make_compact_btn("保存下一张", "保存当前图并切换到下一张 (N / Space)")
        btn_next.setObjectName("PrimaryBtn")
        btn_next.clicked.connect(self.save_and_next.emit)
        save_row.addWidget(btn_next)
        layout.addLayout(save_row)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        btn_prev = _make_compact_btn("上一张", "保存当前图并切换到上一张 (P)")
        btn_prev.clicked.connect(self.save_and_prev.emit)
        nav_row.addWidget(btn_prev)
        btn_skip = _make_compact_btn("跳过", "不保存直接切到下一张 (D)")
        btn_skip.clicked.connect(self.skip.emit)
        nav_row.addWidget(btn_skip)
        layout.addLayout(nav_row)

        edit_row = QHBoxLayout()
        edit_row.setSpacing(4)
        btn_undo = _make_compact_btn("撤销", "撤销最近一次标注 (U / Ctrl+Z)")
        btn_undo.clicked.connect(self.undo.emit)
        edit_row.addWidget(btn_undo)
        btn_clear = _make_compact_btn("清空", "清空当前图所有标注 (C)")
        btn_clear.clicked.connect(self.clear.emit)
        edit_row.addWidget(btn_clear)
        layout.addLayout(edit_row)

        btn_convert = _make_compact_btn("Mask/框互转 (T)", "把 mask 转成框，或把框送给 SAM 生成 mask")
        btn_convert.clicked.connect(self.convert_annotation.emit)
        layout.addWidget(btn_convert)

        layout.addStretch()

    # ---- class management ----

    def set_classes(self, id_name_map: dict[int, str]):
        """Replace class list with given mapping."""
        self.class_list.blockSignals(True)
        self.class_list.clear()
        colors = class_colors_for_ids(id_name_map.keys())
        for cid in sorted(id_name_map.keys()):
            name = id_name_map[cid]
            item = QListWidgetItem(f"{cid}: {name}")
            item.setData(Qt.UserRole, cid)
            # Use the same palette mapping as the canvas overlay so colors match.
            b, g, r = colors[cid]
            item.setForeground(QColor(r, g, b))
            self.class_list.addItem(item)
        self.class_list.blockSignals(False)

    def select_class(self, class_id: int):
        """Programmatically select a class in the list."""
        for i in range(self.class_list.count()):
            if self.class_list.item(i).data(Qt.UserRole) == class_id:
                self.class_list.setCurrentRow(i)
                return

    # ---- internal slots ----

    def _on_row_changed(self, row: int):
        if row >= 0:
            item = self.class_list.item(row)
            if item:
                cid = item.data(Qt.UserRole)
                if cid is not None:
                    self.class_selected.emit(cid)

    def _add_class(self):
        text, ok = QInputDialog.getText(self, "添加类别", "请输入类别名称:")
        if ok and text.strip():
            self.class_add_requested.emit(text.strip())

    def _delete_class(self):
        row = self.class_list.currentRow()
        if row >= 0:
            item = self.class_list.item(row)
            cid = item.data(Qt.UserRole)
            if cid is not None:
                self.class_delete_requested.emit(cid)

    def _rename_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            return
        item = self.class_list.item(row)
        cid = item.data(Qt.UserRole)
        if cid is None:
            return
        current = item.text().split(": ", 1)[1] if ": " in item.text() else item.text()
        text, ok = QInputDialog.getText(self, "重命名类别", "请输入类别名称:", text=current)
        if ok and text.strip():
            self.class_rename_requested.emit(int(cid), text.strip())
