"""Left sidebar: SAM, YOLO, and ROI parameter panels."""

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QLabel, QCheckBox, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QPushButton, QScrollArea,
    QHBoxLayout, QSizePolicy,
)


def _make_compact_btn(text: str, tooltip: str = "") -> QPushButton:
    """Sidebar button that shrinks below its sizeHint instead of pushing
    siblings out of the column. Without this, narrow sidebar widths force
    the button text past the column edge.
    """
    btn = QPushButton(text)
    btn.setMinimumWidth(0)
    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


class Sidebar(QWidget):
    """Left dock with SAM / YOLO / ROI grouped controls.

    Signals:
        weight_manager_requested()      — open SAM weight manager dialog
        yolo_predict_requested(conf, replace)
        roi_draw_requested()
        roi_close_requested()
        roi_pop_requested()
        roi_full_requested()
    """

    yolo_predict_requested = pyqtSignal(float, bool)  # conf, replace
    weight_manager_requested = pyqtSignal()
    roi_draw_requested = pyqtSignal()
    roi_close_requested = pyqtSignal()
    roi_pop_requested = pyqtSignal()
    roi_full_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("SidebarInner")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- SAM group ---
        sam_grp = QGroupBox("SAM")
        sam_lay = QFormLayout(sam_grp)
        sam_lay.setLabelAlignment(Qt.AlignLeft)
        sam_lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        sam_lay.setFormAlignment(Qt.AlignTop)
        sam_lay.setContentsMargins(6, 6, 6, 6)
        sam_lay.setHorizontalSpacing(6)
        self.lbl_sam_status = QLabel("未加载")
        self.lbl_sam_status.setWordWrap(True)
        self.lbl_sam_status.setMinimumWidth(0)
        self.lbl_sam_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sam_lay.addRow("权重:", self.lbl_sam_status)
        self.btn_weight_mgr = _make_compact_btn("加载 SAM 权重", "下载/选择 SAM 权重并加载")
        self.btn_weight_mgr.clicked.connect(self.weight_manager_requested.emit)
        sam_lay.addRow(self.btn_weight_mgr)
        layout.addWidget(sam_grp)

        # --- YOLO group ---
        yolo_grp = QGroupBox("YOLO")
        yolo_lay = QFormLayout(yolo_grp)
        yolo_lay.setLabelAlignment(Qt.AlignLeft)
        yolo_lay.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        yolo_lay.setContentsMargins(6, 6, 6, 6)
        yolo_lay.setHorizontalSpacing(6)
        self.lbl_yolo_w = QLabel("未加载")
        self.lbl_yolo_w.setWordWrap(True)
        self.lbl_yolo_w.setMinimumWidth(0)
        yolo_lay.addRow("权重:", self.lbl_yolo_w)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setMinimumWidth(0)
        self.spin_conf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        yolo_lay.addRow("置信度:", self.spin_conf)
        self.radio_replace = QRadioButton("替换")
        self.radio_append = QRadioButton("追加")
        self.radio_replace.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.radio_replace)
        bg.addButton(self.radio_append)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.radio_replace)
        row.addWidget(self.radio_append)
        yolo_lay.addRow("模式:", row)
        self.btn_yolo = _make_compact_btn("YOLO 预测", "对当前图像运行 YOLO 推理")
        self.btn_yolo.clicked.connect(
            lambda: self.yolo_predict_requested.emit(
                self.spin_conf.value(), self.radio_replace.isChecked()
            )
        )
        yolo_lay.addRow(self.btn_yolo)
        layout.addWidget(yolo_grp)

        # --- ROI group ---
        roi_grp = QGroupBox("ROI 裁剪")
        roi_lay = QVBoxLayout(roi_grp)
        roi_lay.setContentsMargins(6, 6, 6, 6)
        roi_lay.setSpacing(4)
        self.btn_roi_draw = _make_compact_btn("绘制 ROI", "在画布上点击若干顶点画出多边形")
        self.btn_roi_draw.clicked.connect(self.roi_draw_requested.emit)
        roi_lay.addWidget(self.btn_roi_draw)
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self.btn_roi_close = _make_compact_btn("闭合", "闭合多边形并启用 ROI")
        self.btn_roi_close.clicked.connect(self.roi_close_requested.emit)
        self.btn_roi_pop = _make_compact_btn("撤销点", "撤销最近添加的 ROI 顶点")
        self.btn_roi_pop.clicked.connect(self.roi_pop_requested.emit)
        row2.addWidget(self.btn_roi_close)
        row2.addWidget(self.btn_roi_pop)
        roi_lay.addLayout(row2)
        self.btn_roi_full = _make_compact_btn("恢复全图", "退出 ROI 模式")
        self.btn_roi_full.clicked.connect(self.roi_full_requested.emit)
        roi_lay.addWidget(self.btn_roi_full)
        self.chk_roi_crop = QCheckBox("仅编码裁剪块")
        self.chk_roi_crop.setToolTip("ROI 模式下只把 ROI 内的像素送给 SAM 编码，加速大图。")
        self.chk_roi_crop.setChecked(False)
        roi_lay.addWidget(self.chk_roi_crop)
        self.chk_roi_auto = QCheckBox("ROI 外点击回全图")
        self.chk_roi_auto.setToolTip("ROI 外左键点击时自动恢复全图模式。")
        self.chk_roi_auto.setChecked(True)
        roi_lay.addWidget(self.chk_roi_auto)
        layout.addWidget(roi_grp)

        layout.addStretch()
        inner.setLayout(layout)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def set_sam_status(self, text: str):
        """Update the SAM status label (e.g. 'vit_h 已加载' or '未加载')."""
        self.lbl_sam_status.setText(text)

    def set_yolo_weights_label(self, text: str):
        self.lbl_yolo_w.setText(os.path.basename(text) if text else "未加载")
