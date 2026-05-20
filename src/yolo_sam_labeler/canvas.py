"""Canvas: image display, coordinate transforms, and compositing.

Coordinate invariant: all annotations are stored in original-image pixel
coordinates.  Display transforms (zoom, pan) affect only the viewport and
never leak into annotation data.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor,
    QWheelEvent, QMouseEvent, QKeyEvent,
)
from PyQt5.QtWidgets import QLabel, QSizePolicy

from .colors import CLASS_PALETTE, class_colors_for_ids
from .models import AnnotationStore


# ---------------------------------------------------------------------------
# Coordinate transformer
# ---------------------------------------------------------------------------


class CoordTransformer:
    """Stable fit-to-canvas viewport for image annotation.

    The viewport follows the original YOLO_Labeler model:
    zoom=1 shows the full image at fit scale, zoom>1 crops original pixels and
    renders that crop back into the same display rectangle. All coordinates
    returned by this class are original-image pixel coordinates.

    canvas_to_image(px, py) -> (ix, iy) or None
    image_to_canvas(ix, iy) -> (px, py)
    """

    def __init__(self, canvas_width: int, canvas_height: int,
                 image_width: int, image_height: int,
                 zoom: float = 1.0,
                 view_origin: tuple[float, float] = (0.0, 0.0)):
        self.cw = max(1, int(canvas_width))
        self.ch = max(1, int(canvas_height))
        self.iw = max(1, int(image_width))
        self.ih = max(1, int(image_height))
        self.zoom = max(1.0, min(12.0, float(zoom)))
        self.view_x1, self.view_y1 = view_origin
        self._clamp_view()

    @property
    def fit_scale(self) -> float:
        return min(self.cw / self.iw, self.ch / self.ih)

    @property
    def scale(self) -> float:
        return self.fit_scale * self.zoom

    @property
    def view_width(self) -> float:
        return self.iw / self.zoom

    @property
    def view_height(self) -> float:
        return self.ih / self.zoom

    def display_size(self) -> tuple[int, int]:
        return max(1, round(self.iw * self.fit_scale)), max(1, round(self.ih * self.fit_scale))

    def display_rect(self) -> tuple[float, float, int, int]:
        dw, dh = self.display_size()
        return (self.cw - dw) / 2, (self.ch - dh) / 2, dw, dh

    def update_canvas_size(self, canvas_width: int, canvas_height: int):
        center = self.view_center()
        self.cw = max(1, int(canvas_width))
        self.ch = max(1, int(canvas_height))
        self.center_on(*center)

    def reset(self):
        self.zoom = 1.0
        self.view_x1 = 0.0
        self.view_y1 = 0.0

    def view_center(self) -> tuple[float, float]:
        return self.view_x1 + self.view_width / 2, self.view_y1 + self.view_height / 2

    def center_on(self, ix: float, iy: float):
        self.view_x1 = ix - self.view_width / 2
        self.view_y1 = iy - self.view_height / 2
        self._clamp_view()

    def set_view_origin(self, x: float, y: float):
        self.view_x1 = x
        self.view_y1 = y
        self._clamp_view()

    def _clamp_view(self):
        max_x = max(0.0, self.iw - self.view_width)
        max_y = max(0.0, self.ih - self.view_height)
        self.view_x1 = max(0.0, min(float(self.view_x1), max_x))
        self.view_y1 = max(0.0, min(float(self.view_y1), max_y))

    def _canvas_to_image_float(self, cx: int, cy: int) -> Optional[tuple[float, float]]:
        x0, y0, dw, dh = self.display_rect()
        lx = cx - x0
        ly = cy - y0
        if lx < 0 or ly < 0 or lx >= dw or ly >= dh:
            return None
        ix = self.view_x1 + (lx / dw) * self.view_width
        iy = self.view_y1 + (ly / dh) * self.view_height
        if ix < 0 or ix >= self.iw or iy < 0 or iy >= self.ih:
            return None
        return ix, iy

    def canvas_to_image(self, cx: int, cy: int) -> Optional[tuple[int, int]]:
        point = self._canvas_to_image_float(cx, cy)
        if point is None:
            return None
        ix, iy = point
        return max(0, min(self.iw - 1, int(round(ix)))), max(0, min(self.ih - 1, int(round(iy))))

    def image_to_canvas(self, ix: int, iy: int) -> tuple[int, int]:
        x0, y0, dw, dh = self.display_rect()
        px = x0 + ((ix - self.view_x1) / self.view_width) * dw
        py = y0 + ((iy - self.view_y1) / self.view_height) * dh
        return round(px), round(py)

    def zoom_at(self, factor: float, cursor_pos: QPoint):
        mx, my = cursor_pos.x(), cursor_pos.y()
        x0, y0, dw, dh = self.display_rect()
        anchor = self._canvas_to_image_float(mx, my)
        if anchor is None:
            anchor = self.view_center()
            rx = 0.5
            ry = 0.5
        else:
            rx = (mx - x0) / dw
            ry = (my - y0) / dh

        self.zoom = max(1.0, min(12.0, self.zoom * factor))
        self.view_x1 = anchor[0] - rx * self.view_width
        self.view_y1 = anchor[1] - ry * self.view_height
        self._clamp_view()

    def pan_by(self, dx: int, dy: int):
        self.view_x1 -= dx / max(self.scale, 1e-6)
        self.view_y1 -= dy / max(self.scale, 1e-6)
        self._clamp_view()


# ---------------------------------------------------------------------------
# Draw state (rubber-band preview for drag-to-box)
# ---------------------------------------------------------------------------


@dataclass
class DrawState:
    active: bool = False
    start_xy: Optional[tuple[int, int]] = None
    current_xy: Optional[tuple[int, int]] = None


# ---------------------------------------------------------------------------
# Pure rendering function (no Qt dependency)
# ---------------------------------------------------------------------------


def render_composite(
    image_bgr: np.ndarray,
    store: AnnotationStore,
    hover_kind: str = "",
    hover_idx: int = -1,
    draw_state: Optional[DrawState] = None,
    roi_pts: Optional[list[tuple[int, int]]] = None,
    roi_mask: Optional[np.ndarray] = None,
    roi_mode: str = "full",
) -> np.ndarray:
    """Render the full composite display image (BGR uint8).

    Layer order (bottom to top):
    1. Original image
    2. Semi-transparent mask overlays
    3. Mask centroid labels
    4. Hovered mask outline (thick white)
    5. Detection boxes with labels
    6. Hovered box highlight (thicker border)
    7. Rubber-band drag preview
    8. ROI polygon overlay
    """
    h_img, w_img = store.image_height, store.image_width
    if h_img == 0 or w_img == 0:
        return image_bgr.copy()

    class_ids = sorted(store.classes.sorted_ids())
    class_colors = class_colors_for_ids(class_ids) if class_ids else {}

    # 1. Mask overlay (alpha blend)
    out = image_bgr.copy().astype(np.float32)
    for mask in store.masks:
        cid = mask.class_id
        col = np.array(class_colors.get(
            cid, CLASS_PALETTE[abs(cid) % len(CLASS_PALETTE)]
        ), dtype=np.float32)
        m = mask.data == 1
        out[m] = out[m] * 0.42 + col * 0.58
    vis = np.clip(out, 0, 255).astype(np.uint8)

    # 2. Mask labels (centroid text)
    for mi, mask in enumerate(store.masks):
        moments = cv2.moments(mask.data.astype(np.uint8))
        if moments["m00"] < 1e-6:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        label = store.classes.name(mask.class_id)
        col = class_colors.get(
            mask.class_id,
            CLASS_PALETTE[abs(mask.class_id) % len(CLASS_PALETTE)],
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = max(0, min(cx - tw // 2, w_img - tw - 2))
        ty = max(th + 4, min(cy, h_img - 4))
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 2, cv2.LINE_AA)
        # Hover highlight
        if hover_kind == "mask" and hover_idx == mi:
            contours, _ = cv2.findContours(
                mask.data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)

    # 3. Detection boxes
    for bi, box in enumerate(store.boxes):
        col = class_colors.get(
            box.class_id,
            CLASS_PALETTE[abs(box.class_id) % len(CLASS_PALETTE)],
        )
        thick = 4 if (hover_kind == "box" and hover_idx == bi) else 2
        cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), col, thick)
        label = store.classes.name(box.class_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tx = max(0, min(box.x1, w_img - tw - 2))
        ty = max(th + 4, min(box.y1 - 4, h_img - 4))
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 2, cv2.LINE_AA)

    # 4. Rubber-band preview
    if draw_state and draw_state.active and draw_state.start_xy and draw_state.current_xy:
        col = class_colors.get(
            store.classes.sorted_ids()[0] if store.classes.sorted_ids() else 0,
            (0, 255, 0),
        )
        x1, y1 = draw_state.start_xy
        x2, y2 = draw_state.current_xy
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)

    # 5. ROI overlay
    if roi_mode == "drawing" and roi_pts and len(roi_pts) >= 1:
        arr = np.array(roi_pts, dtype=np.int32)
        if len(roi_pts) >= 2:
            cv2.polylines(vis, [arr], False, (0, 255, 255), 2, cv2.LINE_AA)
        for px, py in roi_pts:
            cv2.circle(vis, (px, py), 5, (0, 255, 255), -1, cv2.LINE_AA)
    elif roi_mode == "polygon" and roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 200, 255), 2, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# ImageCanvas: QLabel subclass
# ---------------------------------------------------------------------------


class ImageCanvas(QLabel):
    """Display widget for the composited image.

    Forwards mouse wheel, press, move, release, and key events to callbacks
    set by the parent window.  Does NOT own rendering logic — it receives
    QPixmap from the outside via setPixmap().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageCanvas")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Callbacks — set by MainWindow
        self.on_wheel = None       # (QWheelEvent) -> None
        self.on_mouse_press = None  # (QMouseEvent) -> None
        self.on_mouse_move = None   # (QMouseEvent) -> None
        self.on_mouse_release = None  # (QMouseEvent) -> None
        self.on_key_press = None    # (QKeyEvent) -> None
        self.on_resize = None       # () -> None

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.on_resize:
            self.on_resize()

    def wheelEvent(self, ev: QWheelEvent):
        if self.on_wheel:
            self.on_wheel(ev)
        else:
            super().wheelEvent(ev)

    def mousePressEvent(self, ev: QMouseEvent):
        if self.on_mouse_press:
            self.on_mouse_press(ev)
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.on_mouse_move:
            self.on_mouse_move(ev)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if self.on_mouse_release:
            self.on_mouse_release(ev)
        else:
            super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev: QKeyEvent):
        if self.on_key_press:
            self.on_key_press(ev)
        else:
            super().keyPressEvent(ev)


# ---------------------------------------------------------------------------
# Helper: composite image -> QPixmap
# ---------------------------------------------------------------------------


def composite_to_pixmap(vis: np.ndarray, coords: CoordTransformer,
                        canvas_w: int, canvas_h: int,
                        background: str = "#0d0f12") -> QPixmap:
    """Convert a composited BGR image to a stable viewport QPixmap."""
    h, w = vis.shape[:2]
    x1 = max(0, min(w - 1, int(round(coords.view_x1))))
    y1 = max(0, min(h - 1, int(round(coords.view_y1))))
    x2 = max(x1 + 1, min(w, int(round(coords.view_x1 + coords.view_width))))
    y2 = max(y1 + 1, min(h, int(round(coords.view_y1 + coords.view_height))))
    crop = vis[y1:y2, x1:x2]

    display_w, display_h = coords.display_size()
    if crop.shape[1] != display_w or crop.shape[0] != display_h:
        interpolation = cv2.INTER_AREA if coords.scale < 1.0 else cv2.INTER_LINEAR
        display = cv2.resize(crop, (display_w, display_h), interpolation=interpolation)
    else:
        display = np.ascontiguousarray(crop)

    qimg = QImage(display.data, display_w, display_h, 3 * display_w, QImage.Format_BGR888).copy()
    pix = QPixmap.fromImage(qimg)

    canvas_pix = QPixmap(max(canvas_w, 1), max(canvas_h, 1))
    canvas_pix.fill(QColor(background))

    painter = QPainter(canvas_pix)
    px, py, _, _ = coords.display_rect()
    painter.drawPixmap(QPoint(round(px), round(py)), pix)
    painter.end()
    return canvas_pix
