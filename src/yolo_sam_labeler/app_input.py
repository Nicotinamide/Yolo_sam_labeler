"""Mouse and keyboard input handling (mixin for MainWindow)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

from .canvas import DrawState

if TYPE_CHECKING:
    from .canvas import CoordTransformer
    from .models import AnnotationStore


class InputHandlerMixin:
    """Mixin that provides mouse/keyboard event handlers for MainWindow.

    Expects the host class to expose:
        image_bgr, image_shape, image_paths, index
        coords: CoordTransformer
        store: AnnotationStore
        classes: ClassRegistry
        canvas, sidebar
        draw_state: DrawState
        hover_kind, hover_idx
        current_class_id: int
        roi_mode, roi_pts
        sam: SamService
        _panning, _pan_start, _pan_view_start
        _guard() -> bool
        _refresh_canvas()
        _log(msg, level)
        _sam_predict(x, y)
        _undo()
        _save_current()
        _save_and_next()
        _save_and_prev()
        _save_and_close()
        _skip()
        _clear()
        _delete_hovered()
        _reset_zoom()
        _convert_hovered_annotation()
        _set_current_class(cid)
        _apply_class_key(cid)
    """

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _guard(self) -> bool:
        """Return False if canvas is not ready for interaction."""
        return self.image_bgr is not None and self.coords is not None

    # ------------------------------------------------------------------
    # Wheel
    # ------------------------------------------------------------------

    def _on_wheel(self, ev: QWheelEvent):
        if not self._guard():
            return
        factor = 1.12 if ev.angleDelta().y() > 0 else 1 / 1.12
        self.coords.zoom_at(factor, ev.pos())
        self._refresh_canvas()

    # ------------------------------------------------------------------
    # Mouse press
    # ------------------------------------------------------------------

    def _on_mouse_press(self, ev: QMouseEvent):
        self.canvas.setFocus()
        if not self._guard():
            return

        # Pan: middle button or Alt+left
        alt_left = ev.button() == Qt.LeftButton and (ev.modifiers() & Qt.AltModifier)
        if ev.button() == Qt.MidButton or alt_left:
            self._panning = True
            self._pan_start = QPoint(ev.pos())
            self._pan_view_start = (self.coords.view_x1, self.coords.view_y1)
            return

        p = self.coords.canvas_to_image(ev.pos().x(), ev.pos().y())
        if p is None:
            return
        ix, iy = p

        # ROI drawing mode
        if self.roi_mode == "drawing":
            if ev.button() == Qt.LeftButton:
                self.roi_pts.append((ix, iy))
                self._refresh_canvas()
            elif ev.button() == Qt.RightButton and self.roi_pts:
                self.roi_pts.pop()
                self._refresh_canvas()
            return

        # Left drag draws boxes. Ctrl/Shift + left click runs SAM explicitly.
        if ev.button() == Qt.LeftButton:
            sam_click = bool(ev.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))
            if sam_click:
                if not self.sam.is_ready:
                    self._log("请先加载 SAM 模型。", "warn")
                    return
                self._sam_predict(ix, iy)
                return
            self.draw_state = DrawState(active=True, start_xy=(ix, iy), current_xy=(ix, iy))
        elif ev.button() == Qt.RightButton:
            if self.store.delete_at(ix, iy):
                self._refresh_canvas()

    # ------------------------------------------------------------------
    # Mouse move
    # ------------------------------------------------------------------

    def _on_mouse_move(self, ev: QMouseEvent):
        if not self._guard():
            return

        # Pan
        if self._panning and self._pan_start is not None:
            dx = ev.pos().x() - self._pan_start.x()
            dy = ev.pos().y() - self._pan_start.y()
            self.coords.set_view_origin(
                self._pan_view_start[0] - dx / max(self.coords.scale, 1e-6),
                self._pan_view_start[1] - dy / max(self.coords.scale, 1e-6),
            )
            self._refresh_canvas()
            return

        # Rubber-band update during drag
        if self.draw_state.active:
            p = self.coords.canvas_to_image(ev.pos().x(), ev.pos().y())
            if p is not None:
                self.draw_state.current_xy = p
                self._refresh_canvas()
            return

        # Hover detection
        p = self.coords.canvas_to_image(ev.pos().x(), ev.pos().y())
        if p is not None:
            kind, idx = self.store.find_at(p[0], p[1])
            if kind != self.hover_kind or idx != self.hover_idx:
                self.hover_kind = kind
                self.hover_idx = idx
                self._refresh_canvas()
        elif self.hover_kind:
            self.hover_kind = ""
            self.hover_idx = -1
            self._refresh_canvas()

    # ------------------------------------------------------------------
    # Mouse release
    # ------------------------------------------------------------------

    def _on_mouse_release(self, ev: QMouseEvent):
        if self._panning:
            self._panning = False
            self._pan_start = None
            self._pan_view_start = (0.0, 0.0)
            return

        if not self._guard():
            return

        if self.draw_state.active and ev.button() == Qt.LeftButton:
            self.draw_state.active = False
            p = self.coords.canvas_to_image(ev.pos().x(), ev.pos().y())
            if p is not None and self.draw_state.start_xy:
                ix_end, iy_end = p
                ix_start, iy_start = self.draw_state.start_xy
                dx = abs(ix_end - ix_start)
                dy = abs(iy_end - iy_start)
                h, w = self.image_shape
                if dx >= 5 and dy >= 5:
                    x1 = max(0, min(ix_start, ix_end))
                    y1 = max(0, min(iy_start, iy_end))
                    x2 = min(w - 1, max(ix_start, ix_end))
                    y2 = min(h - 1, max(iy_start, iy_end))
                    self.store.add_box(x1, y1, x2, y2, self.current_class_id)
                    self._log(
                        f"检测框: {self.classes.name(self.current_class_id)} ({x2 - x1}×{y2 - y1})",
                        "ok",
                    )
            self.draw_state = DrawState()
            self._refresh_canvas()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _on_key_press(self, ev: QKeyEvent):
        k = ev.key()
        mod = ev.modifiers()
        ctrl_alt_meta = mod & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)
        shift_only = bool(mod & Qt.ShiftModifier) and not ctrl_alt_meta
        plain = not (ctrl_alt_meta or (mod & Qt.ShiftModifier))

        if (k == Qt.Key_Z and mod & Qt.ControlModifier) or (plain and k == Qt.Key_U):
            self._undo()
            return
        if k == Qt.Key_S and mod & Qt.ControlModifier:
            self._save_current()
            return

        # Navigation
        if plain and k in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_hovered()
            return
        if plain and k in (Qt.Key_Q, Qt.Key_E, Qt.Key_Escape):
            self._save_and_close()
            return
        if plain and k == Qt.Key_S:
            self._save_current()
            return
        if plain and k in (Qt.Key_N, Qt.Key_Space):
            self._save_and_next()
            return
        if plain and k == Qt.Key_P:
            self._save_and_prev()
            return
        if plain and k == Qt.Key_D:
            self._skip()
            return
        if plain and k == Qt.Key_C:
            self._clear()
            return
        if plain and k == Qt.Key_R:
            self._reset_zoom()
            return
        if plain and k == Qt.Key_T:
            self._convert_hovered_annotation()
            return
        if (plain or shift_only) and k in (Qt.Key_Plus, Qt.Key_Equal):
            center = QPoint(max(0, self.canvas.width() // 2), max(0, self.canvas.height() // 2))
            if self.coords is not None:
                self.coords.zoom_at(1.12, center)
                self._refresh_canvas()
            return
        if (plain or shift_only) and k in (Qt.Key_Minus, Qt.Key_Underscore):
            center = QPoint(max(0, self.canvas.width() // 2), max(0, self.canvas.height() // 2))
            if self.coords is not None:
                self.coords.zoom_at(1 / 1.12, center)
                self._refresh_canvas()
            return

        # Class switching
        text = ev.text()
        if plain and text in ("[", ","):
            ids = self.classes.sorted_ids()
            if ids:
                try:
                    idx = ids.index(self.current_class_id)
                    idx = (idx - 1) % len(ids)
                except ValueError:
                    idx = 0
                self._set_current_class(ids[idx])
            return
        if plain and text in ("]", "."):
            ids = self.classes.sorted_ids()
            if ids:
                try:
                    idx = ids.index(self.current_class_id)
                    idx = (idx + 1) % len(ids)
                except ValueError:
                    idx = 0
                self._set_current_class(ids[idx])
            return
        if plain and Qt.Key_0 <= k <= Qt.Key_9:
            self._apply_class_key(k - Qt.Key_0)
            return
        if shift_only and Qt.Key_A <= k <= Qt.Key_Z:
            self._apply_class_key(k - Qt.Key_A + 10)
            return
