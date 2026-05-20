"""Extended tests for canvas.py — CoordTransformer and render_composite."""
import numpy as np
from PyQt5.QtCore import QPoint

from yolo_sam_labeler.canvas import CoordTransformer, DrawState, render_composite
from yolo_sam_labeler.models import AnnotationStore, ClassRegistry, Mask, Box


# ===========================================================================
# CoordTransformer edge cases
# ===========================================================================


class TestCoordTransformerEdge:
    def test_square_image_in_wide_canvas(self):
        """Square image in wider canvas → height-limited fit."""
        coords = CoordTransformer(1000, 500, 500, 500)
        assert coords.fit_scale == 1.0  # min(1000/500, 500/500) = 1.0

    def test_very_small_image(self):
        """Tiny image gets scaled up."""
        coords = CoordTransformer(800, 600, 10, 10)
        assert coords.fit_scale == 60.0  # min(800/10, 600/10)

    def test_zoom_clamped_at_min(self):
        coords = CoordTransformer(800, 600, 800, 600)
        coords.zoom_at(0.1, QPoint(400, 300))  # try to zoom out
        assert coords.zoom == 1.0  # cannot go below 1.0

    def test_zoom_clamped_at_max(self):
        coords = CoordTransformer(800, 600, 800, 600)
        coords.zoom_at(100.0, QPoint(400, 300))  # try extreme zoom
        assert coords.zoom == 12.0  # max

    def test_canvas_to_image_outside_display(self):
        """Clicking outside the displayed image area returns None."""
        coords = CoordTransformer(800, 600, 400, 300)
        # Image is displayed at center, 800×600 (fit_scale=2.0)
        # Full canvas is covered, so edges are still in image
        # But if we had a narrow image:
        coords2 = CoordTransformer(800, 600, 100, 600)
        # fit_scale = min(800/100, 600/600) = 1.0, display_size = (100, 600)
        # display is centered: x offset = (800-100)/2 = 350
        result = coords2.canvas_to_image(0, 300)  # far left of canvas
        assert result is None  # outside display rect

    def test_image_to_canvas_inverse(self):
        coords = CoordTransformer(800, 600, 1600, 1200)
        # At zoom=1, image point (800,600) is the center
        cx, cy = coords.image_to_canvas(800, 600)
        result = coords.canvas_to_image(cx, cy)
        assert result is not None
        ix, iy = result
        assert abs(ix - 800) <= 1
        assert abs(iy - 600) <= 1

    def test_pan_does_nothing_at_zoom_1(self):
        """At zoom=1, the entire image is visible so panning clamps to (0,0)."""
        coords = CoordTransformer(800, 600, 800, 600)
        coords.pan_by(100, 100)
        assert coords.view_x1 == 0.0
        assert coords.view_y1 == 0.0

    def test_reset(self):
        coords = CoordTransformer(800, 600, 1600, 1200)
        coords.zoom_at(3.0, QPoint(400, 300))
        coords.reset()
        assert coords.zoom == 1.0
        assert coords.view_x1 == 0.0
        assert coords.view_y1 == 0.0


# ===========================================================================
# render_composite smoke tests
# ===========================================================================


class TestRenderComposite:
    def _make_store(self, w=100, h=100):
        cr = ClassRegistry({0: "a", 1: "b"})
        store = AnnotationStore(cr)
        store.image_width = w
        store.image_height = h
        return store

    def test_empty_store(self):
        store = self._make_store()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        vis = render_composite(img, store)
        assert vis.shape == (100, 100, 3)
        assert vis.dtype == np.uint8

    def test_with_mask_overlay(self):
        store = self._make_store()
        data = np.zeros((100, 100), dtype=np.uint8)
        data[20:80, 20:80] = 1
        store.add_mask(data, 0)
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        vis = render_composite(img, store)
        # The masked area should be tinted (not pure 128)
        assert not np.array_equal(vis[50, 50], [128, 128, 128])

    def test_with_box(self):
        store = self._make_store()
        store.add_box(10, 10, 90, 90, 1)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        vis = render_composite(img, store)
        # Box border should have non-zero pixels
        assert vis[10, 10].sum() > 0

    def test_with_draw_state(self):
        store = self._make_store()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ds = DrawState(active=True, start_xy=(10, 10), current_xy=(50, 50))
        vis = render_composite(img, store, draw_state=ds)
        # Rubber-band rectangle drawn
        assert vis.shape == (100, 100, 3)

    def test_with_roi_drawing(self):
        store = self._make_store()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        roi_pts = [(10, 10), (50, 10), (50, 50)]
        vis = render_composite(img, store, roi_pts=roi_pts, roi_mode="drawing")
        assert vis.shape == (100, 100, 3)

    def test_hover_highlight(self):
        store = self._make_store()
        store.add_box(10, 10, 90, 90, 0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        vis_no_hover = render_composite(img, store)
        vis_hover = render_composite(img, store, hover_kind="box", hover_idx=0)
        # Hover version should have thicker border → more non-zero pixels
        assert vis_hover.sum() >= vis_no_hover.sum()
