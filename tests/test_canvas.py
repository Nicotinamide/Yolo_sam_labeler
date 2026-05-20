from PyQt5.QtCore import QPoint

from yolo_sam_labeler.canvas import CoordTransformer


def test_initial_view_fits_large_image_to_canvas():
    coords = CoordTransformer(800, 600, 1600, 1200)

    assert coords.zoom == 1.0
    assert coords.fit_scale == 0.5
    assert coords.display_size() == (800, 600)
    assert coords.canvas_to_image(400, 300) == (800, 600)


def test_initial_view_upscales_small_images_to_canvas():
    coords = CoordTransformer(800, 600, 400, 300)

    assert coords.fit_scale == 2.0
    assert coords.display_size() == (800, 600)
    assert coords.canvas_to_image(400, 300) == (200, 150)


def test_zoom_at_keeps_cursor_image_point_stable():
    coords = CoordTransformer(800, 600, 1600, 1200)
    before = coords.canvas_to_image(200, 150)

    coords.zoom_at(2.0, QPoint(200, 150))

    assert coords.zoom == 2.0
    assert coords.canvas_to_image(200, 150) == before


def test_pan_clamps_to_image_bounds():
    coords = CoordTransformer(800, 600, 1600, 1200)
    coords.zoom_at(4.0, QPoint(400, 300))

    coords.pan_by(-10000, -10000)
    assert coords.view_x1 == 1200
    assert coords.view_y1 == 900

    coords.pan_by(10000, 10000)
    assert coords.view_x1 == 0
    assert coords.view_y1 == 0


def test_resize_keeps_view_center():
    coords = CoordTransformer(800, 600, 1600, 1200)
    coords.zoom_at(2.0, QPoint(400, 300))
    center = coords.view_center()

    coords.update_canvas_size(1000, 600)

    assert coords.view_center() == center
