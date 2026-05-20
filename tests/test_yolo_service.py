"""Tests for yolo_service.py — result extraction logic."""
import numpy as np

from yolo_sam_labeler.yolo_service import _build_prediction


class _Boxes:
    def __init__(self, cls, xyxy):
        self.cls = np.asarray(cls, dtype=np.float32)
        self.xyxy = np.asarray(xyxy, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _Masks:
    def __init__(self, xy=None, xyn=None, data=None):
        self.xy = xy
        self.xyn = xyn
        self.data = data


class _Result:
    def __init__(self, boxes=None, masks=None, obb=None):
        self.boxes = boxes
        self.masks = masks
        self.obb = obb


# ===========================================================================
# masks.data path (preferred — bilinear upsample of raw prototype)
# ===========================================================================


def test_build_prediction_uses_data_tensor_when_available():
    """When masks.data exists, it is upsampled via bilinear interpolation."""
    boxes = _Boxes([2], [[10, 20, 30, 50]])
    # A small prototype tensor: 4x4 with a 2x2 square of 1s in the center
    proto = np.zeros((1, 4, 4), dtype=np.float32)
    proto[0, 1:3, 1:3] = 1.0
    masks = _Masks(
        xy=[np.array([[10, 20], [30, 20], [30, 50], [10, 50]], dtype=np.float32)],
        data=proto,
    )
    result = _Result(boxes=boxes, masks=masks)

    pred = _build_prediction(result, (100, 200), replace=True)

    assert pred.replace is True
    assert pred.boxes == []
    assert pred.mask_class_ids == [2]
    assert len(pred.masks) == 1
    mask = pred.masks[0]
    assert mask.shape == (100, 200)
    # Center of the mask should be 1 (the prototype center was 1)
    assert mask[50, 100] == 1
    # Corners should be 0 (bilinear upsample of zeros at edges)
    assert mask[0, 0] == 0
    assert mask[99, 199] == 0


def test_build_prediction_falls_back_to_xy_segments():
    """When masks.data is None, fall back to masks.xy polygon."""
    boxes = _Boxes([1], [[10, 20, 80, 90]])
    masks = _Masks(
        xy=[np.array([[10, 20], [80, 20], [80, 90], [10, 90]], dtype=np.float32)],
        data=None,  # no raw tensor
    )
    result = _Result(boxes=boxes, masks=masks)

    pred = _build_prediction(result, (100, 100), replace=False)

    assert pred.replace is False
    assert pred.mask_class_ids == [1]
    assert len(pred.masks) == 1
    mask = pred.masks[0]
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 1   # inside polygon
    assert mask[5, 5] == 0     # outside polygon


def test_build_prediction_falls_back_to_xyn_segments():
    """When masks.xy is None, fall back to masks.xyn (normalized coords)."""
    boxes = _Boxes([0], [[20, 20, 80, 80]])
    masks = _Masks(
        xy=None,
        xyn=[np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], dtype=np.float32)],
        data=None,
    )
    result = _Result(boxes=boxes, masks=masks)

    pred = _build_prediction(result, (100, 100), replace=False)

    assert len(pred.masks) == 1
    mask = pred.masks[0]
    assert mask[50, 50] == 1
    assert mask[5, 5] == 0


# ===========================================================================
# OBB → axis-aligned box
# ===========================================================================


def test_build_prediction_imports_obb_as_axis_aligned_boxes():
    obb = _Boxes([1], [[10, 20, 40, 60]])
    result = _Result(obb=obb)

    pred = _build_prediction(result, (100, 200), replace=False)

    assert pred.replace is False
    assert pred.masks == []
    assert pred.boxes == [(10, 20, 40, 60)]
    assert pred.box_class_ids == [1]


# ===========================================================================
# Detection-only (no masks at all)
# ===========================================================================


def test_build_prediction_detection_only():
    """Model without seg output → pure boxes."""
    boxes = _Boxes([0, 1], [[10, 10, 50, 50], [60, 60, 90, 90]])
    result = _Result(boxes=boxes, masks=None)

    pred = _build_prediction(result, (100, 100), replace=True)

    assert pred.masks == []
    assert pred.mask_class_ids == []
    assert len(pred.boxes) == 2
    assert pred.box_class_ids == [0, 1]


# ===========================================================================
# Empty results
# ===========================================================================


def test_build_prediction_empty_boxes():
    boxes = _Boxes([], np.zeros((0, 4)))
    result = _Result(boxes=boxes)

    pred = _build_prediction(result, (100, 100), replace=False)

    assert pred.masks == []
    assert pred.boxes == []


# ===========================================================================
# Tiny box filtering
# ===========================================================================


def test_build_prediction_skips_tiny_boxes():
    """Boxes smaller than 3px in either dimension are skipped."""
    boxes = _Boxes([0], [[50, 50, 51, 51]])  # 1x1 → too small
    result = _Result(boxes=boxes, masks=None)

    pred = _build_prediction(result, (100, 100), replace=False)

    assert pred.boxes == []
    assert pred.box_class_ids == []


# ===========================================================================
# Multiple detections
# ===========================================================================


def test_build_prediction_multiple_mixed():
    """Mix of masks and detection-only boxes."""
    boxes = _Boxes([0, 1, 2], [[10, 10, 60, 60], [70, 70, 90, 90], [50, 50, 51, 51]])
    # Only first detection has a valid mask
    proto = np.zeros((3, 8, 8), dtype=np.float32)
    proto[0, 2:6, 2:6] = 1.0  # valid for index 0
    # index 1: all zeros → sum<30 → falls back to box
    # index 2: all zeros → sum<30 AND box is tiny (1×1) → skipped entirely
    masks = _Masks(data=proto)
    result = _Result(boxes=boxes, masks=masks)

    pred = _build_prediction(result, (100, 100), replace=True)

    assert len(pred.masks) == 1
    assert pred.mask_class_ids == [0]
    # index 1 had no valid mask → becomes a box
    assert (70, 70, 90, 90) in pred.boxes
    # index 2 box is too small (< 3px) → skipped
    assert len(pred.boxes) == 1
