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


def test_build_prediction_uses_original_coordinate_mask_segments():
    boxes = _Boxes([2], [[10, 20, 30, 50]])
    masks = _Masks(
        xy=[np.array([[10, 20], [30, 20], [30, 50], [10, 50]], dtype=np.float32)],
        # This raw tensor is deliberately unrelated to the original geometry.
        data=np.ones((1, 2, 2), dtype=np.float32),
    )
    result = _Result(boxes=boxes, masks=masks)

    pred = _build_prediction(result, (100, 200), replace=True)

    assert pred.replace is True
    assert pred.boxes == []
    assert pred.mask_class_ids == [2]
    assert len(pred.masks) == 1
    mask = pred.masks[0]
    assert mask.shape == (100, 200)
    assert mask[25, 15] == 1
    assert mask[5, 5] == 0


def test_build_prediction_imports_obb_as_axis_aligned_boxes():
    obb = _Boxes([1], [[10, 20, 40, 60]])
    result = _Result(obb=obb)

    pred = _build_prediction(result, (100, 200), replace=False)

    assert pred.replace is False
    assert pred.masks == []
    assert pred.boxes == [(10, 20, 40, 60)]
    assert pred.box_class_ids == [1]
