"""Extended tests for io_utils.py — edge cases, round-trips, error handling."""
import os

import cv2
import numpy as np
import pytest

from yolo_sam_labeler.io_utils import (
    load_class_names,
    save_class_names,
    scan_images,
    load_image_bgr,
    masks_to_yolo_lines,
    load_masks_from_txt,
    boxes_to_yolo_lines,
    load_boxes_from_txt,
    save_labels,
    load_labels_for_image,
)
from yolo_sam_labeler.models import AnnotationStore, ClassRegistry, Box, Mask


# ===========================================================================
# Class names
# ===========================================================================


class TestClassNames:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "classes.txt"
        path.write_text("", encoding="utf-8")
        assert load_class_names(path) == {}

    def test_comments_and_blanks(self, tmp_path):
        path = tmp_path / "classes.txt"
        path.write_text("# comment\n\ncat\n\n# another\ndog\n", encoding="utf-8")
        result = load_class_names(path)
        assert result == {0: "cat", 1: "dog"}

    def test_mixed_format(self, tmp_path):
        """Mix of implicit and explicit ids."""
        path = tmp_path / "classes.txt"
        path.write_text("cat\n5 truck\nbus\n", encoding="utf-8")
        result = load_class_names(path)
        assert result[0] == "cat"
        assert result[5] == "truck"
        assert result[6] == "bus"  # next_id after max(5)+1

    def test_unicode_names(self, tmp_path):
        path = tmp_path / "classes.txt"
        save_class_names(path, {0: "螺母", 1: "螺栓"})
        result = load_class_names(path)
        assert result == {0: "螺母", 1: "螺栓"}

    def test_sparse_ids_save_format(self, tmp_path):
        path = tmp_path / "classes.txt"
        save_class_names(path, {0: "a", 5: "b", 10: "c"})
        content = path.read_text(encoding="utf-8")
        # Sparse → "id name" format
        assert "0 a" in content
        assert "5 b" in content
        assert "10 c" in content

    def test_nonexistent_file(self, tmp_path):
        assert load_class_names(tmp_path / "nope.txt") == {}

    def test_colon_format(self, tmp_path):
        path = tmp_path / "classes.txt"
        path.write_text("0: apple\n3: orange\n", encoding="utf-8")
        result = load_class_names(path)
        assert result == {0: "apple", 3: "orange"}


# ===========================================================================
# Image scanning
# ===========================================================================


class TestScanImages:
    def test_empty_dir(self, tmp_path):
        assert scan_images(str(tmp_path)) == []

    def test_filters_non_images(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.txt").write_bytes(b"")
        (tmp_path / "c.png").write_bytes(b"")
        result = scan_images(str(tmp_path))
        assert len(result) == 2
        basenames = [os.path.basename(p) for p in result]
        assert "a.jpg" in basenames
        assert "c.png" in basenames
        assert "b.txt" not in basenames

    def test_sorted_order(self, tmp_path):
        (tmp_path / "c.jpg").write_bytes(b"")
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.jpg").write_bytes(b"")
        result = scan_images(str(tmp_path))
        basenames = [os.path.basename(p) for p in result]
        assert basenames == ["a.jpg", "b.jpg", "c.jpg"]

    def test_nonexistent_dir(self):
        assert scan_images("/nonexistent/path/xyz") == []

    def test_case_insensitive_ext(self, tmp_path):
        (tmp_path / "a.JPG").write_bytes(b"")
        (tmp_path / "b.Png").write_bytes(b"")
        result = scan_images(str(tmp_path))
        assert len(result) == 2


# ===========================================================================
# Image loading
# ===========================================================================


class TestLoadImage:
    def test_load_valid_image(self, tmp_path):
        # Create a small real PNG
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        img[5, 10] = [0, 0, 255]
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        loaded = load_image_bgr(path)
        assert loaded is not None
        assert loaded.shape == (10, 20, 3)

    def test_load_nonexistent(self):
        assert load_image_bgr("/no/such/file.jpg") is None

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        assert load_image_bgr(str(path)) is None

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"not an image at all")
        assert load_image_bgr(str(path)) is None


# ===========================================================================
# YOLO segmentation format
# ===========================================================================


class TestYoloSeg:
    def test_mask_round_trip(self):
        """A mask saved and loaded back should approximate the original."""
        w, h = 200, 200
        data = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(data, (100, 100), 50, 1, -1)
        mask = Mask(class_id=3, data=data)

        lines = masks_to_yolo_lines([mask], w, h)
        assert len(lines) == 1
        assert lines[0].startswith("3 ")

        # Write to file and read back
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines))
            path = f.name
        try:
            loaded = load_masks_from_txt(path, w, h)
            assert len(loaded) == 1
            assert loaded[0].class_id == 3
            # IoU should be high (polygon approximation loses some precision)
            intersection = np.sum(loaded[0].data & data)
            union = np.sum(loaded[0].data | data)
            iou = intersection / union
            assert iou > 0.90
        finally:
            os.unlink(path)

    def test_tiny_mask_skipped(self):
        """Masks with area < 50 should be skipped during save."""
        w, h = 100, 100
        data = np.zeros((h, w), dtype=np.uint8)
        data[0:2, 0:2] = 1  # area = 4
        mask = Mask(class_id=0, data=data)
        lines = masks_to_yolo_lines([mask], w, h)
        assert lines == []

    def test_load_too_few_tokens(self, tmp_path):
        """Lines with < 7 tokens should be skipped."""
        path = tmp_path / "bad.txt"
        path.write_text("0 0.5 0.5\n", encoding="utf-8")
        masks = load_masks_from_txt(str(path), 100, 100)
        assert masks == []

    def test_load_odd_coords(self, tmp_path):
        """Lines with odd number of coordinates should be skipped."""
        path = tmp_path / "odd.txt"
        path.write_text("0 0.1 0.1 0.5 0.1 0.5\n", encoding="utf-8")  # 5 coords
        masks = load_masks_from_txt(str(path), 100, 100)
        assert masks == []


# ===========================================================================
# YOLO detection format
# ===========================================================================


class TestYoloDetect:
    def test_box_round_trip(self):
        w, h = 640, 480
        box = Box(class_id=2, x1=100, y1=50, x2=300, y2=250)
        lines = boxes_to_yolo_lines([box], w, h)
        assert len(lines) == 1
        assert lines[0].startswith("2 ")

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines))
            path = f.name
        try:
            loaded = load_boxes_from_txt(path, w, h)
            assert len(loaded) == 1
            b = loaded[0]
            assert b.class_id == 2
            # Allow 1px rounding
            assert abs(b.x1 - 100) <= 1
            assert abs(b.y1 - 50) <= 1
            assert abs(b.x2 - 300) <= 1
            assert abs(b.y2 - 250) <= 1
        finally:
            os.unlink(path)

    def test_tiny_box_skipped(self, tmp_path):
        """Boxes with w<3 or h<3 are skipped on load."""
        path = tmp_path / "tiny.txt"
        # cx=0.5, cy=0.5, w=0.01, h=0.01 → 1px wide on 100x100
        path.write_text("0 0.500000 0.500000 0.010000 0.010000\n", encoding="utf-8")
        boxes = load_boxes_from_txt(str(path), 100, 100)
        assert boxes == []

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        boxes = load_boxes_from_txt(str(path), 100, 100)
        assert boxes == []


# ===========================================================================
# Unified save/load
# ===========================================================================


class TestSaveLoad:
    def test_full_round_trip(self, tmp_path):
        label_dir = str(tmp_path / "labels")
        cr = ClassRegistry({0: "cat", 1: "dog"})
        store = AnnotationStore(cr, label_dir)

        # Add annotations
        mask_data = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask_data, (50, 50), 30, 1, -1)
        store.add_mask(mask_data, 0)
        store.add_box(10, 10, 60, 60, 1)

        # Save
        save_labels(store, "test_img", 100, 100)

        # Verify files exist
        assert os.path.exists(os.path.join(label_dir, "test_img.txt"))
        assert os.path.exists(os.path.join(label_dir + "_detect", "test_img.txt"))

        # Load into fresh store
        store2 = AnnotationStore(ClassRegistry({0: "cat", 1: "dog"}), label_dir)
        fake_img_path = str(tmp_path / "test_img.jpg")
        open(fake_img_path, "w").close()
        load_labels_for_image(store2, fake_img_path, 100, 100)

        assert len(store2.masks) == 1
        assert store2.masks[0].class_id == 0
        assert len(store2.boxes) == 1
        assert store2.boxes[0].class_id == 1

    def test_save_no_boxes_no_detect_file(self, tmp_path):
        """If no boxes exist and detect file doesn't exist, don't create it."""
        label_dir = str(tmp_path / "labels")
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, label_dir)
        mask_data = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask_data, (50, 50), 30, 1, -1)
        store.add_mask(mask_data, 0)

        save_labels(store, "only_masks", 100, 100)

        assert os.path.exists(os.path.join(label_dir, "only_masks.txt"))
        # detect file should not exist
        detect_path = os.path.join(label_dir + "_detect", "only_masks.txt")
        assert not os.path.exists(detect_path)

    def test_load_ensures_unknown_classes(self, tmp_path):
        """Class IDs in label files not in registry get auto-created."""
        label_dir = str(tmp_path / "labels")
        os.makedirs(label_dir)

        # Write a label with class 99
        with open(os.path.join(label_dir, "img.txt"), "w") as f:
            f.write("99 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n")

        cr = ClassRegistry({0: "known"})
        store = AnnotationStore(cr, label_dir)
        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 100, 100)

        assert 99 in cr
        assert cr.name(99) == "99"
