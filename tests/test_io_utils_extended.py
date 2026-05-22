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
    inspect_label_dir_format,
    split_mixed_label_dir,
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
        seg_dir = str(tmp_path / "labels_seg")
        detect_dir = str(tmp_path / "labels_det")
        cr = ClassRegistry({0: "cat", 1: "dog"})
        store = AnnotationStore(cr, "")
        store.seg_dir = seg_dir
        store.detect_dir = detect_dir

        # Add annotations
        mask_data = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask_data, (50, 50), 30, 1, -1)
        store.add_mask(mask_data, 0)
        store.add_box(10, 10, 60, 60, 1)

        # Save
        save_labels(store, "test_img", 100, 100)

        # Verify files exist
        assert os.path.exists(os.path.join(seg_dir, "test_img.txt"))
        assert os.path.exists(os.path.join(detect_dir, "test_img.txt"))

        # Load into fresh store
        store2 = AnnotationStore(ClassRegistry({0: "cat", 1: "dog"}), "")
        store2.seg_dir = seg_dir
        store2.detect_dir = detect_dir
        fake_img_path = str(tmp_path / "test_img.jpg")
        open(fake_img_path, "w").close()
        load_labels_for_image(store2, fake_img_path, 100, 100)

        assert len(store2.masks) == 1
        assert store2.masks[0].class_id == 0
        assert len(store2.boxes) == 1
        assert store2.boxes[0].class_id == 1

    def test_save_no_boxes_no_detect_file(self, tmp_path):
        """If no boxes exist and detect file doesn't exist, don't create it."""
        seg_dir = str(tmp_path / "labels_seg")
        detect_dir = str(tmp_path / "labels_det")
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        store.seg_dir = seg_dir
        store.detect_dir = detect_dir
        mask_data = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask_data, (50, 50), 30, 1, -1)
        store.add_mask(mask_data, 0)

        save_labels(store, "only_masks", 100, 100)

        assert os.path.exists(os.path.join(seg_dir, "only_masks.txt"))
        # detect file should not exist
        detect_path = os.path.join(detect_dir, "only_masks.txt")
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


# ===========================================================================
# Mixed-format / data-loss regressions
# ===========================================================================


class TestSharedDirAutoSniff:
    """Regression tests for the original data-loss bug.

    When the user selects one directory whose actual format we don't yet
    know, the app seeds both ``seg_dir`` and ``detect_dir`` with that same
    path. The loader must sniff each file's content (5 vs ≥7 tokens) so that
    detection labels in such a 'shared' directory aren't parsed as empty seg
    and then overwritten on the next save.
    """

    def _write_detect_file(self, path, *lines):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _shared_store(self, label_dir, classes=None):
        cr = classes if classes is not None else ClassRegistry({0: "a"})
        store = AnnotationStore(cr, str(label_dir))  # seeds seg_dir == detect_dir
        return store

    def test_detect_in_shared_dir_loads_as_boxes(self, tmp_path):
        label_dir = tmp_path / "label"
        self._write_detect_file(
            label_dir / "img.txt",
            "0 0.500000 0.500000 0.200000 0.200000",
            "1 0.250000 0.250000 0.100000 0.100000",
        )
        cr = ClassRegistry({0: "a"})
        store = self._shared_store(label_dir, cr)
        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 1000, 1000)

        assert len(store.boxes) == 2
        assert store.masks == []
        assert 1 in cr  # auto-registered

    def test_navigate_without_editing_preserves_detect_file(self, tmp_path):
        """Original data-loss scenario: open detect-format dir, navigate, save."""
        label_dir = tmp_path / "label"
        primary = label_dir / "img.txt"
        original = "0 0.500000 0.500000 0.200000 0.200000"
        self._write_detect_file(primary, original)

        store = self._shared_store(label_dir)
        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 1000, 1000)
        # User just navigates — save without any edit.
        save_labels(store, "img", 1000, 1000)

        loaded_back = load_boxes_from_txt(str(primary), 1000, 1000)
        assert len(loaded_back) == 1
        assert loaded_back[0].class_id == 0

    def test_seg_in_shared_dir_loads_as_masks(self, tmp_path):
        label_dir = tmp_path / "label"
        os.makedirs(label_dir)
        (label_dir / "img.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n",
            encoding="utf-8",
        )
        store = self._shared_store(label_dir)
        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 100, 100)

        assert len(store.masks) == 1
        assert store.boxes == []

    def test_shared_dir_round_trip_detect_only(self, tmp_path):
        """A shared dir that only ever sees detect saves stays a single
        detect file — no _seg sibling spawned."""
        label_dir = tmp_path / "label"
        store = self._shared_store(label_dir)
        store.add_box(20, 20, 80, 80, 0)
        save_labels(store, "img", 100, 100)

        # Detect file written.
        assert (label_dir / "img.txt").exists()
        # No empty seg sibling created since seg_dir == detect_dir == label_dir
        # and there were no masks → seg branch is a no-op for non-existing files.
        # Re-reading should classify it as detect and reload boxes.
        store2 = self._shared_store(label_dir)
        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store2, fake_img, 100, 100)
        assert len(store2.boxes) == 1
        assert store2.masks == []



# ===========================================================================
# Directory-level format inspection
# ===========================================================================


class TestInspectLabelDirFormat:
    def test_all_detect(self, tmp_path):
        for stem in ("a", "b", "c"):
            (tmp_path / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        kind, stats = inspect_label_dir_format(str(tmp_path))
        assert kind == "detect"
        assert stats["detect"] == 3
        assert stats["seg"] == 0

    def test_all_seg(self, tmp_path):
        for stem in ("a", "b"):
            (tmp_path / f"{stem}.txt").write_text(
                "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n",
                encoding="utf-8",
            )
        kind, _ = inspect_label_dir_format(str(tmp_path))
        assert kind == "seg"

    def test_empty_dir(self, tmp_path):
        kind, stats = inspect_label_dir_format(str(tmp_path))
        assert kind == "empty"
        assert stats["total"] == 0

    def test_empty_files_only(self, tmp_path):
        for stem in ("a", "b", "c"):
            (tmp_path / f"{stem}.txt").write_text("", encoding="utf-8")
        kind, stats = inspect_label_dir_format(str(tmp_path))
        assert kind == "empty"
        assert stats["empty"] == 3

    def test_majority_detect(self, tmp_path):
        # 19 detect + 1 seg → detect wins (95%); below this falls into "mixed"
        for i in range(19):
            (tmp_path / f"d{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        (tmp_path / "s.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n", encoding="utf-8"
        )
        kind, _ = inspect_label_dir_format(str(tmp_path))
        assert kind == "detect"

    def test_below_majority_threshold_is_mixed(self, tmp_path):
        # 9 detect + 1 seg → only 90%, no longer enough to auto-decide.
        for i in range(9):
            (tmp_path / f"d{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        (tmp_path / "s.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n", encoding="utf-8"
        )
        kind, _ = inspect_label_dir_format(str(tmp_path))
        assert kind == "mixed"

    def test_mixed_split(self, tmp_path):
        for i in range(3):
            (tmp_path / f"d{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        for i in range(3):
            (tmp_path / f"s{i}.txt").write_text(
                "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n", encoding="utf-8"
            )
        kind, _ = inspect_label_dir_format(str(tmp_path))
        assert kind == "mixed"

    def test_ignores_classes_txt(self, tmp_path):
        (tmp_path / "classes.txt").write_text("a\nb\n", encoding="utf-8")
        (tmp_path / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        kind, stats = inspect_label_dir_format(str(tmp_path))
        assert kind == "detect"
        assert stats["scanned"] == 1


class TestEmptyDirSeedFlow:
    """When the user picks an empty directory the app seeds seg_dir and
    detect_dir to the same path. The first save must commit cleanly, leaving
    a single label file of the right format and never spawning empties.
    """

    def test_empty_dir_seed_then_save_box(self, tmp_path):
        label_dir = tmp_path / "label"
        label_dir.mkdir()
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, str(label_dir))  # seeds both to label_dir
        # Sanity: both dirs point to the same place.
        assert store.seg_dir == store.detect_dir == str(label_dir)

        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 100, 100)
        assert store.masks == []
        assert store.boxes == []

        # User draws and saves one detection.
        store.add_box(10, 10, 60, 60, 0)
        save_labels(store, "img", 100, 100)

        # File written, looks like detect (5 tokens).
        out = (label_dir / "img.txt").read_text(encoding="utf-8").strip().split()
        assert len(out) == 5
        # No empty seg sibling created.
        assert not (tmp_path / "label_seg" / "img.txt").exists()

    def test_empty_dir_seed_then_save_mask(self, tmp_path):
        label_dir = tmp_path / "label"
        label_dir.mkdir()
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, str(label_dir))

        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 100, 100)

        m = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(m, (50, 50), 30, 1, -1)
        store.add_mask(m, 0)
        save_labels(store, "img", 100, 100)

        out = (label_dir / "img.txt").read_text(encoding="utf-8").strip().split()
        assert len(out) >= 7  # seg row
        assert not (tmp_path / "label_detect" / "img.txt").exists()


# ===========================================================================
# Explicit seg / detect directory overrides (mixed datasets)
# ===========================================================================


class TestExplicitSegDetectDirs:
    """When the user pins ``store.seg_dir`` and ``store.detect_dir`` to two
    independent folders, both formats must round-trip without interfering."""

    def _write(self, path, *lines):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def test_load_from_two_independent_dirs(self, tmp_path):
        seg_dir = tmp_path / "labels_seg"
        detect_dir = tmp_path / "labels_det"
        self._write(
            seg_dir / "img.txt",
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50",
        )
        self._write(
            detect_dir / "img.txt",
            "1 0.500000 0.500000 0.200000 0.200000",
        )

        cr = ClassRegistry({0: "a", 1: "b"})
        store = AnnotationStore(cr, "")
        store.seg_dir = str(seg_dir)
        store.detect_dir = str(detect_dir)

        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 100, 100)

        assert len(store.masks) == 1
        assert store.masks[0].class_id == 0
        assert len(store.boxes) == 1
        assert store.boxes[0].class_id == 1

    def test_save_writes_to_explicit_dirs_only(self, tmp_path):
        seg_dir = tmp_path / "labels_seg"
        detect_dir = tmp_path / "labels_det"
        seg_dir.mkdir()
        detect_dir.mkdir()

        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        store.seg_dir = str(seg_dir)
        store.detect_dir = str(detect_dir)

        # Add one mask + one box.
        m = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(m, (100, 100), 40, 1, -1)
        store.add_mask(m, 0)
        store.add_box(20, 30, 80, 90, 0)

        save_labels(store, "img", 200, 200)

        assert (seg_dir / "img.txt").exists()
        assert (detect_dir / "img.txt").exists()
        # Seg file has polygon (≥7 tokens), detect file has 5 tokens.
        seg_tokens = (seg_dir / "img.txt").read_text(encoding="utf-8").split()
        det_tokens = (detect_dir / "img.txt").read_text(encoding="utf-8").split()
        assert len(seg_tokens) >= 7
        assert len(det_tokens) == 5
        # No accidental sibling dirs spawned.
        assert not (tmp_path / "labels_seg_detect").exists()
        assert not (tmp_path / "labels_det_seg").exists()

    def test_explicit_seg_only_does_not_save_boxes(self, tmp_path):
        """Pinning only seg_dir means detect saves are simply skipped."""
        seg_dir = tmp_path / "labels_seg"
        seg_dir.mkdir()

        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        store.seg_dir = str(seg_dir)
        # detect_dir intentionally left empty.

        store.add_box(10, 10, 60, 60, 0)
        save_labels(store, "img", 100, 100)

        # No detect dir was created and no detect file written.
        assert not (tmp_path / "labels_detect").exists()
        # Pinned seg dir wasn't polluted with empty file (there were no masks).
        assert not (seg_dir / "img.txt").exists()



# ===========================================================================
# Mixed-directory splitter
# ===========================================================================


class TestSplitMixedLabelDir:
    def _seg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n")

    def _det(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    def test_split_to_default_siblings(self, tmp_path):
        src = tmp_path / "labels"
        self._seg(src / "a.txt")
        self._seg(src / "b.txt")
        self._det(src / "c.txt")
        self._det(src / "d.txt")

        stats = split_mixed_label_dir(str(src))

        # Defaults: src + "_seg" and src + "_detect"
        assert stats["seg_dst"] == str(src) + "_seg"
        assert stats["detect_dst"] == str(src) + "_detect"
        assert stats["moved_seg"] == 2
        assert stats["moved_detect"] == 2
        assert (tmp_path / "labels_seg" / "a.txt").exists()
        assert (tmp_path / "labels_seg" / "b.txt").exists()
        assert (tmp_path / "labels_detect" / "c.txt").exists()
        assert (tmp_path / "labels_detect" / "d.txt").exists()
        # Source files were moved away.
        assert not (src / "a.txt").exists()
        assert not (src / "c.txt").exists()

    def test_keep_majority_in_place(self, tmp_path):
        """Setting seg_dst == src keeps seg files where they are."""
        src = tmp_path / "labels"
        self._seg(src / "a.txt")
        self._seg(src / "b.txt")
        self._seg(src / "c.txt")
        self._det(src / "d.txt")

        stats = split_mixed_label_dir(
            str(src), seg_dst=str(src), detect_dst=str(tmp_path / "labels_detect")
        )

        assert stats["kept_seg"] == 3
        assert stats["moved_detect"] == 1
        assert stats["moved_seg"] == 0
        # seg files untouched
        assert (src / "a.txt").exists()
        assert (src / "b.txt").exists()
        assert (src / "c.txt").exists()
        # detect file moved
        assert (tmp_path / "labels_detect" / "d.txt").exists()
        assert not (src / "d.txt").exists()

    def test_dry_run_does_not_move(self, tmp_path):
        src = tmp_path / "labels"
        self._seg(src / "a.txt")
        self._det(src / "b.txt")

        stats = split_mixed_label_dir(str(src), dry_run=True)
        # Counts reflect the planned moves...
        assert stats["moved_seg"] == 1
        assert stats["moved_detect"] == 1
        # ...but nothing actually changed.
        assert (src / "a.txt").exists()
        assert (src / "b.txt").exists()
        assert not (tmp_path / "labels_seg").exists()
        assert not (tmp_path / "labels_detect").exists()

    def test_preserves_classes_txt(self, tmp_path):
        src = tmp_path / "labels"
        os.makedirs(src)
        (src / "classes.txt").write_text("a\nb\n", encoding="utf-8")
        self._det(src / "img.txt")

        split_mixed_label_dir(str(src))

        # classes.txt is left in source dir.
        assert (src / "classes.txt").exists()
        assert (tmp_path / "labels_detect" / "img.txt").exists()

    def test_skips_empty_and_unknown(self, tmp_path):
        src = tmp_path / "labels"
        os.makedirs(src)
        (src / "empty.txt").write_text("", encoding="utf-8")
        (src / "garbage.txt").write_text("not a yolo line\n", encoding="utf-8")
        self._det(src / "ok.txt")

        stats = split_mixed_label_dir(str(src))
        assert stats["skipped_empty"] == 1
        assert stats["skipped_unknown"] == 1
        assert stats["moved_detect"] == 1
        # empty + unknown stay put
        assert (src / "empty.txt").exists()
        assert (src / "garbage.txt").exists()

    def test_conflict_keeps_source(self, tmp_path):
        src = tmp_path / "labels"
        dst = tmp_path / "labels_detect"
        self._det(src / "img.txt")
        # Pre-create a conflicting target.
        os.makedirs(dst)
        (dst / "img.txt").write_text("0 0.1 0.1 0.05 0.05\n", encoding="utf-8")

        stats = split_mixed_label_dir(str(src), detect_dst=str(dst))
        assert stats["conflicts"] == 1
        assert stats["moved_detect"] == 0
        # Source untouched.
        assert (src / "img.txt").exists()
        # Existing target file unchanged.
        assert (dst / "img.txt").read_text(encoding="utf-8").startswith("0 0.1")



class TestSiblingSeedOnLoad:
    """Regression: when the user opens a single label dir we still need a
    place to write the *other* kind. The app seeds a sibling path on the
    empty side so SAM-derived masks (in a detect-only dir) or boxes (in a
    seg-only dir) round-trip through save/load."""

    def test_save_box_in_detect_only_dir_then_save_mask_creates_seg_sibling(
        self, tmp_path
    ):
        # Simulate: user has only detect labels and seg_dir was left empty.
        detect_dir = tmp_path / "label"
        detect_dir.mkdir()
        (detect_dir / "img.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        store.detect_dir = str(detect_dir)
        store.seg_dir = str(detect_dir) + "_seg"  # what the app would seed
        fake_img = str(tmp_path / "img.png")
        open(fake_img, "w").close()
        load_labels_for_image(store, fake_img, 100, 100)
        # Existing box is loaded.
        assert len(store.boxes) == 1
        assert store.masks == []

        # User runs SAM, gets a mask. Save.
        m = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(m, (50, 50), 30, 1, -1)
        store.add_mask(m, 0)
        save_labels(store, "img", 100, 100)

        seg_path = tmp_path / "label_seg" / "img.txt"
        assert seg_path.exists()
        seg_text = seg_path.read_text(encoding="utf-8").strip()
        assert seg_text  # non-empty
        # Re-load through the same dirs and make sure both kinds are recovered.
        store2 = AnnotationStore(ClassRegistry({0: "a"}), "")
        store2.detect_dir = str(detect_dir)
        store2.seg_dir = str(detect_dir) + "_seg"
        load_labels_for_image(store2, fake_img, 100, 100)
        assert len(store2.masks) == 1
        assert len(store2.boxes) == 1



# ===========================================================================
# SaveReport flags
# ===========================================================================


from yolo_sam_labeler.io_utils import SaveReport, _seg_detect_share_target  # noqa: E402


class TestSaveReport:
    """Verify save_labels returns a populated SaveReport in each scenario."""

    def _make_store(self, tmp_path, seg_dir=None, detect_dir=None):
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        if seg_dir is not None:
            store.seg_dir = str(seg_dir)
        if detect_dir is not None:
            store.detect_dir = str(detect_dir)
        return store

    def test_wrote_seg_only(self, tmp_path):
        seg = tmp_path / "seg"
        det = tmp_path / "det"
        store = self._make_store(tmp_path, seg, det)
        m = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(m, (50, 50), 30, 1, -1)
        store.add_mask(m, 0)
        report = save_labels(store, "img", 100, 100)
        assert report.wrote_seg is True
        assert report.wrote_detect is False
        assert report.refused_seg is False
        assert report.skipped_no_dir == []

    def test_wrote_both(self, tmp_path):
        seg = tmp_path / "seg"
        det = tmp_path / "det"
        store = self._make_store(tmp_path, seg, det)
        m = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(m, (50, 50), 30, 1, -1)
        store.add_mask(m, 0)
        store.add_box(10, 10, 60, 60, 0)
        report = save_labels(store, "img", 100, 100)
        assert report.wrote_seg and report.wrote_detect

    def test_skipped_no_dir(self, tmp_path):
        # Only seg_dir set; box exists but detect_dir is empty.
        seg = tmp_path / "seg"
        store = self._make_store(tmp_path, seg)
        store.add_box(10, 10, 60, 60, 0)
        report = save_labels(store, "img", 100, 100)
        assert "detect" in report.skipped_no_dir
        assert report.wrote_detect is False

    def test_refused_format(self, tmp_path):
        # detect file already exists with detect content; we have empty masks
        # and seg_dir == detect_dir → refusing to clear the detect file via
        # the seg path.
        shared = tmp_path / "label"
        shared.mkdir()
        (shared / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        store = self._make_store(tmp_path, shared, shared)
        # No annotations → save_labels_seg sees existing detect content.
        report = save_labels(store, "img", 100, 100)
        assert report.refused_seg is True
        assert report.wrote_seg is False
        # Detect side is empty too; would attempt to clear, but file is detect-format
        # which matches expected_kind, so it gets cleared.
        assert (shared / "img.txt").read_text(encoding="utf-8") == ""

    def test_cleared_detect(self, tmp_path):
        det = tmp_path / "det"
        det.mkdir()
        (det / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        store = self._make_store(tmp_path, None, det)
        # boxes empty → cleared
        report = save_labels(store, "img", 100, 100)
        assert report.cleared_detect is True
        assert (det / "img.txt").read_text(encoding="utf-8") == ""

    def test_conflict_shared(self, tmp_path):
        shared = tmp_path / "label"
        shared.mkdir()
        store = self._make_store(tmp_path, shared, shared)
        m = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(m, (50, 50), 30, 1, -1)
        store.add_mask(m, 0)
        store.add_box(10, 10, 60, 60, 0)
        store.last_kind = "mask"  # seg wins this turn
        report = save_labels(store, "img", 100, 100)
        assert report.conflict_shared is True
        # seg got written, detect skipped.
        assert report.wrote_seg is True
        assert report.wrote_detect is False


class TestShareTargetHelper:
    def test_disjoint_dirs(self, tmp_path):
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        store.seg_dir = str(tmp_path / "a")
        store.detect_dir = str(tmp_path / "b")
        assert _seg_detect_share_target(store) is False

    def test_shared(self, tmp_path):
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, str(tmp_path / "x"))
        assert _seg_detect_share_target(store) is True

    def test_one_empty(self, tmp_path):
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        store.seg_dir = str(tmp_path / "x")
        # detect_dir = ""
        assert _seg_detect_share_target(store) is False



# ===========================================================================
# Lazy reconcile + empty cleanup
# ===========================================================================


from yolo_sam_labeler.io_utils import (  # noqa: E402
    reconcile_label_file_for_image,
    cleanup_empty_label_files,
)


class TestReconcileLabelFile:
    def _make_store(self, seg, det):
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, "")
        store.seg_dir = str(seg) if seg else ""
        store.detect_dir = str(det) if det else ""
        return store

    def test_detect_in_seg_dir_moves_to_detect(self, tmp_path):
        seg = tmp_path / "txt"
        det = tmp_path / "txt_detect"
        seg.mkdir()
        # detect-format file in seg_dir.
        (seg / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        store = self._make_store(seg, det)
        result = reconcile_label_file_for_image(store, "img")
        assert result["moved_to_detect"] == 1
        assert not (seg / "img.txt").exists()
        assert (det / "img.txt").exists()

    def test_seg_in_detect_dir_moves_to_seg(self, tmp_path):
        seg = tmp_path / "txt"
        det = tmp_path / "txt_detect"
        det.mkdir()
        (det / "img.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n", encoding="utf-8"
        )
        store = self._make_store(seg, det)
        result = reconcile_label_file_for_image(store, "img")
        assert result["moved_to_seg"] == 1
        assert (seg / "img.txt").exists()
        assert not (det / "img.txt").exists()

    def test_matching_format_no_move(self, tmp_path):
        seg = tmp_path / "txt"
        det = tmp_path / "txt_detect"
        seg.mkdir()
        (seg / "img.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n", encoding="utf-8"
        )
        store = self._make_store(seg, det)
        result = reconcile_label_file_for_image(store, "img")
        assert result["moved_to_seg"] == 0
        assert result["moved_to_detect"] == 0
        assert (seg / "img.txt").exists()

    def test_conflict_keeps_source(self, tmp_path):
        seg = tmp_path / "txt"
        det = tmp_path / "txt_detect"
        seg.mkdir()
        det.mkdir()
        # detect file in seg_dir
        (seg / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        # also a pre-existing file in detect_dir at the same stem
        (det / "img.txt").write_text("0 0.1 0.1 0.05 0.05\n", encoding="utf-8")
        store = self._make_store(seg, det)
        result = reconcile_label_file_for_image(store, "img")
        assert result["conflicts"] == 1
        assert result["moved_to_detect"] == 0
        assert (seg / "img.txt").exists()  # source untouched

    def test_shared_dir_skips_reconcile(self, tmp_path):
        shared = tmp_path / "labels"
        shared.mkdir()
        (shared / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        cr = ClassRegistry({0: "a"})
        store = AnnotationStore(cr, str(shared))  # seeds both fields equal
        result = reconcile_label_file_for_image(store, "img")
        assert result["moved_to_seg"] == 0
        assert result["moved_to_detect"] == 0
        assert (shared / "img.txt").exists()

    def test_creates_target_dir_on_demand(self, tmp_path):
        seg = tmp_path / "txt"
        det = tmp_path / "txt_detect"  # not created
        seg.mkdir()
        (seg / "img.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        store = self._make_store(seg, det)
        result = reconcile_label_file_for_image(store, "img")
        assert result["moved_to_detect"] == 1
        assert det.is_dir()
        assert (det / "img.txt").exists()


class TestCleanupEmptyLabelFiles:
    def test_removes_only_empty_txt(self, tmp_path):
        d = tmp_path / "labels"
        d.mkdir()
        (d / "empty.txt").write_text("", encoding="utf-8")
        (d / "real.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = cleanup_empty_label_files(str(d))
        assert result["removed"] == 1
        assert not (d / "empty.txt").exists()
        assert (d / "real.txt").exists()

    def test_preserves_classes_txt(self, tmp_path):
        d = tmp_path / "labels"
        d.mkdir()
        (d / "classes.txt").write_text("", encoding="utf-8")
        (d / "img.txt").write_text("", encoding="utf-8")
        result = cleanup_empty_label_files(str(d))
        # classes.txt skipped even when empty
        assert (d / "classes.txt").exists()
        assert not (d / "img.txt").exists()
        assert result["removed"] == 1

    def test_handles_multiple_dirs_dedup(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "x.txt").write_text("", encoding="utf-8")
        (b / "y.txt").write_text("", encoding="utf-8")
        # Pass same dir twice — should not double-count or error.
        result = cleanup_empty_label_files(str(a), str(b), str(a), "")
        assert result["removed"] == 2
        assert not (a / "x.txt").exists()
        assert not (b / "y.txt").exists()

    def test_nonexistent_dir(self, tmp_path):
        result = cleanup_empty_label_files(str(tmp_path / "nope"))
        assert result["removed"] == 0
