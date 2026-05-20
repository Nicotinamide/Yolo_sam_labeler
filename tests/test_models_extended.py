"""Extended tests for models.py — edge cases and boundary conditions."""
import numpy as np
import pytest

from yolo_sam_labeler.models import Box, Mask, ClassRegistry, AnnotationStore


# ===========================================================================
# Box edge cases
# ===========================================================================


class TestBoxEdge:
    def test_zero_area_box(self):
        b = Box(class_id=0, x1=5, y1=5, x2=5, y2=5)
        assert b.width == 0
        assert b.height == 0
        assert b.center == (5.0, 5.0)
        # A point-box should still contain its own corner
        assert b.contains(5, 5)

    def test_negative_coords(self):
        """Boxes can technically have negative coords (e.g., from math error)."""
        b = Box(class_id=0, x1=-10, y1=-10, x2=10, y2=10)
        assert b.width == 20
        assert b.contains(0, 0)
        assert b.contains(-10, -10)
        assert not b.contains(-11, 0)

    def test_contains_on_boundary(self):
        b = Box(class_id=0, x1=0, y1=0, x2=100, y2=100)
        assert b.contains(0, 0)       # top-left corner
        assert b.contains(100, 100)   # bottom-right corner
        assert b.contains(0, 50)      # left edge
        assert not b.contains(101, 50)


# ===========================================================================
# Mask edge cases
# ===========================================================================


class TestMaskEdge:
    def test_empty_mask(self):
        data = np.zeros((50, 50), dtype=np.uint8)
        m = Mask(class_id=0, data=data)
        assert not m.contains(25, 25)
        assert not m.contains(0, 0)

    def test_full_mask(self):
        data = np.ones((50, 50), dtype=np.uint8)
        m = Mask(class_id=0, data=data)
        assert m.contains(0, 0)
        assert m.contains(49, 49)
        assert not m.contains(50, 50)  # out of bounds

    def test_out_of_bounds(self):
        data = np.ones((10, 10), dtype=np.uint8)
        m = Mask(class_id=0, data=data)
        assert not m.contains(-1, 5)
        assert not m.contains(5, -1)
        assert not m.contains(10, 5)
        assert not m.contains(5, 10)

    def test_single_pixel_mask(self):
        data = np.zeros((100, 100), dtype=np.uint8)
        data[42, 77] = 1
        m = Mask(class_id=1, data=data)
        assert m.contains(77, 42)  # x=col, y=row
        assert not m.contains(42, 77)


# ===========================================================================
# ClassRegistry edge cases
# ===========================================================================


class TestClassRegistryEdge:
    def test_empty_registry(self):
        cr = ClassRegistry()
        assert len(cr) == 0
        assert cr.sorted_ids() == []
        assert cr.max_id() == -1
        assert 0 not in cr

    def test_name_fallback(self):
        """Requesting name for nonexistent id returns str(id)."""
        cr = ClassRegistry({0: "cat"})
        assert cr.name(99) == "99"

    def test_add_increments_from_max(self):
        cr = ClassRegistry({5: "a", 10: "b"})
        new_id = cr.add("c")
        assert new_id == 11  # max(5,10)+1

    def test_ensure_no_duplicate_signal(self):
        cr = ClassRegistry({0: "a"})
        # ensure existing should not change
        assert cr.ensure(0) is False

    def test_ensure_ids_empty(self):
        cr = ClassRegistry({0: "x"})
        assert cr.ensure_ids([]) is False

    def test_set_names_cleans_empty(self):
        cr = ClassRegistry()
        cr.set_names({0: "  cat  ", 1: ""})
        assert cr.name(0) == "cat"
        assert cr.name(1) == "1"  # empty name → str(id)

    def test_remove_nonexistent(self):
        cr = ClassRegistry({0: "a"})
        assert cr.remove(99) is False

    def test_rename_nonexistent(self):
        cr = ClassRegistry({0: "a"})
        assert cr.rename(99, "b") is False


# ===========================================================================
# AnnotationStore edge cases
# ===========================================================================


class TestAnnotationStoreEdge:
    def _store(self):
        cr = ClassRegistry({0: "a", 1: "b"})
        return AnnotationStore(cr)

    def test_find_at_empty(self):
        s = self._store()
        kind, idx = s.find_at(50, 50)
        assert kind == ""
        assert idx == -1

    def test_undo_empty_store(self):
        s = self._store()
        assert s.undo_last() is False

    def test_delete_at_miss(self):
        s = self._store()
        assert s.delete_at(50, 50) is False

    def test_relabel_out_of_range(self):
        s = self._store()
        assert s.relabel("box", 0, 1) is False
        assert s.relabel("mask", -1, 0) is False
        assert s.relabel("invalid", 0, 0) is False

    def test_replace_mask_with_box_invalid_index(self):
        s = self._store()
        assert s.replace_mask_with_box(-1, 0, 0, 10, 10) is False
        assert s.replace_mask_with_box(99, 0, 0, 10, 10) is False

    def test_replace_box_with_mask_snapshot_mismatch(self):
        s = self._store()
        s.add_box(10, 10, 50, 50, 0)
        # Wrong snapshot
        wrong_snapshot = (1, 10, 10, 50, 50)  # class_id=1, but actual is 0
        mask = np.ones((100, 100), dtype=np.uint8)
        assert s.replace_box_with_mask(0, wrong_snapshot, mask, 0) is False

    def test_replace_box_with_mask_finds_by_scan(self):
        """If idx is wrong but snapshot matches another box, find it."""
        s = self._store()
        s.add_box(10, 10, 50, 50, 0)
        s.add_box(60, 60, 90, 90, 1)
        # Snapshot matches box at index 1 but we pass idx=0
        snapshot = (1, 60, 60, 90, 90)
        mask = np.ones((100, 100), dtype=np.uint8)
        assert s.replace_box_with_mask(0, snapshot, mask, 1) is True
        assert len(s.boxes) == 1  # second box removed
        assert len(s.masks) == 1

    def test_clear_emits_changed(self):
        s = self._store()
        s.add_box(0, 0, 10, 10, 0)
        s.add_mask(np.ones((10, 10), dtype=np.uint8), 0)
        s.clear()
        assert s.total_count == 0
        assert s.last_kind == ""

    def test_apply_yolo_predictions_replace(self):
        s = self._store()
        s.add_box(0, 0, 10, 10, 0)
        masks = [np.ones((10, 10), dtype=np.uint8)]
        s.apply_yolo_predictions(masks, [1], [(20, 20, 40, 40)], [0], replace=True)
        assert len(s.masks) == 1
        assert len(s.boxes) == 1

    def test_apply_yolo_predictions_append(self):
        s = self._store()
        s.add_box(0, 0, 10, 10, 0)
        s.apply_yolo_predictions([], [], [(20, 20, 40, 40)], [1], replace=False)
        assert len(s.boxes) == 2

    def test_find_at_topmost_box_over_mask(self):
        """Boxes always take priority over masks at same location."""
        s = self._store()
        s.add_mask(np.ones((100, 100), dtype=np.uint8), 0)
        s.add_box(10, 10, 90, 90, 1)
        kind, idx = s.find_at(50, 50)
        assert kind == "box"

    def test_find_at_topmost_later_mask(self):
        """Among masks, later ones are on top."""
        s = self._store()
        m1 = np.ones((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m2[40:60, 40:60] = 1
        s.add_mask(m1, 0)
        s.add_mask(m2, 1)
        kind, idx = s.find_at(50, 50)
        assert kind == "mask"
        assert idx == 1  # second mask (on top)
