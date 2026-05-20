"""Tests for data model (no Qt required)."""
import numpy as np
import pytest

from yolo_sam_labeler.models import Box, Mask, ClassRegistry, AnnotationStore


class TestBox:
    def test_creation(self):
        b = Box(class_id=3, x1=10, y1=20, x2=100, y2=200)
        assert b.class_id == 3
        assert b.width == 90
        assert b.height == 180
        assert b.center == (55.0, 110.0)

    def test_contains(self):
        b = Box(class_id=0, x1=10, y1=10, x2=50, y2=50)
        assert b.contains(30, 30)
        assert not b.contains(5, 5)
        assert not b.contains(60, 60)


class TestMask:
    def test_creation(self):
        data = np.zeros((100, 100), dtype=np.uint8)
        data[20:40, 30:60] = 1
        m = Mask(class_id=2, data=data)
        assert m.class_id == 2
        assert m.data.shape == (100, 100)

    def test_contains(self):
        data = np.zeros((100, 100), dtype=np.uint8)
        data[20:40, 30:60] = 1
        m = Mask(class_id=0, data=data)
        assert m.contains(35, 30)
        assert not m.contains(10, 10)


class TestClassRegistry:
    def test_add(self):
        cr = ClassRegistry()
        cid = cr.add("nut")
        assert cid == 0
        assert cr.name(0) == "nut"
        assert cr.sorted_ids() == [0]

    def test_add_multiple(self):
        cr = ClassRegistry()
        cr.add("a")
        cr.add("b")
        assert cr.sorted_ids() == [0, 1]
        assert cr.name(0) == "a"
        assert cr.name(1) == "b"

    def test_remove(self):
        cr = ClassRegistry({"0": 0, "1": 1}.items())  # no, wait...
        # initial is dict[int, str]
        cr = ClassRegistry({0: "nut", 1: "screw"})
        assert cr.remove(0)
        assert 0 not in cr
        assert cr.remove(99) is False

    def test_rename(self):
        cr = ClassRegistry({0: "nut"})
        assert cr.rename(0, "bolt")
        assert cr.name(0) == "bolt"


class TestAnnotationStore:
    def _make_store(self):
        cr = ClassRegistry({0: "nut", 1: "screw"})
        return AnnotationStore(cr)

    def test_add_mask(self):
        s = self._make_store()
        data = np.ones((100, 100), dtype=np.uint8)
        s.add_mask(data, 0)
        assert s.total_count == 1
        assert len(s.masks) == 1
        assert s.masks[0].class_id == 0

    def test_add_box(self):
        s = self._make_store()
        s.add_box(10, 20, 100, 200, 1)
        assert s.total_count == 1
        assert len(s.boxes) == 1
        assert s.boxes[0].class_id == 1

    def test_find_at_returns_topmost(self):
        s = self._make_store()
        data = np.ones((100, 100), dtype=np.uint8)
        s.add_mask(data, 0)           # mask at index 0
        s.add_box(10, 10, 50, 50, 1)  # box at index 0
        kind, idx = s.find_at(30, 30)
        assert kind == "box"
        assert idx == 0

    def test_delete_at(self):
        s = self._make_store()
        data = np.ones((100, 100), dtype=np.uint8)
        s.add_mask(data, 0)
        s.add_box(10, 10, 50, 50, 1)
        assert s.delete_at(30, 30)  # deletes box
        assert len(s.boxes) == 0
        assert len(s.masks) == 1

    def test_undo_last(self):
        s = self._make_store()
        s.add_box(10, 20, 100, 200, 1)
        s.add_mask(np.ones((100, 100), dtype=np.uint8), 0)
        assert s.total_count == 2
        assert s.undo_last()  # undo mask (last added)
        assert s.total_count == 1
        assert len(s.masks) == 0

    def test_relabel(self):
        s = self._make_store()
        data = np.ones((100, 100), dtype=np.uint8)
        s.add_mask(data, 0)
        assert s.relabel("mask", 0, 1)
        assert s.masks[0].class_id == 1
        assert not s.relabel("mask", 99, 0)

    def test_clear(self):
        s = self._make_store()
        s.add_box(10, 20, 100, 200, 1)
        s.add_mask(np.ones((100, 100), dtype=np.uint8), 0)
        s.clear()
        assert s.total_count == 0
