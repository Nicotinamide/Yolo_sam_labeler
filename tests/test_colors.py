"""Tests for colors.py — palette and color mapping consistency."""
from yolo_sam_labeler.colors import (
    CLASS_PALETTE,
    class_colors_for_ids,
    class_color,
    bgr_to_qcolor_tuple,
)


class TestPalette:
    def test_palette_has_10_entries(self):
        assert len(CLASS_PALETTE) == 10

    def test_palette_all_bgr_tuples(self):
        for color in CLASS_PALETTE:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


class TestClassColorsForIds:
    def test_single_id(self):
        colors = class_colors_for_ids([5])
        assert 5 in colors
        assert colors[5] == CLASS_PALETTE[0]  # first sorted → first color

    def test_sorted_position_determines_color(self):
        """Color is assigned by sorted position, not by id value."""
        colors = class_colors_for_ids([10, 3, 7])
        # sorted: [3, 7, 10] → positions 0, 1, 2
        assert colors[3] == CLASS_PALETTE[0]
        assert colors[7] == CLASS_PALETTE[1]
        assert colors[10] == CLASS_PALETTE[2]

    def test_wraps_around_palette(self):
        """More than 10 classes → palette cycles."""
        ids = list(range(12))
        colors = class_colors_for_ids(ids)
        assert colors[0] == CLASS_PALETTE[0]
        assert colors[10] == CLASS_PALETTE[0]  # wraps
        assert colors[11] == CLASS_PALETTE[1]

    def test_empty_ids(self):
        colors = class_colors_for_ids([])
        assert colors == {}

    def test_duplicate_ids_handled(self):
        """Duplicate ids should be deduplicated."""
        colors = class_colors_for_ids([1, 1, 1, 2, 2])
        assert len(colors) == 2

    def test_stable_across_calls(self):
        """Same input → same output."""
        c1 = class_colors_for_ids([5, 2, 8])
        c2 = class_colors_for_ids([5, 2, 8])
        assert c1 == c2


class TestClassColor:
    def test_with_registry(self):
        registry = {0: "a", 5: "b", 10: "c"}
        c = class_color(5, registry)
        expected = class_colors_for_ids(registry.keys())[5]
        assert c == expected

    def test_without_registry(self):
        c = class_color(3)
        assert c == CLASS_PALETTE[3 % len(CLASS_PALETTE)]

    def test_negative_id_without_registry(self):
        c = class_color(-2)
        assert c == CLASS_PALETTE[2]  # abs(-2) % 10


class TestBgrToQcolor:
    def test_swap(self):
        assert bgr_to_qcolor_tuple((255, 128, 0)) == (0, 128, 255)

    def test_identity_gray(self):
        assert bgr_to_qcolor_tuple((100, 100, 100)) == (100, 100, 100)
