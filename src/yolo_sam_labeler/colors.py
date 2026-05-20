"""Class color palette and color utilities."""

# 10-color BGR palette, cycled by class_id % 10
CLASS_PALETTE: list[tuple[int, int, int]] = [
    (0, 255, 0),      # green
    (255, 64, 0),     # cyan
    (0, 128, 255),    # orange
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (180, 105, 255),  # pink
    (255, 180, 0),    # sky blue
    (128, 255, 128),  # light green
    (203, 192, 255),  # lavender
    (147, 20, 255),   # violet
]


def class_colors_for_ids(class_ids) -> dict[int, tuple[int, int, int]]:
    """Build a stable ``{class_id: BGR tuple}`` mapping.

    The mapping is keyed on the *position* of an id within the sorted set, so
    every consumer (canvas overlay, right-panel list, log labels) renders the
    same id in the same color regardless of how many extra ids exist around it.
    """
    sorted_ids = sorted({int(cid) for cid in class_ids})
    return {cid: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i, cid in enumerate(sorted_ids)}


def class_color(class_id: int, registry=None) -> tuple[int, int, int]:
    """Return BGR tuple for ``class_id`` consistent with :func:`class_colors_for_ids`.

    If ``registry`` (mapping of id→name, or any iterable of ids) is provided we
    align with the sorted-position scheme; otherwise we cycle the palette by
    ``abs(class_id) % len(palette)``.
    """
    cid = int(class_id)
    if registry is not None:
        ids = registry.keys() if hasattr(registry, "keys") else registry
        colors = class_colors_for_ids(ids)
        if cid in colors:
            return colors[cid]
    return CLASS_PALETTE[abs(cid) % len(CLASS_PALETTE)]


def bgr_to_qcolor_tuple(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Swap a BGR tuple to the (r, g, b) order ``QColor`` expects."""
    b, g, r = bgr
    return r, g, b
