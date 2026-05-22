"""Data model: annotations, class registry, and annotation store.

All annotations use original-image pixel coordinates. Display transforms
(zoom, pan, scale) live in canvas.py and never leak into saved data.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


DEFAULT_CLASS_NAMES = {0: "0"}


# ---------------------------------------------------------------------------
# Annotation data classes
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """Bounding box in original image pixel coordinates."""
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Mask:
    """Segmentation mask in original image coordinates.

    data is a 2D uint8 numpy array with shape (H, W), values 0 or 1.
    """
    class_id: int
    data: np.ndarray  # shape (H, W), dtype uint8, binary

    def contains(self, x: int, y: int) -> bool:
        h, w = self.data.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        return bool(self.data[y, x])


# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------


class ClassRegistry(QObject):
    """Bi-directional class_id <-> class_name mapping with change notification.

    Signals:
        classes_changed() — emitted after any add/remove/rename.
    """
    classes_changed = pyqtSignal()

    def __init__(self, initial: dict[int, str] | None = None):
        super().__init__()
        self._names: dict[int, str] = dict(initial) if initial else {}

    # ---- accessors ----

    def name(self, class_id: int) -> str:
        return self._names.get(class_id, str(class_id))

    def sorted_ids(self) -> list[int]:
        return sorted(self._names.keys())

    def max_id(self) -> int:
        return max(self._names.keys()) if self._names else -1

    def __contains__(self, class_id: int) -> bool:
        return class_id in self._names

    def __len__(self) -> int:
        return len(self._names)

    # ---- mutation ----

    def add(self, name: str) -> int:
        """Add a new class. Returns the assigned id."""
        new_id = max(self._names.keys(), default=-1) + 1
        self._names[new_id] = name.strip()
        self.classes_changed.emit()
        return new_id

    def set_names(self, names: dict[int, str]):
        """Replace the complete class map."""
        cleaned = {
            int(cid): str(name).strip() or str(cid)
            for cid, name in names.items()
        }
        self._names = cleaned
        self.classes_changed.emit()

    def ensure(self, class_id: int, name: str | None = None) -> bool:
        """Ensure a class id exists. Returns True if the registry changed."""
        cid = int(class_id)
        if cid in self._names:
            return False
        self._names[cid] = (name or str(cid)).strip() or str(cid)
        self.classes_changed.emit()
        return True

    def ensure_ids(self, class_ids) -> bool:
        """Ensure all ids from an iterable exist. Emits once if anything changed."""
        changed = False
        for cid_raw in class_ids:
            cid = int(cid_raw)
            if cid not in self._names:
                self._names[cid] = str(cid)
                changed = True
        if changed:
            self.classes_changed.emit()
        return changed

    def remove(self, class_id: int) -> bool:
        """Remove a class by id. Returns True if it existed."""
        if class_id not in self._names:
            return False
        del self._names[class_id]
        self.classes_changed.emit()
        return True

    def rename(self, class_id: int, new_name: str) -> bool:
        """Rename an existing class. Returns True if it existed."""
        if class_id not in self._names:
            return False
        self._names[class_id] = new_name.strip()
        self.classes_changed.emit()
        return True

    # ---- serialization ----

    def to_dict(self) -> dict[int, str]:
        return dict(self._names)

    def to_names(self) -> dict[int, str]:
        """Alias for backward compatibility with old code."""
        return dict(self._names)


# ---------------------------------------------------------------------------
# Annotation store
# ---------------------------------------------------------------------------


class AnnotationStore(QObject):
    """Per-image annotation storage with change notification.

    All mutations emit `changed`.  The store does NOT own ClassRegistry
    (classes are shared across images), nor does it own the image data.
    """

    changed = pyqtSignal()

    def __init__(self, classes: ClassRegistry, label_dir: str = ""):
        super().__init__()
        self.classes = classes
        self.masks: list[Mask] = []
        self.boxes: list[Box] = []
        self.image_width: int = 0
        self.image_height: int = 0
        self.last_kind: str = ""  # "mask" | "box" — latest edited annotation kind
        # Two independent label directories.
        # - ``seg_dir`` holds YOLO segmentation files (1 + 2N ≥ 7 tokens/line).
        # - ``detect_dir`` holds YOLO detection files (exactly 5 tokens/line).
        # Either may be ``""`` (no IO for that kind). They may also point at
        # the same physical directory: in that case the loader sniffs each
        # file's actual format (5 vs ≥7 tokens) so masks and boxes route
        # correctly.
        # The legacy ``label_dir`` constructor argument becomes a soft seed:
        # whichever kind the directory is recognized as gets it; if the
        # contents are unknown both fields are seeded so the first save is
        # safe.
        self.seg_dir: str = label_dir
        self.detect_dir: str = label_dir

    # ---- legacy alias -------------------------------------------------
    @property
    def label_dir(self) -> str:
        """Backward-compatible accessor for code that asks for "the" label dir.

        Returns ``seg_dir`` if set, else ``detect_dir``. New code should read
        ``seg_dir`` / ``detect_dir`` directly.
        """
        return self.seg_dir or self.detect_dir

    @label_dir.setter
    def label_dir(self, value: str):
        self.seg_dir = value
        self.detect_dir = value

    # ---- annotation queries ----

    @property
    def total_count(self) -> int:
        return len(self.masks) + len(self.boxes)

    def find_at(self, x: int, y: int) -> tuple[str, int]:
        """Return ("box", idx) or ("mask", idx) or ("", -1) for the topmost
        annotation at pixel (x, y).  Boxes are drawn on top of masks."""
        for i in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[i].contains(x, y):
                return "box", i
        for i in range(len(self.masks) - 1, -1, -1):
            if self.masks[i].contains(x, y):
                return "mask", i
        return "", -1

    # ---- mutation (all emit changed) ----

    def add_mask(self, mask_2d: np.ndarray, class_id: int):
        self.masks.append(Mask(class_id=class_id, data=mask_2d))
        self.last_kind = "mask"
        self.changed.emit()

    def add_box(self, x1: int, y1: int, x2: int, y2: int, class_id: int):
        self.boxes.append(Box(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2))
        self.last_kind = "box"
        self.changed.emit()

    def replace_mask_with_box(self, idx: int, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Replace one mask with its detection box. Returns True if valid."""
        if not 0 <= idx < len(self.masks):
            return False
        mask = self.masks.pop(idx)
        self.boxes.append(Box(class_id=mask.class_id, x1=x1, y1=y1, x2=x2, y2=y2))
        self.last_kind = "box"
        self.changed.emit()
        return True

    def replace_box_with_mask(
        self,
        idx: int,
        snapshot: tuple[int, int, int, int, int],
        mask_2d: np.ndarray,
        class_id: int,
    ) -> bool:
        """Replace one box with a mask. Returns False if the source box changed."""
        box_idx = self._find_box_snapshot(idx, snapshot)
        if box_idx < 0:
            return False
        del self.boxes[box_idx]
        self.masks.append(Mask(class_id=class_id, data=mask_2d))
        self.last_kind = "mask"
        self.changed.emit()
        return True

    def delete_at(self, x: int, y: int) -> bool:
        """Delete the topmost annotation at (x, y). Returns True if found."""
        kind, idx = self.find_at(x, y)
        if kind == "box":
            del self.boxes[idx]
            self._refresh_last_kind()
            self.changed.emit()
            return True
        if kind == "mask":
            del self.masks[idx]
            self._refresh_last_kind()
            self.changed.emit()
            return True
        return False

    def undo_last(self) -> bool:
        """Undo the last-added annotation. Returns True if something was undone."""
        if self.last_kind == "box" and self.boxes:
            self.boxes.pop()
            self._refresh_last_kind()
            self.changed.emit()
            return True
        if self.masks:
            self.masks.pop()
            self._refresh_last_kind()
            self.changed.emit()
            return True
        return False

    def relabel(self, kind: str, idx: int, new_class_id: int) -> bool:
        """Change the class of annotation at (kind, idx). Returns True if valid."""
        if kind == "box" and 0 <= idx < len(self.boxes):
            self.boxes[idx].class_id = new_class_id
            self.changed.emit()
            return True
        if kind == "mask" and 0 <= idx < len(self.masks):
            self.masks[idx].class_id = new_class_id
            self.changed.emit()
            return True
        return False

    def clear(self):
        self.masks.clear()
        self.boxes.clear()
        self.last_kind = ""
        self.changed.emit()

    def apply_yolo_seg_predictions(self, masks: list[np.ndarray],
                                    class_ids: list[int], replace: bool):
        """Import YOLO segmentation results (legacy entry point)."""
        self.apply_yolo_predictions(
            masks=masks,
            mask_class_ids=class_ids,
            boxes=[],
            box_class_ids=[],
            replace=replace,
        )

    def apply_yolo_predictions(
        self,
        masks: list[np.ndarray],
        mask_class_ids: list[int],
        boxes: list[tuple[int, int, int, int]],
        box_class_ids: list[int],
        replace: bool,
    ):
        """Import YOLO predictions (mask + box) into the store atomically."""
        if replace:
            self.masks.clear()
            self.boxes.clear()
        for m, cid in zip(masks, mask_class_ids):
            self.masks.append(Mask(class_id=cid, data=m))
        for (x1, y1, x2, y2), cid in zip(boxes, box_class_ids):
            self.boxes.append(Box(class_id=cid, x1=x1, y1=y1, x2=x2, y2=y2))
        if boxes:
            self.last_kind = "box"
        elif masks:
            self.last_kind = "mask"
        else:
            self._refresh_last_kind()
        self.changed.emit()

    def set_mask(self, mask_2d: np.ndarray, class_id: int):
        """Alias for add_mask — used by SAM prediction callback."""
        self.add_mask(mask_2d, class_id)

    def _refresh_last_kind(self):
        if self.last_kind == "box" and self.boxes:
            return
        if self.last_kind == "mask" and self.masks:
            return
        if self.boxes:
            self.last_kind = "box"
        elif self.masks:
            self.last_kind = "mask"
        else:
            self.last_kind = ""

    def _find_box_snapshot(self, idx: int, snapshot: tuple[int, int, int, int, int]) -> int:
        if 0 <= idx < len(self.boxes):
            box = self.boxes[idx]
            if (box.class_id, box.x1, box.y1, box.x2, box.y2) == snapshot:
                return idx
        for i, box in enumerate(self.boxes):
            if (box.class_id, box.x1, box.y1, box.x2, box.y2) == snapshot:
                return i
        return -1
