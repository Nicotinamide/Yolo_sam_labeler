"""YOLO format read/write and image loading utilities.

Supported formats:
  - YOLO segmentation:  class_id x1_norm y1_norm x2_norm y2_norm ...
  - YOLO detection:     class_id cx_norm cy_norm w_norm h_norm
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import Box, Mask

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------


def load_class_names(path: str | Path) -> dict[int, str]:
    """Load class names from a YOLO-style classes.txt file.

    Supported line formats:
      - ``person``              -> id is the non-empty line index
      - ``0 person`` / ``0: person`` -> explicit id
    """
    class_path = Path(path)
    if not class_path.is_file():
        return {}

    names: dict[int, str] = {}
    next_id = 0
    try:
        raw_lines = class_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        cid: int | None = None
        name = line
        if ":" in line:
            left, right = line.split(":", 1)
            try:
                cid = int(left.strip())
                name = right.strip()
            except ValueError:
                cid = None
        if cid is None:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    cid = int(parts[0])
                    name = parts[1].strip()
                except ValueError:
                    cid = None
        if cid is None:
            cid = next_id

        if name:
            names[cid] = name
            next_id = max(next_id, cid + 1)

    return names


def save_class_names(path: str | Path, class_names: dict[int, str]):
    """Save class names in a format that stays compatible with YOLO tools."""
    class_path = Path(path)
    class_path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(int(cid) for cid in class_names.keys())
    if ids == list(range(len(ids))):
        lines = [str(class_names[cid]) for cid in ids]
    else:
        lines = [f"{cid} {class_names[cid]}" for cid in ids]
    class_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def scan_images(directory: str) -> list[str]:
    """Return sorted list of absolute image paths in a directory."""
    if not directory or not os.path.isdir(directory):
        return []
    paths: list[str] = []
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.abspath(os.path.join(directory, name)))
    return paths


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    """Load image as BGR uint8, robust to non-ASCII paths and EXIF rotation.

    ``cv2.imread`` chokes on non-ASCII paths on Windows and ignores EXIF
    orientation. We read the raw bytes and decode with ``cv2.imdecode``, then
    apply EXIF orientation when present.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return _apply_exif_orientation(path, img)


def _apply_exif_orientation(path: str, bgr: np.ndarray) -> np.ndarray:
    """Apply EXIF orientation tag to a BGR image when available.

    Falls back to the original image if PIL is missing or the file has no EXIF.
    """
    try:
        from PIL import Image, ExifTags  # type: ignore
    except ImportError:
        return bgr
    try:
        with Image.open(path) as im:
            exif = im.getexif() if hasattr(im, "getexif") else None
            if not exif:
                return bgr
            orient_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orient_key is None:
                return bgr
            value = exif.get(orient_key, 1)
    except Exception:
        return bgr

    # Orientation values per EXIF spec: 1 normal, 3 180°, 6 90°CW, 8 90°CCW.
    if value == 3:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if value == 6:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if value == 8:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if value == 2:
        return cv2.flip(bgr, 1)
    if value == 4:
        return cv2.flip(bgr, 0)
    if value == 5:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), 1)
    if value == 7:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    return bgr


def load_image_rgb(path: str) -> Optional[np.ndarray]:
    """Load image as RGB uint8 (needed by SAM). Returns None on failure."""
    bgr = load_image_bgr(path)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# YOLO segmentation format
# ---------------------------------------------------------------------------


def masks_to_yolo_lines(masks: list[Mask], w_img: int, h_img: int) -> list[str]:
    """Convert masks to YOLO segmentation text lines."""
    lines: list[str] = []
    for m in masks:
        contours, _ = cv2.findContours(
            m.data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 50:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        polygon = contour.reshape(-1, 2)
        if len(polygon) < 3:
            continue
        parts = [str(m.class_id)]
        for x, y in polygon:
            parts.append(f"{x / w_img:.6f}")
            parts.append(f"{y / h_img:.6f}")
        lines.append(" ".join(parts))
    return lines


def load_masks_from_txt(label_path: str, w_img: int, h_img: int) -> list[Mask]:
    """Read YOLO segmentation .txt into Mask list."""
    masks: list[Mask] = []
    if not label_path or not os.path.isfile(label_path):
        return masks
    try:
        with open(label_path, encoding="utf-8") as f:
            raw = f.readlines()
    except OSError:
        return masks
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(coords) % 2 != 0:
            continue
        pts = np.array([
            [int(coords[i] * w_img), int(coords[i + 1] * h_img)]
            for i in range(0, len(coords), 2)
        ], dtype=np.int32)
        mask_2d = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(mask_2d, [pts], 1)
        if int(mask_2d.sum()) < 30:
            continue
        masks.append(Mask(class_id=class_id, data=mask_2d))
    return masks


# ---------------------------------------------------------------------------
# YOLO detection format
# ---------------------------------------------------------------------------


def boxes_to_yolo_lines(boxes: list[Box], w_img: int, h_img: int) -> list[str]:
    """Convert boxes to YOLO detection text lines."""
    lines: list[str] = []
    for b in boxes:
        cx = ((b.x1 + b.x2) / 2) / w_img
        cy = ((b.y1 + b.y2) / 2) / h_img
        bw = b.width / w_img
        bh = b.height / h_img
        lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def load_boxes_from_txt(label_path: str, w_img: int, h_img: int) -> list[Box]:
    """Read YOLO detection .txt into Box list."""
    boxes: list[Box] = []
    if not label_path or not os.path.isfile(label_path):
        return boxes
    try:
        with open(label_path, encoding="utf-8") as f:
            raw = f.readlines()
    except OSError:
        return boxes
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx = float(parts[1]) * w_img
            cy = float(parts[2]) * h_img
            bw = float(parts[3]) * w_img
            bh = float(parts[4]) * h_img
        except ValueError:
            continue
        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(w_img - 1, int(cx + bw / 2))
        y2 = min(h_img - 1, int(cy + bh / 2))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        boxes.append(Box(class_id=cid, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


# ---------------------------------------------------------------------------
# Unified save/load (dual format)
# ---------------------------------------------------------------------------


def save_labels_seg(store: "AnnotationStore", stem: str, w_img: int, h_img: int):
    """Save masks to {label_dir}/{stem}.txt (YOLO seg format)."""
    if not store.label_dir:
        return
    os.makedirs(store.label_dir, exist_ok=True)
    lines = masks_to_yolo_lines(store.masks, w_img, h_img)
    path = os.path.join(store.label_dir, f"{stem}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_labels_detect(store: "AnnotationStore", stem: str, w_img: int, h_img: int):
    """Save boxes to {label_dir}_detect/{stem}.txt (YOLO detect format)."""
    if not store.label_dir:
        return
    box_dir = store.label_dir + "_detect"
    lines = boxes_to_yolo_lines(store.boxes, w_img, h_img)
    path = os.path.join(box_dir, f"{stem}.txt")
    if not lines and not os.path.exists(path):
        return
    os.makedirs(box_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_labels(store: "AnnotationStore", stem: str, w_img: int, h_img: int):
    """Save both segmentation and detection labels."""
    save_labels_seg(store, stem, w_img, h_img)
    save_labels_detect(store, stem, w_img, h_img)


def load_labels_for_image(store: "AnnotationStore", image_path: str,
                          w_img: int, h_img: int):
    """Load both segmentation and detection labels for an image into the store."""
    store.masks.clear()
    store.boxes.clear()
    store.last_kind = ""
    if not store.label_dir:
        return
    stem = os.path.splitext(os.path.basename(image_path))[0]
    # segmentation
    seg_path = os.path.join(store.label_dir, f"{stem}.txt")
    store.masks = load_masks_from_txt(seg_path, w_img, h_img)
    # detection
    box_path = os.path.join(store.label_dir + "_detect", f"{stem}.txt")
    store.boxes = load_boxes_from_txt(box_path, w_img, h_img)
    if bool(store.masks) != bool(store.boxes):
        store.last_kind = "mask" if store.masks else "box"
    store.classes.ensure_ids([m.class_id for m in store.masks])
    store.classes.ensure_ids([b.class_id for b in store.boxes])
    store.changed.emit()
