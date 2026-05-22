"""YOLO format read/write and image loading utilities.

Supported formats:
  - YOLO segmentation:  class_id x1_norm y1_norm x2_norm y2_norm ...
  - YOLO detection:     class_id cx_norm cy_norm w_norm h_norm
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .models import Box, Mask


# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------


@dataclass
class SaveReport:
    """Structured outcome of a :func:`save_labels` call.

    The UI consumes this to drive logging — it does *not* feed back into the
    next save. Each field is independent: a single save can cause multiple
    flags to fire (e.g. wrote_seg + cleared_detect).
    """

    wrote_seg: bool = False        # seg file was created or rewritten with content
    wrote_detect: bool = False     # detect file was created or rewritten with content
    cleared_seg: bool = False      # seg file existed and was truncated to empty
    cleared_detect: bool = False   # detect file existed and was truncated to empty
    refused_seg: bool = False      # refused to clobber a wrong-format file at seg path
    refused_detect: bool = False   # same for detect path
    skipped_no_dir: list[str] = field(default_factory=list)  # ["seg"] / ["detect"]
    conflict_shared: bool = False  # seg_dir == detect_dir != "" AND both kinds present




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
    """Return sorted list of absolute image paths in a directory.

    Only scans the directory itself, not subdirectories.
    """
    if not directory or not os.path.isdir(directory):
        return []
    paths: list[str] = []
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.abspath(os.path.join(directory, name)))
    return paths


def discover_image_dir(root: str) -> str:
    """Given a root directory, find where the images actually are.

    Logic:
    1. If root itself contains images → return root
    2. If root has a subdirectory containing images (prefer 'images', 'imgs',
       'train', or whichever subdir has the most image files) → return that subdir
    3. Fall back to root
    """
    if not root or not os.path.isdir(root):
        return root or ""

    # Check root first
    for name in os.listdir(root):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            return root  # root has images directly

    # Scan immediate subdirs
    best_dir = ""
    best_count = 0
    preferred_names = {"images", "imgs", "train", "img", "pics"}
    try:
        for entry in os.scandir(root):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            count = sum(
                1 for f in os.scandir(entry.path)
                if f.is_file() and os.path.splitext(f.name)[1].lower() in IMAGE_EXTS
            )
            if count == 0:
                continue
            # Prefer well-known names
            if entry.name.lower() in preferred_names:
                return entry.path
            if count > best_count:
                best_count = count
                best_dir = entry.path
    except PermissionError:
        pass

    return best_dir if best_dir else root


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

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def _seg_yolo_line(parts: list[str]) -> bool:
    """True iff a token list looks like a YOLO seg row (1 + 2N ≥ 7, N ≥ 3)."""
    return len(parts) >= 7 and (len(parts) - 1) % 2 == 0


def _detect_yolo_line(parts: list[str]) -> bool:
    """True iff a token list looks like a YOLO detect row (exactly 5 tokens)."""
    return len(parts) == 5


def _classify_first_data_line(path: str) -> str:
    """Classify a YOLO ``.txt`` by inspecting its first valid data line.

    Returns ``"seg"``, ``"detect"`` or ``""`` (empty / unreadable / unknown).

    YOLO writes one format per file (5 tokens for detect, 1 + 2N ≥ 7 for seg)
    so a single recognized line is enough to classify the entire file.
    """
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    for tok in parts:
                        float(tok)
                except ValueError:
                    continue
                if _detect_yolo_line(parts):
                    return "detect"
                if _seg_yolo_line(parts):
                    return "seg"
    except OSError:
        return ""
    return ""


def _detect_yolo_format(path: str) -> str:
    """Backward-compatible alias for :func:`_classify_first_data_line`."""
    return _classify_first_data_line(path)


def inspect_label_dir_format(label_dir: str, sample_size: int = 20) -> tuple[str, dict]:
    """Sniff a label directory and decide which YOLO format it primarily holds.

    Returns ``(kind, stats)`` where ``kind`` is one of:

    - ``"seg"`` / ``"detect"`` if all (or ≥95% of) format-bearing files agree.
    - ``"empty"`` if the directory has no recognizable YOLO file.
    - ``"mixed"`` if both formats appear without a clear majority.

    ``stats`` contains the raw counts: ``seg``, ``detect``, ``empty``,
    ``scanned``, ``total``.

    Notes:
        - Only the first valid data line of each file is parsed (``5`` tokens
          → detect, ``≥7`` with even-coordinate-pairs → seg). YOLO writes one
          format per file so a single line is enough.
        - Sampling stops after ``sample_size`` *format-bearing* files have
          been classified. The default of ``20`` keeps directory scans cheap
          while staying robust enough for typical projects.
    """
    stats = {"seg": 0, "detect": 0, "empty": 0, "scanned": 0, "total": 0}
    if not label_dir or not os.path.isdir(label_dir):
        return "empty", stats
    try:
        entries = sorted(
            f for f in os.listdir(label_dir)
            if f.endswith(".txt") and f != "classes.txt"
        )
    except OSError:
        return "empty", stats
    stats["total"] = len(entries)
    for name in entries:
        path = os.path.join(label_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        stats["scanned"] += 1
        if size == 0:
            stats["empty"] += 1
            continue
        kind = _classify_first_data_line(path)
        if kind == "seg":
            stats["seg"] += 1
        elif kind == "detect":
            stats["detect"] += 1
        else:
            stats["empty"] += 1
        if stats["seg"] + stats["detect"] >= sample_size:
            break

    decided = stats["seg"] + stats["detect"]
    if decided == 0:
        return "empty", stats
    if stats["seg"] == 0:
        return "detect", stats
    if stats["detect"] == 0:
        return "seg", stats
    if stats["seg"] >= 0.95 * decided:
        return "seg", stats
    if stats["detect"] >= 0.95 * decided:
        return "detect", stats
    return "mixed", stats


# ---------------------------------------------------------------------------
# Unified save / load
# ---------------------------------------------------------------------------


def _safe_overwrite(path: str, new_text: str, expected_kind: str) -> str:
    """Write ``new_text`` to ``path`` without clobbering the wrong format.

    ``expected_kind`` is ``"seg"`` or ``"detect"``. Returns one of:

    - ``"wrote"``: ``new_text`` was non-empty and was written to disk.
    - ``"cleared"``: ``new_text`` was empty and an existing file (same kind or
      unknown) was truncated.
    - ``"skipped_no_target"``: ``new_text`` was empty and the file did not
      exist; nothing was created.
    - ``"refused_format"``: ``new_text`` was empty and an existing file is of
      the *other* kind; the source was left untouched (data-loss guard).
    """
    if new_text:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        return "wrote"
    if not os.path.exists(path):
        return "skipped_no_target"
    existing_kind = _classify_first_data_line(path)
    if existing_kind and existing_kind != expected_kind:
        return "refused_format"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
    except OSError:
        return "skipped_no_target"
    return "cleared"


def save_labels_seg(store: "AnnotationStore", stem: str, w_img: int, h_img: int) -> str:
    """Save masks to ``store.seg_dir/{stem}.txt``.

    Returns the :func:`_safe_overwrite` status string, or ``"skipped_no_dir"``
    when ``store.seg_dir`` is empty.
    """
    target_dir = getattr(store, "seg_dir", "") or ""
    if not target_dir:
        return "skipped_no_dir"
    lines = masks_to_yolo_lines(store.masks, w_img, h_img)
    text = "\n".join(lines)
    path = os.path.join(target_dir, f"{stem}.txt")
    return _safe_overwrite(path, text, expected_kind="seg")


def save_labels_detect(store: "AnnotationStore", stem: str, w_img: int, h_img: int) -> str:
    """Save boxes to ``store.detect_dir/{stem}.txt``.

    Returns the :func:`_safe_overwrite` status string, or ``"skipped_no_dir"``
    when ``store.detect_dir`` is empty.
    """
    target_dir = getattr(store, "detect_dir", "") or ""
    if not target_dir:
        return "skipped_no_dir"
    lines = boxes_to_yolo_lines(store.boxes, w_img, h_img)
    text = "\n".join(lines)
    path = os.path.join(target_dir, f"{stem}.txt")
    return _safe_overwrite(path, text, expected_kind="detect")


def _seg_detect_share_target(store: "AnnotationStore") -> bool:
    """Return True iff seg_dir == detect_dir != "".

    A "shared" directory means the loader and saver have to be careful: a
    single physical file can only hold one format, so writing both kinds at
    the same stem is a conflict.
    """
    seg = getattr(store, "seg_dir", "") or ""
    det = getattr(store, "detect_dir", "") or ""
    if not seg or not det:
        return False
    return os.path.abspath(seg) == os.path.abspath(det)


def save_labels(store: "AnnotationStore", stem: str, w_img: int, h_img: int) -> SaveReport:
    """Save both segmentation and detection labels and return a SaveReport.

    Conflict policy: when ``seg_dir == detect_dir`` and both ``store.masks``
    and ``store.boxes`` are non-empty, only one kind is written this turn —
    the one matching ``store.last_kind`` (defaulting to seg). The skipped
    kind is reported via ``conflict_shared`` so the UI can prompt the user
    to split the layout.
    """
    report = SaveReport()

    seg_dir = getattr(store, "seg_dir", "") or ""
    det_dir = getattr(store, "detect_dir", "") or ""

    has_masks = bool(store.masks)
    has_boxes = bool(store.boxes)
    shared = _seg_detect_share_target(store)
    conflict = shared and has_masks and has_boxes

    skip_kind = ""
    if conflict:
        report.conflict_shared = True
        # Pick which kind to honor this turn. Last edited wins; default seg.
        if store.last_kind == "box":
            skip_kind = "seg"
        else:
            skip_kind = "detect"

    seg_status = (
        "skipped_no_dir"
        if skip_kind == "seg"
        else save_labels_seg(store, stem, w_img, h_img)
    )
    det_status = (
        "skipped_no_dir"
        if skip_kind == "detect"
        else save_labels_detect(store, stem, w_img, h_img)
    )

    _apply_status(report, "seg", seg_status)
    _apply_status(report, "detect", det_status)
    return report


def _apply_status(report: SaveReport, kind: str, status: str):
    """Translate one kind's _safe_overwrite status into ``report`` flags."""
    if status == "wrote":
        if kind == "seg":
            report.wrote_seg = True
        else:
            report.wrote_detect = True
    elif status == "cleared":
        if kind == "seg":
            report.cleared_seg = True
        else:
            report.cleared_detect = True
    elif status == "refused_format":
        if kind == "seg":
            report.refused_seg = True
        else:
            report.refused_detect = True
    elif status == "skipped_no_dir":
        report.skipped_no_dir.append(kind)
    # "skipped_no_target" is benign — neither file existed, nothing to do.


def load_labels_for_image(store: "AnnotationStore", image_path: str,
                          w_img: int, h_img: int) -> list[str]:
    """Load both segmentation and detection labels for an image.

    - ``store.seg_dir/{stem}.txt`` is parsed as seg into ``store.masks``.
    - ``store.detect_dir/{stem}.txt`` is parsed as detect into ``store.boxes``.
    - When ``seg_dir == detect_dir`` (single shared directory) each file is
      classified by content and routed to the matching collection.
    - When the file in ``seg_dir`` actually looks like detect (or vice versa)
      we skip it and return a warning string instead of mis-parsing.

    Returns a list of human-readable warnings (possibly empty). Always emits
    ``store.changed`` exactly once before returning.
    """
    store.masks.clear()
    store.boxes.clear()
    store.last_kind = ""
    warnings: list[str] = []
    seg_dir = getattr(store, "seg_dir", "") or ""
    detect_dir = getattr(store, "detect_dir", "") or ""
    if not seg_dir and not detect_dir:
        store.changed.emit()
        return warnings
    stem = os.path.splitext(os.path.basename(image_path))[0]

    seg_path = os.path.join(seg_dir, f"{stem}.txt") if seg_dir else ""
    detect_path = os.path.join(detect_dir, f"{stem}.txt") if detect_dir else ""

    same_file = (
        seg_path and detect_path
        and os.path.abspath(seg_path) == os.path.abspath(detect_path)
    )
    if same_file:
        kind = _classify_first_data_line(seg_path)
        if kind == "detect":
            store.boxes = load_boxes_from_txt(seg_path, w_img, h_img)
        elif kind == "seg":
            store.masks = load_masks_from_txt(seg_path, w_img, h_img)
        # empty / unknown → both stay empty.
    else:
        if seg_path and os.path.isfile(seg_path):
            kind = _classify_first_data_line(seg_path)
            if kind in ("seg", ""):
                store.masks = load_masks_from_txt(seg_path, w_img, h_img)
            else:
                warnings.append(
                    f"分割目录里发现非分割格式文件，已跳过: {seg_path}"
                )
        if detect_path and os.path.isfile(detect_path):
            kind = _classify_first_data_line(detect_path)
            if kind in ("detect", ""):
                store.boxes = load_boxes_from_txt(detect_path, w_img, h_img)
            else:
                warnings.append(
                    f"检测目录里发现非检测格式文件，已跳过: {detect_path}"
                )

    if bool(store.masks) != bool(store.boxes):
        store.last_kind = "mask" if store.masks else "box"
    store.classes.ensure_ids([m.class_id for m in store.masks])
    store.classes.ensure_ids([b.class_id for b in store.boxes])
    store.changed.emit()
    return warnings


# ---------------------------------------------------------------------------
# Mixed-directory splitter
# ---------------------------------------------------------------------------


def split_mixed_label_dir(
    src: str,
    seg_dst: str = "",
    detect_dst: str = "",
    *,
    dry_run: bool = False,
) -> dict:
    """Walk ``src`` and route each ``.txt`` to ``seg_dst`` / ``detect_dst``.

    Args:
        src: Directory holding a mix of YOLO seg and detect files.
        seg_dst: Where seg files should live afterwards. Defaults to
            ``src + "_seg"``. May equal ``src`` (keep seg in place).
        detect_dst: Where detect files should live afterwards. Defaults to
            ``src + "_detect"``. May equal ``src`` (keep detect in place).
        dry_run: If True, no files are moved; the returned ``stats`` still
            contain what *would* have been moved (useful for previews).

    Returns:
        Dict with counts:

        - ``moved_seg`` / ``moved_detect``: files relocated.
        - ``kept_seg`` / ``kept_detect``: files left where they were because
          ``*_dst == src``.
        - ``skipped_unknown``: files that could not be classified.
        - ``skipped_empty``: zero-byte ``.txt`` files (left in place).
        - ``conflicts``: target path already existed; left untouched.
        - ``seg_dst`` / ``detect_dst``: the resolved destinations.

    The function never deletes data: on conflict (target exists) the source
    file is left in place and counted in ``conflicts``. ``classes.txt`` is
    always preserved.
    """
    stats = {
        "moved_seg": 0,
        "moved_detect": 0,
        "kept_seg": 0,
        "kept_detect": 0,
        "skipped_unknown": 0,
        "skipped_empty": 0,
        "conflicts": 0,
        "seg_dst": "",
        "detect_dst": "",
    }
    if not src or not os.path.isdir(src):
        return stats

    src = os.path.abspath(src)
    seg_dst = os.path.abspath(seg_dst) if seg_dst else src + "_seg"
    detect_dst = os.path.abspath(detect_dst) if detect_dst else src + "_detect"
    stats["seg_dst"] = seg_dst
    stats["detect_dst"] = detect_dst

    try:
        entries = sorted(
            f for f in os.listdir(src)
            if f.endswith(".txt") and f != "classes.txt"
        )
    except OSError:
        return stats

    if not dry_run:
        if seg_dst != src:
            os.makedirs(seg_dst, exist_ok=True)
        if detect_dst != src:
            os.makedirs(detect_dst, exist_ok=True)

    import shutil

    for name in entries:
        src_path = os.path.join(src, name)
        if not os.path.isfile(src_path):
            continue
        try:
            size = os.path.getsize(src_path)
        except OSError:
            continue
        if size == 0:
            stats["skipped_empty"] += 1
            continue
        kind = _classify_first_data_line(src_path)
        if kind == "seg":
            if seg_dst == src:
                stats["kept_seg"] += 1
                continue
            dst_path = os.path.join(seg_dst, name)
            if os.path.exists(dst_path):
                stats["conflicts"] += 1
                continue
            if not dry_run:
                shutil.move(src_path, dst_path)
            stats["moved_seg"] += 1
        elif kind == "detect":
            if detect_dst == src:
                stats["kept_detect"] += 1
                continue
            dst_path = os.path.join(detect_dst, name)
            if os.path.exists(dst_path):
                stats["conflicts"] += 1
                continue
            if not dry_run:
                shutil.move(src_path, dst_path)
            stats["moved_detect"] += 1
        else:
            stats["skipped_unknown"] += 1

    return stats



# ---------------------------------------------------------------------------
# Lazy reconciliation (per-image format check)
# ---------------------------------------------------------------------------


def reconcile_label_file_for_image(store: "AnnotationStore", stem: str) -> dict:
    """Move ``{stem}.txt`` between seg_dir and detect_dir based on actual content.

    The directory-level format is decided by majority on open, but individual
    files that disagree with the directory's role are moved to the correct
    sibling directory the moment we visit that image. This is the "lazy
    reconcile" pattern: no upfront scan, just-in-time relocation.

    Behaviour:

    - When ``seg_dir == detect_dir`` (shared layout) reconciliation is a
      no-op; the loader's per-file sniffing routes content correctly.
    - The candidate target directory is created on demand.
    - If the target path already holds a file, we leave both source and
      destination untouched (counted in ``conflicts``) — never destroy data.

    Returns a dict with counts: ``moved_to_seg`` / ``moved_to_detect`` /
    ``conflicts`` / ``skipped`` / ``actions`` (list of human-readable strings).
    """
    import shutil

    actions: list[str] = []
    result = {
        "moved_to_seg": 0,
        "moved_to_detect": 0,
        "conflicts": 0,
        "skipped": 0,
        "actions": actions,
    }

    seg_dir = getattr(store, "seg_dir", "") or ""
    detect_dir = getattr(store, "detect_dir", "") or ""
    if not seg_dir or not detect_dir:
        return result
    if os.path.abspath(seg_dir) == os.path.abspath(detect_dir):
        return result  # shared layout — nothing to reconcile

    seg_path = os.path.join(seg_dir, f"{stem}.txt")
    detect_path = os.path.join(detect_dir, f"{stem}.txt")

    # Case 1: file in seg_dir but content is detect → move to detect_dir.
    if os.path.isfile(seg_path):
        kind = _classify_first_data_line(seg_path)
        if kind == "detect":
            if os.path.exists(detect_path):
                result["conflicts"] += 1
                actions.append(
                    f"冲突: {stem}.txt 在分割目录里是检测格式，但检测目录已有同名文件，未移动。"
                )
            else:
                try:
                    os.makedirs(detect_dir, exist_ok=True)
                    shutil.move(seg_path, detect_path)
                    result["moved_to_detect"] += 1
                    actions.append(
                        f"已搬移: {stem}.txt → 检测目录 (内容为检测格式)"
                    )
                except OSError as exc:
                    result["skipped"] += 1
                    actions.append(f"搬移失败: {stem}.txt: {exc}")

    # Case 2: file in detect_dir but content is seg → move to seg_dir.
    if os.path.isfile(detect_path):
        kind = _classify_first_data_line(detect_path)
        if kind == "seg":
            new_seg_path = os.path.join(seg_dir, f"{stem}.txt")
            if os.path.exists(new_seg_path):
                result["conflicts"] += 1
                actions.append(
                    f"冲突: {stem}.txt 在检测目录里是分割格式，但分割目录已有同名文件，未移动。"
                )
            else:
                try:
                    os.makedirs(seg_dir, exist_ok=True)
                    shutil.move(detect_path, new_seg_path)
                    result["moved_to_seg"] += 1
                    actions.append(
                        f"已搬移: {stem}.txt → 分割目录 (内容为分割格式)"
                    )
                except OSError as exc:
                    result["skipped"] += 1
                    actions.append(f"搬移失败: {stem}.txt: {exc}")

    return result


# ---------------------------------------------------------------------------
# Empty-file cleanup
# ---------------------------------------------------------------------------


def cleanup_empty_label_files(*dirs: str) -> dict:
    """Remove zero-byte ``.txt`` files from each given directory.

    Skips ``classes.txt`` and any non-``.txt`` files. Returns a dict with the
    total ``removed`` count and a list of ``paths`` that were deleted, so the
    UI can surface a single concise log line.
    """
    removed = 0
    paths: list[str] = []
    seen: set[str] = set()
    for d in dirs:
        if not d:
            continue
        d_abs = os.path.abspath(d)
        if d_abs in seen or not os.path.isdir(d_abs):
            continue
        seen.add(d_abs)
        try:
            entries = os.listdir(d_abs)
        except OSError:
            continue
        for name in entries:
            if not name.endswith(".txt") or name == "classes.txt":
                continue
            p = os.path.join(d_abs, name)
            try:
                if os.path.isfile(p) and os.path.getsize(p) == 0:
                    os.remove(p)
                    removed += 1
                    paths.append(p)
            except OSError:
                continue
    return {"removed": removed, "paths": paths}
