import os

from yolo_sam_labeler.io_utils import (
    load_class_names,
    save_class_names,
    load_labels_for_image,
    save_labels,
)
from yolo_sam_labeler.models import AnnotationStore, ClassRegistry


def test_class_names_round_trip_contiguous(tmp_path):
    path = tmp_path / "classes.txt"
    save_class_names(path, {0: "nut", 1: "bolt"})

    assert path.read_text(encoding="utf-8") == "nut\nbolt\n"
    assert load_class_names(path) == {0: "nut", 1: "bolt"}


def test_class_names_parse_explicit_ids(tmp_path):
    path = tmp_path / "classes.txt"
    path.write_text("2 gear\n5: washer\n", encoding="utf-8")

    assert load_class_names(path) == {2: "gear", 5: "washer"}


def test_load_labels_ensures_unknown_class_ids(tmp_path):
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"")
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "img.txt").write_text(
        "7 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000\n",
        encoding="utf-8",
    )

    classes = ClassRegistry({0: "zero"})
    store = AnnotationStore(classes, str(label_dir))
    load_labels_for_image(store, str(image_path), 100, 100)

    assert 7 in classes
    assert classes.name(7) == "7"
    assert store.masks[0].class_id == 7


def test_save_labels_overwrites_empty_detection_file(tmp_path):
    label_dir = tmp_path / "labels"
    detect_dir = tmp_path / "labels_detect"
    detect_dir.mkdir()
    (detect_dir / "img.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

    store = AnnotationStore(ClassRegistry({0: "zero"}), str(label_dir))
    save_labels(store, "img", 100, 100)

    assert os.path.exists(detect_dir / "img.txt")
    assert (detect_dir / "img.txt").read_text(encoding="utf-8") == ""
