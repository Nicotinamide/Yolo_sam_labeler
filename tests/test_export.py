"""Tests for the export dialog packaging logic.

We only test the pure-logic side: filename suggestions, directory selection,
file collection, and archive writing. The QDialog UI itself is exercised
elsewhere via app smoke tests.
"""

from __future__ import annotations

import os
import re
import tarfile
import zipfile

import pytest

from yolo_sam_labeler.export_dialog import (
    ExportConfig,
    ExportWorker,
    suggest_directory,
    suggest_filename,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labels(dirpath: str, stems: list[str], *, empty: list[str] = ()) -> None:
    """Create label files with realistic content. Mark some as empty."""
    os.makedirs(dirpath, exist_ok=True)
    for stem in stems:
        path = os.path.join(dirpath, f"{stem}.txt")
        if stem in empty:
            open(path, "w", encoding="utf-8").close()
        else:
            with open(path, "w", encoding="utf-8") as f:
                # Plausible YOLO seg line.
                f.write("0 0.1 0.1 0.5 0.1 0.5 0.5\n")


def _make_classes(dirpath: str, names: list[str]) -> str:
    """Drop a classes.txt file and return its path."""
    path = os.path.join(dirpath, "classes.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(names))
    return path


def _make_images(dirpath: str, stems: list[str]) -> None:
    """Create stub jpg files (zero bytes is fine — collection only checks ext)."""
    os.makedirs(dirpath, exist_ok=True)
    for stem in stems:
        with open(os.path.join(dirpath, f"{stem}.jpg"), "wb") as f:
            f.write(b"\xFF\xD8\xFF\xE0")  # tiny JPEG header so size > 0


# ---------------------------------------------------------------------------
# Filename / directory suggestions
# ---------------------------------------------------------------------------


class TestSuggest:
    def test_filename_uses_image_dir_basename(self):
        name = suggest_filename(
            "/data/dataset_v3/images", "/data/dataset_v3/labels", "",
            content="labels", label_source="seg", fmt="zip",
        )
        assert name.startswith("dataset_v3_seg-labels_")
        assert name.endswith(".zip")

    def test_filename_skips_generic_basename(self):
        # When image_dir basename is "images" it falls through to parent.
        name = suggest_filename(
            "/data/myproj/images", "/data/myproj/labels", "",
            content="labels", label_source="seg", fmt="zip",
        )
        assert name.startswith("myproj_")

    def test_filename_uses_label_dir_when_no_image_dir(self):
        name = suggest_filename(
            "", "/home/user/cool-set/labels", "",
            content="labels", label_source="seg", fmt="zip",
        )
        assert name.startswith("cool-set_")

    def test_filename_dataset_content(self):
        name = suggest_filename(
            "/data/proj/images", "/data/proj/labels", "",
            content="dataset", label_source="seg", fmt="targz",
        )
        assert "_dataset-seg_" in name
        assert name.endswith(".tar.gz")

    def test_filename_both_label_sources(self):
        name = suggest_filename(
            "/data/proj/images", "/data/proj/seg", "/data/proj/det",
            content="labels", label_source="both", fmt="zip",
        )
        # When both are packed we drop the seg/detect qualifier.
        assert "_labels_" in name
        assert "seg-labels" not in name and "detect-labels" not in name

    def test_filename_timestamp_is_yyyymmdd_hhmm(self):
        name = suggest_filename(
            "/p/img", "/p/lbl", "",
            content="labels", label_source="seg", fmt="zip",
        )
        m = re.search(r"_(\d{8}-\d{4})\.zip$", name)
        assert m is not None, name

    def test_filename_special_chars_are_slugged(self):
        name = suggest_filename(
            "/data/proj name/images", "", "",
            content="labels", label_source="seg", fmt="zip",
        )
        assert "proj_name" in name or "proj name" not in name

    def test_directory_falls_back_to_image_parent(self, tmp_path):
        img = tmp_path / "myproj" / "images"
        img.mkdir(parents=True)
        result = suggest_directory(str(img), "", "", remembered="")
        assert result == str(tmp_path / "myproj")

    def test_directory_prefers_remembered(self, tmp_path):
        img = tmp_path / "p" / "images"
        img.mkdir(parents=True)
        remembered = tmp_path / "elsewhere"
        remembered.mkdir()
        result = suggest_directory(str(img), "", "", remembered=str(remembered))
        assert result == str(remembered)

    def test_directory_falls_back_home_when_nothing(self):
        result = suggest_directory("", "", "", remembered="")
        assert result == os.path.expanduser("~")


# ---------------------------------------------------------------------------
# Worker — file collection
# ---------------------------------------------------------------------------


class TestCollect:
    def _make_cfg(self, **overrides) -> ExportConfig:
        defaults = dict(
            content="labels",
            label_source="seg",
            archive_format="zip",
            exclude_meta=True,
            skip_empty_txt=True,
            output_path="/tmp/out.zip",
            image_dir="",
            seg_dir="",
            detect_dir="",
            classes_file="",
        )
        defaults.update(overrides)
        return ExportConfig(**defaults)

    def test_seg_only_collects_label_txt(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["001", "002", "003"])
        cfg = self._make_cfg(seg_dir=str(seg))
        entries = ExportWorker(cfg)._collect_entries()
        names = sorted(arc for arc, _ in entries)
        assert names == ["labels/001.txt", "labels/002.txt", "labels/003.txt"]

    def test_skip_empty_drops_zero_byte_files(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["001", "002", "003"], empty=["002"])
        cfg = self._make_cfg(seg_dir=str(seg), skip_empty_txt=True)
        entries = ExportWorker(cfg)._collect_entries()
        names = [arc for arc, _ in entries]
        assert "labels/002.txt" not in names
        assert len(names) == 2

    def test_keep_empty_when_flag_off(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["001", "002"], empty=["002"])
        cfg = self._make_cfg(seg_dir=str(seg), skip_empty_txt=False)
        entries = ExportWorker(cfg)._collect_entries()
        assert any(arc.endswith("002.txt") for arc, _ in entries)

    def test_classes_file_added_at_root(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["001"])
        classes = _make_classes(str(seg), ["foo", "bar"])
        cfg = self._make_cfg(seg_dir=str(seg), classes_file=classes)
        entries = ExportWorker(cfg)._collect_entries()
        arcnames = [arc for arc, _ in entries]
        # Appears at archive root, not nested under labels/.
        assert "classes.txt" in arcnames
        assert "labels/classes.txt" not in arcnames
        assert arcnames.count("classes.txt") == 1
        assert "labels/001.txt" in arcnames

    def test_classes_in_label_dir_not_double_packed(self, tmp_path):
        """classes.txt sitting inside the label dir must not also appear under labels/."""
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["001", "002"])
        classes = _make_classes(str(seg), ["foo"])
        cfg = self._make_cfg(seg_dir=str(seg), classes_file=classes)
        entries = ExportWorker(cfg)._collect_entries()
        arcnames = [arc for arc, _ in entries]
        # The classes.txt sitting in seg/ should not be picked up as a label.
        assert "labels/classes.txt" not in arcnames
        # Only the root-level entry exists.
        assert arcnames.count("classes.txt") == 1

    def test_meta_dir_excluded_by_default(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["001"])
        meta = seg / ".meta"
        meta.mkdir()
        (meta / "manifest.json").write_text("{}", encoding="utf-8")
        cfg = self._make_cfg(seg_dir=str(seg), exclude_meta=True)
        entries = ExportWorker(cfg)._collect_entries()
        names = [arc for arc, _ in entries]
        assert all(".meta" not in n for n in names)

    def test_dataset_includes_images_and_labels(self, tmp_path):
        img = tmp_path / "images"
        seg = tmp_path / "labels"
        _make_images(str(img), ["001", "002"])
        _make_labels(str(seg), ["001", "002"])
        cfg = self._make_cfg(
            content="dataset", image_dir=str(img), seg_dir=str(seg))
        entries = ExportWorker(cfg)._collect_entries()
        names = sorted(arc for arc, _ in entries)
        assert "images/001.jpg" in names
        assert "images/002.jpg" in names
        assert "labels/001.txt" in names
        assert "labels/002.txt" in names

    def test_both_source_uses_distinct_arc_dirs(self, tmp_path):
        seg = tmp_path / "seg"
        det = tmp_path / "det"
        _make_labels(str(seg), ["001"])
        _make_labels(str(det), ["001"])
        cfg = self._make_cfg(
            label_source="both", seg_dir=str(seg), detect_dir=str(det))
        entries = ExportWorker(cfg)._collect_entries()
        names = sorted(arc for arc, _ in entries)
        assert "labels_seg/001.txt" in names
        assert "labels_detect/001.txt" in names

    def test_shared_layout_does_not_double_pack(self, tmp_path):
        shared = tmp_path / "labels"
        _make_labels(str(shared), ["001", "002"])
        cfg = self._make_cfg(
            label_source="shared", seg_dir=str(shared), detect_dir=str(shared))
        entries = ExportWorker(cfg)._collect_entries()
        names = sorted(arc for arc, _ in entries)
        # Each file appears exactly once.
        assert names == ["labels/001.txt", "labels/002.txt"]

    def test_images_only_skips_labels(self, tmp_path):
        img = tmp_path / "images"
        seg = tmp_path / "labels"
        _make_images(str(img), ["001"])
        _make_labels(str(seg), ["001"])
        cfg = self._make_cfg(
            content="images", image_dir=str(img), seg_dir=str(seg))
        entries = ExportWorker(cfg)._collect_entries()
        names = [arc for arc, _ in entries]
        assert names == ["images/001.jpg"]


# ---------------------------------------------------------------------------
# Worker — archive writing
# ---------------------------------------------------------------------------


class TestArchive:
    def test_zip_roundtrip(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["a", "b"])
        out = tmp_path / "out.zip"
        cfg = ExportConfig(
            content="labels", label_source="seg", archive_format="zip",
            exclude_meta=True, skip_empty_txt=True,
            output_path=str(out),
            image_dir="", seg_dir=str(seg), detect_dir="", classes_file="",
        )
        worker = ExportWorker(cfg)
        worker.run()
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            names = sorted(zf.namelist())
        assert names == ["labels/a.txt", "labels/b.txt"]

    def test_targz_roundtrip(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), ["a"])
        out = tmp_path / "out.tar.gz"
        cfg = ExportConfig(
            content="labels", label_source="seg", archive_format="targz",
            exclude_meta=True, skip_empty_txt=True,
            output_path=str(out),
            image_dir="", seg_dir=str(seg), detect_dir="", classes_file="",
        )
        worker = ExportWorker(cfg)
        worker.run()
        assert out.exists()
        with tarfile.open(out, "r:gz") as tf:
            names = sorted(tf.getnames())
        assert names == ["labels/a.txt"]

    def test_no_files_returns_failure(self, tmp_path):
        empty = tmp_path / "labels"
        empty.mkdir()
        out = tmp_path / "out.zip"
        cfg = ExportConfig(
            content="labels", label_source="seg", archive_format="zip",
            exclude_meta=True, skip_empty_txt=True,
            output_path=str(out),
            image_dir="", seg_dir=str(empty), detect_dir="", classes_file="",
        )
        worker = ExportWorker(cfg)
        captured: list[tuple[bool, str]] = []
        worker.finished.connect(lambda ok, msg: captured.append((ok, msg)))
        worker.run()
        assert captured and captured[0][0] is False
        assert not out.exists()

    def test_cancel_during_collection_removes_partial(self, tmp_path):
        seg = tmp_path / "labels"
        _make_labels(str(seg), [f"img{i:03d}" for i in range(20)])
        out = tmp_path / "out.zip"
        cfg = ExportConfig(
            content="labels", label_source="seg", archive_format="zip",
            exclude_meta=True, skip_empty_txt=True,
            output_path=str(out),
            image_dir="", seg_dir=str(seg), detect_dir="", classes_file="",
        )
        worker = ExportWorker(cfg)
        # Pre-cancel: the run loop checks _cancelled per file.
        worker.cancel()
        captured: list[tuple[bool, str]] = []
        worker.finished.connect(lambda ok, msg: captured.append((ok, msg)))
        worker.run()
        assert captured and captured[0][0] is False
        assert "取消" in captured[0][1]
        assert not out.exists()
