"""Lightweight smoke tests for MainWindow's label-dir orchestration.

Runs under offscreen Qt; we instantiate ``MainWindow`` with empty args, then
poke ``_seed_label_dirs`` / ``_apply_label_dir_choice`` / ``_save_current``
in isolation to validate the decision tree without driving a real event loop.
"""
import os
import sys

import pytest

# Skip if PyQt5 isn't available.
pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QSettings, QCoreApplication
from PyQt5.QtWidgets import QApplication

from yolo_sam_labeler.app import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Wipe the real QSettings used by MainWindow before each test.

    MainWindow creates ``QSettings("yolo-sam-labeler", "yolo-sam-labeler")``
    directly, so we can't redirect it via setPath; instead we clear it.
    """
    s = QSettings("yolo-sam-labeler", "yolo-sam-labeler")
    s.clear()
    s.sync()
    yield
    s.clear()
    s.sync()


@pytest.fixture
def main_window(qapp, isolated_settings):
    win = MainWindow()
    yield win
    win.close()


def _write_detect(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")


def _write_seg(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n")


# ===========================================================================
# _seed_label_dirs decision tree
# ===========================================================================


class TestSeedLabelDirs:
    def test_branch_1_cli_arg_wins_over_legacy(self, main_window, tmp_path):
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        _write_detect(cli_dir / "img.txt")
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        _write_seg(legacy / "img.txt")
        main_window._seed_label_dirs(
            label_dir_arg=str(cli_dir),
            stored_seg_dir="",
            stored_detect_dir="",
            stored_legacy_label_dir=str(legacy),
        )
        # CLI dir was sniffed as detect → detect_dir = cli, seg_dir = sibling
        assert main_window.store.detect_dir == str(cli_dir)
        assert main_window.store.seg_dir == str(cli_dir) + "_seg"

    def test_branch_2_new_keys_with_only_seg(self, main_window, tmp_path):
        seg = tmp_path / "seg"
        seg.mkdir()
        main_window._seed_label_dirs(
            label_dir_arg="",
            stored_seg_dir=str(seg),
            stored_detect_dir="",
            stored_legacy_label_dir="",
        )
        # detect_dir auto-seeded from sibling rule.
        assert main_window.store.seg_dir == str(seg)
        assert main_window.store.detect_dir == str(seg) + "_detect"

    def test_branch_3_legacy_migration_clears_key(
        self, main_window, tmp_path
    ):
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        _write_detect(legacy / "img.txt")
        main_window.settings.setValue("paths/label_dir", str(legacy))
        main_window._seed_label_dirs(
            label_dir_arg="",
            stored_seg_dir="",
            stored_detect_dir="",
            stored_legacy_label_dir=str(legacy),
        )
        assert main_window.store.detect_dir == str(legacy)
        assert main_window.store.seg_dir == str(legacy) + "_seg"
        # Legacy key was removed.
        assert main_window.settings.value("paths/label_dir") in (None, "")

    def test_branch_4_all_empty_leaves_store_empty(
        self, main_window
    ):
        main_window._seed_label_dirs(
            label_dir_arg="",
            stored_seg_dir="",
            stored_detect_dir="",
            stored_legacy_label_dir="",
        )
        assert main_window.store.seg_dir == ""
        assert main_window.store.detect_dir == ""


# ===========================================================================
# _save_current with no dirs
# ===========================================================================


class TestSaveWithoutDirs:
    def test_no_image_returns_false(self, main_window):
        assert main_window._save_current(silent=True) is False

    def test_silent_no_dirs_does_not_crash(self, main_window, qapp):
        # No image loaded → returns False silently.
        main_window.store.seg_dir = ""
        main_window.store.detect_dir = ""
        assert main_window._save_current(silent=True) is False


# ===========================================================================
# _apply_label_dir_choice for the three kinds
# ===========================================================================


class TestApplyLabelDirChoice:
    def test_kind_seg_seeds_detect_sibling(self, main_window, tmp_path):
        seg = tmp_path / "labels"
        seg.mkdir()
        main_window.store.seg_dir = ""
        main_window.store.detect_dir = ""
        main_window._apply_label_dir_choice(str(seg), kind="seg")
        assert main_window.store.seg_dir == str(seg)
        assert main_window.store.detect_dir == str(seg) + "_detect"

    def test_kind_detect_seeds_seg_sibling(self, main_window, tmp_path):
        det = tmp_path / "labels"
        det.mkdir()
        main_window.store.seg_dir = ""
        main_window.store.detect_dir = ""
        main_window._apply_label_dir_choice(str(det), kind="detect")
        assert main_window.store.detect_dir == str(det)
        assert main_window.store.seg_dir == str(det) + "_seg"

    def test_kind_seg_then_detect_keeps_both(self, main_window, tmp_path):
        seg = tmp_path / "seg"
        det = tmp_path / "det"
        seg.mkdir()
        det.mkdir()
        main_window.store.seg_dir = ""
        main_window.store.detect_dir = ""
        main_window._apply_label_dir_choice(str(seg), kind="seg")
        # detect was auto-seeded; explicit pick should override it.
        main_window._apply_label_dir_choice(str(det), kind="detect")
        assert main_window.store.seg_dir == str(seg)
        assert main_window.store.detect_dir == str(det)



# ===========================================================================
# image_dir switch clears in-project label dirs
# ===========================================================================


class TestImageDirSwitchClears:
    def test_seg_dir_inside_old_image_dir_gets_cleared(
        self, main_window, tmp_path
    ):
        old_proj = tmp_path / "old"
        old_proj.mkdir()
        labels = old_proj / "labels"
        labels.mkdir()
        # Put store inside the old project.
        main_window.image_dir = str(old_proj)
        main_window.store.seg_dir = str(labels)
        main_window.store.detect_dir = ""
        # External path should survive.
        main_window.store.detect_dir = str(tmp_path / "external_det")

        new_proj = tmp_path / "new"
        new_proj.mkdir()
        main_window._load_directory(str(new_proj))

        # seg_dir was a sub-path of old → cleared.
        assert main_window.store.seg_dir == "" or not main_window._is_subpath(
            main_window.store.seg_dir, str(old_proj)
        )
        # detect_dir was external → kept.
        assert main_window.store.detect_dir == str(tmp_path / "external_det")
