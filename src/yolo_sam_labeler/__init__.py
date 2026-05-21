"""YOLO SAM Labeler — PyQt5 annotation tool combining SAM segmentation and YOLO detection.

Version: 0.3.0
"""

import os
import sys

__version__ = "0.3.0"

# ---------------------------------------------------------------------------
# Qt platform plugin path fix
#
# The Qt plugin discovery path (env var QT_QPA_PLATFORM_PLUGIN_PATH) is often
# hijacked before our app starts by:
#   1. OpenCV — sets it to .../cv2/qt/plugins on import (Jetson/aarch64)
#   2. Conda — sets it to $CONDA_PREFIX/bin/platforms when a conda env is active
#
# Both cases break the venv's own PyQt5 (it has its own Qt5/plugins). We
# detect the hijack and point the var back to:
#   a. The venv's own PyQt5/Qt5/plugins/platforms (preferred — matches the
#      Qt that PyQt5 was built against)
#   b. System Qt plugins (Jetson with apt python3-pyqt5, x86 fallback)
# ---------------------------------------------------------------------------
_fixed = False


def _venv_pyqt5_plugin_dir() -> str | None:
    """Return the platforms/ dir inside the active venv's PyQt5, if importable."""
    try:
        import PyQt5  # noqa: F401  (we only need its file path)
    except ImportError:
        return None
    pyqt_dir = os.path.dirname(os.path.abspath(PyQt5.__file__))
    for candidate in (
        os.path.join(pyqt_dir, "Qt5", "plugins", "platforms"),
        os.path.join(pyqt_dir, "Qt", "plugins", "platforms"),
        os.path.join(pyqt_dir, "plugins", "platforms"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


def _system_qt_plugin_dir() -> str | None:
    """Return a system-wide Qt platforms/ dir (Jetson apt PyQt5, etc.)."""
    for candidate in (
        "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms",
        "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
        "/usr/lib/qt5/plugins/platforms",
        "/usr/lib/qt6/plugins/platforms",
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


def _is_hijacked(path: str) -> bool:
    """Detect known hijackers (cv2, conda) that point Qt to broken plugin dirs."""
    if not path:
        return False
    # cv2 ships its own Qt 5.15.x — different patch from PyQt5's Qt
    if "/cv2/qt/plugins" in path:
        return True
    # Conda activates and sets QT_QPA_PLATFORM_PLUGIN_PATH to its bin/platforms,
    # which doesn't match the venv's PyQt5
    if "/anaconda" in path or "/miniconda" in path or "/conda/envs/" in path:
        return True
    return False


def _fix_qt_plugins():
    """Repair QT_QPA_PLATFORM_PLUGIN_PATH if cv2 or conda hijacked it.

    Must run BEFORE QApplication creation (and ideally after ``import cv2``).
    Prefers the venv's PyQt5 plugins, then falls back to system Qt.
    """
    global _fixed
    if _fixed:
        return

    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")

    # If unset or hijacked, replace it
    if not current or _is_hijacked(current):
        target = _venv_pyqt5_plugin_dir() or _system_qt_plugin_dir()
        if target:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = target

    _fixed = True
