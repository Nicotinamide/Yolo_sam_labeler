"""YOLO SAM Labeler — PyQt5 annotation tool combining SAM segmentation and YOLO detection.

Version: 0.2.0
"""

import os
import sys

# ---------------------------------------------------------------------------
# Qt platform plugin fix (Jetson / aarch64)
# OpenCV overrides QT_QPA_PLATFORM_PLUGIN_PATH on import, causing xcb load failure.
# This must run AFTER cv2 import but BEFORE QApplication creation.
# ---------------------------------------------------------------------------
_fixed = False


def _fix_qt_plugins():
    """Fix Qt platform plugin path after cv2 overrides it.

    Must be called AFTER ``import cv2`` (which sets QT_QPA_PLATFORM_PLUGIN_PATH
    to OpenCV's bundled plugins) but BEFORE QApplication is created.
    """
    global _fixed
    if _fixed:
        return
    current = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if "/cv2/qt/plugins" not in current:
        # Either not set, or user pointed it to a real path — leave alone
        _fixed = True
        return
    # OpenCV hijacked the path — point back to system Qt plugins
    for candidate in (
        "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms",
        "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms",
        "/usr/lib/qt5/plugins/platforms",
        "/usr/lib/qt6/plugins/platforms",
    ):
        if os.path.isdir(candidate):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = candidate
            _fixed = True
            return


__version__ = "0.2.0"
