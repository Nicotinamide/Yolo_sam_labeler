"""Entry point: python -m yolo_sam_labeler or yolo-sam-label console script."""

import argparse
import os
import sys

# Qt fix — run before anything else that touches Qt
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yolo_sam_labeler import _fix_qt_plugins

# cv2 import triggers Qt plugin path override; fix must already be ready
import cv2                    # noqa: E402
_fix_qt_plugins()             # called AFTER cv2 import (cv2 overrides the env var)

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="YOLO SAM Labeler — Segmentation + Detection annotation")
    p.add_argument("--image-dir", default=os.environ.get("SAM_LABEL_IMAGE_DIR", ""))
    p.add_argument("--label-dir", default=os.environ.get("SAM_LABEL_DIR", ""))
    p.add_argument("--sam-checkpoint", default=os.environ.get("SAM_CHECKPOINT", ""))
    p.add_argument("--model-type", default=os.environ.get("SAM_MODEL_TYPE"),
                   choices=("vit_h", "vit_l", "vit_b"))
    p.add_argument("--yolo-weights", default=os.environ.get("YOLO_WEIGHTS", ""))
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    from PyQt5.QtWidgets import QApplication  # noqa: E402
    from yolo_sam_labeler.app import MainWindow  # noqa: E402

    app = QApplication.instance() or QApplication([sys.argv[0]])
    win = MainWindow(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        yolo_weights=args.yolo_weights,
    )
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
