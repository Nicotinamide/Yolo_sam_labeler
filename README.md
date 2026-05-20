# YOLO SAM Labeler

PyQt5 annotation tool combining SAM point-to-mask and drag-to-box in a unified interface. Outputs YOLO segmentation + detection format for training.

## Features

- **Ctrl/Shift + click → segmentation mask** — point at an object, SAM generates a pixel-accurate mask
- **Drag → bounding box** — drag to draw detection boxes alongside masks
- **Hover + key → relabel** — point at any annotation, press the class key, and it's re-labeled instantly
- **Auto-save on edit** — annotation changes are written immediately; save buttons remain as explicit fallbacks
- **Class rename** — double-click a class or use the rename button
- **Dual output** — masks saved as YOLO seg format, boxes as YOLO detect format
- **Mask/box conversion helpers** — generate a detection box from a mask, or a SAM mask from a box prompt
- **YOLO pre-annotation** — let a YOLO model annotate first, then refine with SAM
- **ROI cropping** — draw a polygon to restrict SAM to a sub-region (faster for large images)
- **Auto-download SAM weights** — weights are fetched from Meta CDN on first use
- **Auto-load SAM weights** — an existing checkpoint is loaded on startup
- **Works on Jetson** — pre-configured Qt plugin fix for aarch64

## Install

PyTorch is NOT included as a dependency — it must be installed separately for your platform.

### conda (recommended for Jetson)

```bash
conda env create -f environment.yml
conda activate yolo-sam-labeler

# Install PyTorch for your platform:
#   x86_64:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   Jetson:  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
#   CPU:     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

yolo-sam-label
```

### uv (recommended for x86_64)

```bash
uv sync --extra all

# PyTorch must be installed separately (not included in uv deps):
#   GPU:  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   CPU:  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

uv run yolo-sam-label
```

## Usage

```bash
# Default paths
yolo-sam-label

# Custom paths
yolo-sam-label \
  --image-dir /path/to/images \
  --label-dir /path/to/labels \
  --sam-checkpoint ./sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --yolo-weights ./yolov8s-seg.pt
```

If only an image folder is provided, labels are saved to `IMAGE_DIR/labels`.
Segmentation masks are written to `LABEL_DIR/*.txt`; detection boxes are written
to `LABEL_DIR_detect/*.txt` so the two YOLO formats do not overwrite each other.

## Shortcuts

| Key | Action |
|-----|--------|
| `S` | Save current image |
| `N` / `Space` | Save & next image |
| `P` | Save & previous image |
| `D` | Skip |
| `C` | Clear all annotations |
| `Del` / `Backspace` | Delete hovered annotation |
| `R` | Reset zoom |
| `T` | Convert hovered/latest annotation: mask to box, box to SAM mask |
| `U` / `Ctrl+Z` | Undo last annotation |
| `Q` / `E` / `Esc` | Save & quit |
| `0`–`9` | Select class 0–9 (or relabel hovered annotation) |
| `Shift+A`–`Shift+Z` | Select class 10–35 (or relabel hovered annotation) |
| `[` / `]` | Cycle class |
| Scroll | Zoom (centered on cursor) |
| `+` / `-` | Zoom at viewport center |
| Middle-drag / Alt+Left-drag | Pan |
| Left-drag | Draw detection box |
| Ctrl/Shift + Left-click | Run SAM segmentation |
| Right-click | Delete annotation under cursor |

## Project structure

```
yolo_sam_labeler/
├── src/yolo_sam_labeler/
│   ├── __init__.py          # Qt plugin fix
│   ├── __main__.py          # Entry point
│   ├── app.py               # MainWindow controller
│   ├── models.py            # Data model (Mask, Box, stores)
│   ├── canvas.py            # Rendering + coordinate transforms
│   ├── sidebar.py           # Left panel (SAM/YOLO/ROI)
│   ├── right_panel.py       # Right panel (classes + actions)
│   ├── sam_service.py       # SAM loading + async predict + download
│   ├── yolo_service.py      # YOLO prediction wrapper
│   ├── workers.py           # SAM QThread worker
│   ├── io_utils.py          # YOLO format read/write
│   └── colors.py            # Color palette
├── tests/
│   └── test_models.py       # Data model unit tests
├── pyproject.toml
├── environment.yml
└── README.md
```
