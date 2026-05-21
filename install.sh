#!/usr/bin/env bash
# =============================================================================
# YOLO SAM Labeler — 一键安装脚本
# 自动检测平台 (x86_64 / aarch64 Jetson), 创建环境并安装所有依赖
# =============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

ARCH=$(uname -m)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

info "检测到架构: $ARCH"
info "工作目录: $SCRIPT_DIR"

# =============================================================================
# 检查基础工具
# =============================================================================

check_uv() {
    if command -v uv &> /dev/null; then
        info "uv 已安装: $(uv --version)"
        return 0
    fi
    return 1
}

install_uv() {
    info "正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        error "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    info "uv 安装成功"
}

# =============================================================================
# x86_64 安装路径 (使用 uv)
# =============================================================================

install_x86() {
    info "=== x86_64 安装模式 (uv) ==="

    if ! check_uv; then
        install_uv
    fi

    info "创建虚拟环境并安装基础依赖..."
    uv sync --extra all

    info "安装 PyTorch (GPU CUDA 12.4)..."
    echo ""
    echo "  如果没有 NVIDIA GPU, 按 Ctrl+C 中断, 然后运行:"
    echo "    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    echo ""
    echo "  5 秒后自动安装 GPU 版本..."
    sleep 5

    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    info "验证安装..."
    uv run python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import cv2
print(f'  OpenCV {cv2.__version__}')
from PyQt5.QtCore import QT_VERSION_STR
print(f'  PyQt5 {QT_VERSION_STR}')
import segment_anything
print(f'  SAM 1: ✓')
try:
    import sam2
    print(f'  SAM 2: ✓')
except ImportError:
    print(f'  SAM 2: ✗ (可选，不影响 SAM 1 使用)')
print('  ✓ 所有依赖就绪')
"

    echo ""
    info "=== 安装完成 ==="
    info "运行方式（推荐）: bash run.sh"
    info "或者: uv run yolo-sam-label"
    info "或者: source .venv/bin/activate && yolo-sam-label"
}

# =============================================================================
# aarch64 / Jetson 安装路径 (使用 uv + 系统 PyQt5)
# =============================================================================

install_jetson() {
    info "=== Jetson / aarch64 安装模式 (uv + system-site-packages) ==="

    if ! check_uv; then
        install_uv
    fi

    # Jetson 上 PyQt5 没有 pip wheel，必须用系统 apt 包
    if ! python3 -c "import PyQt5" 2>/dev/null; then
        info "安装系统 PyQt5..."
        sudo apt-get update && sudo apt-get install -y python3-pyqt5
    fi

    info "创建虚拟环境 (继承系统 site-packages 以使用系统 PyQt5)..."
    uv venv --system-site-packages --python python3.10 .venv

    info "安装基础依赖..."
    uv pip install -e ".[sam,yolo]"

    info "安装 PyTorch for Jetson (JetPack 6 / CUDA 12.6)..."
    uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

    info "验证安装..."
    .venv/bin/python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import cv2
print(f'  OpenCV {cv2.__version__}')
from PyQt5.QtCore import QT_VERSION_STR
print(f'  PyQt5 {QT_VERSION_STR}')
import segment_anything
print(f'  SAM 1: ✓')
try:
    import sam2
    print(f'  SAM 2: ✓')
except ImportError:
    print(f'  SAM 2: ✗ (可选，不影响 SAM 1 使用)')
print('  ✓ 所有依赖就绪')
"

    echo ""
    info "=== 安装完成 ==="
    info "运行方式（推荐）: bash run.sh"
    info "或者: uv run yolo-sam-label"
    info "或者: source .venv/bin/activate && yolo-sam-label"
}

# =============================================================================
# 主入口
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     YOLO SAM Labeler 安装向导            ║"
echo "╚══════════════════════════════════════════╝"
echo ""

case "$ARCH" in
    x86_64|amd64)
        install_x86
        ;;
    aarch64|arm64)
        install_jetson
        ;;
    *)
        error "不支持的架构: $ARCH (仅支持 x86_64 和 aarch64/Jetson)"
        ;;
esac
