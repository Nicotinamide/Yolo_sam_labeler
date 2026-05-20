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

check_conda() {
    if command -v conda &> /dev/null; then
        info "conda 已安装: $(conda --version)"
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
print('  ✓ 所有依赖就绪')
"

    echo ""
    info "=== 安装完成 ==="
    info "运行方式: uv run yolo-sam-label"
    info "或者激活环境后运行: source .venv/bin/activate && yolo-sam-label"
}

# =============================================================================
# aarch64 / Jetson 安装路径 (使用 conda)
# =============================================================================

install_jetson() {
    info "=== Jetson / aarch64 安装模式 (conda) ==="

    if ! check_conda; then
        error "Jetson 平台需要 conda。请先安装 miniforge:
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
    bash Miniforge3-Linux-aarch64.sh
    然后重新运行本脚本。"
    fi

    ENV_NAME="yolo-sam-labeler"

    if conda env list | grep -q "^${ENV_NAME} "; then
        info "conda 环境 $ENV_NAME 已存在，跳过创建"
    else
        info "创建 conda 环境..."
        conda env create -f environment.yml
    fi

    info "激活环境并安装 PyTorch..."
    echo ""
    echo "  请手动执行以下命令完成安装:"
    echo ""
    echo "    conda activate $ENV_NAME"
    echo "    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126"
    echo "    pip install -e ."
    echo ""
    echo "  安装完成后运行:"
    echo "    yolo-sam-label"
    echo ""

    # 尝试自动执行
    eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
    if conda activate "$ENV_NAME" 2>/dev/null; then
        info "自动激活环境成功，正在安装 PyTorch for Jetson..."
        pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
        pip install -e .

        info "验证安装..."
        python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import cv2
print(f'  OpenCV {cv2.__version__}')
from PyQt5.QtCore import QT_VERSION_STR
print(f'  PyQt5 {QT_VERSION_STR}')
print('  ✓ 所有依赖就绪')
"
        echo ""
        info "=== 安装完成 ==="
        info "运行方式: conda activate $ENV_NAME && yolo-sam-label"
    else
        warn "无法自动激活 conda 环境，请按上方提示手动执行。"
    fi
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
