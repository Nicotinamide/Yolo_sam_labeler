#!/usr/bin/env bash
# =============================================================================
# YOLO SAM Labeler — 一键安装脚本
# 自动检测平台 (x86_64 / aarch64 Jetson) 和环境管理器 (conda / uv)
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

# 解析命令行参数: --uv / --conda / --conda-env <name> 强制指定路径
FORCE_MODE=""
TARGET_CONDA_ENV=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --uv)
            FORCE_MODE="uv"; shift ;;
        --conda)
            FORCE_MODE="conda"; shift ;;
        --conda-env)
            FORCE_MODE="conda"; TARGET_CONDA_ENV="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
用法: bash install.sh [选项]

选项:
  --uv                   强制走 uv 路线（创建 .venv，忽略 conda）
  --conda                强制走 conda 路线（必须先激活 conda 环境）
  --conda-env <name>     在指定的 conda 环境里安装（自动激活）
  -h, --help             显示此帮助

默认: 已激活 conda 环境 → conda; 否则 → uv
示例:
  bash install.sh                          # 自动选择
  bash install.sh --uv                     # 强制用 uv
  bash install.sh --conda-env yolo-labeler # 用指定 conda 环境
EOF
            exit 0 ;;
        *)
            error "未知参数: $1 (用 --help 查看)" ;;
    esac
done

# =============================================================================
# 工具检查
# =============================================================================

check_uv() { command -v uv &> /dev/null; }
check_conda() { command -v conda &> /dev/null; }

install_uv() {
    info "正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! check_uv; then
        error "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"
    fi
}

# 验证 site-packages 可写 — conda 在权限错乱时 pip 会装到 ~/.local 而不是环境里
verify_writable_site_packages() {
    local py="$1"
    local sp
    sp=$("$py" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    if [ ! -w "$sp" ]; then
        error "site-packages 不可写: $sp
    可能是 anaconda 目录归属 root 导致。修复方式:
        sudo chown -R \$USER:\$USER \$(dirname \$(dirname \$(dirname $sp)))
    或者使用 uv 路线（不需要 conda）:
        conda deactivate
        bash install.sh"
    fi
}

# 检测 PyTorch 安装方式
prompt_pytorch_choice() {
    case "$ARCH" in
        aarch64|arm64)
            echo "jetson"
            return
            ;;
    esac
    # x86_64 — 让用户选 GPU/CPU
    echo ""
    info "选择 PyTorch 版本:"
    echo "  1) NVIDIA GPU (CUDA 12.4)  [推荐]"
    echo "  2) CPU only"
    echo "  3) 跳过 (我自己装)"
    echo ""
    local choice
    read -r -p "  输入 1/2/3 [默认 1]: " choice
    case "${choice:-1}" in
        1) echo "gpu" ;;
        2) echo "cpu" ;;
        3) echo "skip" ;;
        *) echo "gpu" ;;
    esac
}

install_pytorch_for() {
    # $1 = pip command (e.g. "pip", "uv pip")
    # $2 = "gpu" | "cpu" | "jetson" | "skip"
    case "$2" in
        gpu)
            $1 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
            ;;
        cpu)
            $1 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
            ;;
        jetson)
            $1 install torch==2.8.0 torchvision==0.23.0 \
                --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
            ;;
        skip)
            warn "跳过 PyTorch 安装"
            ;;
    esac
}

verify_install() {
    local py="$1"
    info "验证安装..."
    "$py" -c "
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
    print(f'  SAM 2: ✗ (可选)')
print('  ✓ 所有依赖就绪')
"
}

# =============================================================================
# uv 路径 (x86_64)
# =============================================================================

install_uv_x86() {
    info "=== uv 安装路径 (x86_64) ==="

    if ! check_uv; then
        install_uv
    fi

    info "创建虚拟环境并安装依赖..."
    uv sync --extra all

    info "切换到 opencv-python-headless 避免 Qt 冲突..."
    uv pip uninstall opencv-python -y 2>/dev/null || true
    uv pip install --force-reinstall opencv-python-headless

    local pt_choice
    pt_choice=$(prompt_pytorch_choice)
    install_pytorch_for "uv pip" "$pt_choice"

    verify_install ".venv/bin/python"

    echo ""
    info "=== 安装完成 ==="
    info "运行方式: bash run.sh"
}

# =============================================================================
# uv 路径 (Jetson aarch64)
# =============================================================================

install_uv_jetson() {
    info "=== uv 安装路径 (Jetson aarch64) ==="

    if ! check_uv; then
        install_uv
    fi

    if ! python3 -c "import PyQt5" 2>/dev/null; then
        info "安装系统 PyQt5 (Jetson 上 pip 没有 aarch64 wheel)..."
        sudo apt-get update && sudo apt-get install -y python3-pyqt5
    fi

    info "创建虚拟环境 (继承系统 site-packages 以使用系统 PyQt5)..."
    uv venv --system-site-packages --python python3.10 .venv

    info "安装项目依赖..."
    uv pip install -e ".[sam,yolo]"

    info "切换到 opencv-python-headless..."
    uv pip uninstall opencv-python -y 2>/dev/null || true
    uv pip install --force-reinstall opencv-python-headless

    install_pytorch_for "uv pip" "jetson"

    verify_install ".venv/bin/python"

    echo ""
    info "=== 安装完成 ==="
    info "运行方式: bash run.sh"
}

# =============================================================================
# conda 路径 (用户已经在 conda 环境里)
# =============================================================================

install_conda() {
    info "=== conda 安装路径 (环境: $CONDA_DEFAULT_ENV) ==="

    local py="$CONDA_PREFIX/bin/python"
    if [ ! -x "$py" ]; then
        error "找不到 conda 环境的 python: $py"
    fi

    verify_writable_site_packages "$py"

    info "安装项目依赖..."
    "$py" -m pip install -e ".[sam,yolo]"

    info "切换到 opencv-python-headless..."
    "$py" -m pip uninstall opencv-python -y 2>/dev/null || true
    "$py" -m pip install --force-reinstall opencv-python-headless

    local pt_choice
    pt_choice=$(prompt_pytorch_choice)
    install_pytorch_for "$py -m pip" "$pt_choice"

    verify_install "$py"

    echo ""
    info "=== 安装完成 ==="
    info "运行方式（在当前 conda 环境里）: bash run.sh"
    info "下次开机记得先 conda activate $CONDA_DEFAULT_ENV"
}

# =============================================================================
# 主入口
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     YOLO SAM Labeler 安装向导            ║"
echo "╚══════════════════════════════════════════╝"
echo ""
info "架构: $ARCH"
info "工作目录: $SCRIPT_DIR"

# 路径选择规则:
# 1. 命令行 --uv / --conda / --conda-env 强制指定
# 2. 已激活非 base conda 环境 → conda
# 3. 默认 → uv

# 如果指定了 --conda-env，先激活那个环境
if [ -n "$TARGET_CONDA_ENV" ]; then
    if ! check_conda; then
        error "找不到 conda 命令。请先安装 anaconda/miniforge"
    fi
    info "激活 conda 环境: $TARGET_CONDA_ENV"
    eval "$(conda shell.bash hook)"
    if ! conda activate "$TARGET_CONDA_ENV" 2>/dev/null; then
        info "环境 $TARGET_CONDA_ENV 不存在，先创建..."
        conda create -n "$TARGET_CONDA_ENV" python=3.10 -y
        conda activate "$TARGET_CONDA_ENV"
    fi
fi

# 选择安装方式
if [ "$FORCE_MODE" = "conda" ]; then
    if [ -z "$CONDA_PREFIX" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
        error "--conda 模式要求先激活非 base 的 conda 环境，或使用 --conda-env <name>"
    fi
    install_conda
    exit 0
fi

if [ "$FORCE_MODE" = "uv" ]; then
    case "$ARCH" in
        x86_64|amd64) install_uv_x86 ;;
        aarch64|arm64) install_uv_jetson ;;
        *) error "不支持的架构: $ARCH" ;;
    esac
    exit 0
fi

# 自动模式
if [ -n "$CONDA_PREFIX" ] && [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    install_conda
    exit 0
fi

case "$ARCH" in
    x86_64|amd64) install_uv_x86 ;;
    aarch64|arm64) install_uv_jetson ;;
    *) error "不支持的架构: $ARCH" ;;
esac
