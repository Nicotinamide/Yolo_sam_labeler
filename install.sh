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
DEFAULT_CONDA_ENV="yolo-sam-labeler"
CLEAN_USER_LOCAL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --uv)
            FORCE_MODE="uv"; shift ;;
        --conda)
            FORCE_MODE="conda"; shift ;;
        --conda-env)
            if [ $# -lt 2 ] || [[ "$2" == -* ]]; then
                error "--conda-env 需要指定环境名"
            fi
            FORCE_MODE="conda"; TARGET_CONDA_ENV="$2"; shift 2 ;;
        --clean-user-local)
            CLEAN_USER_LOCAL=1; shift ;;
        -h|--help)
            cat <<EOF
用法: bash install.sh [选项]

选项:
  --uv                   强制走 uv 路线（创建 .venv，忽略 conda）
  --conda                强制走 conda 路线（未激活环境时自动创建 yolo-sam-labeler）
  --conda-env <name>     在指定的 conda 环境里安装（不存在则自动创建）
  --clean-user-local     清理 ~/.local 里的 torch/nvidia/triton 残留包
  -h, --help             显示此帮助

默认: 交互式选择 conda / uv；非交互时：已激活 conda 环境 → conda，否则 → uv
示例:
  bash install.sh                          # 交互选择 conda / uv
  bash install.sh --uv                     # 强制用 uv
  bash install.sh --conda                  # 使用/创建默认 conda 环境
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

using_active_conda_env() {
    [ -n "$CONDA_PREFIX" ] && [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]
}

install_uv() {
    info "正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! check_uv; then
        error "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"
    fi
}

prompt_install_mode() {
    if [ -n "$FORCE_MODE" ]; then
        return
    fi

    if [ ! -t 0 ]; then
        if using_active_conda_env; then
            FORCE_MODE="conda"
        else
            FORCE_MODE="uv"
        fi
        return
    fi

    echo ""
    info "请选择环境管理器:"
    if check_conda; then
        if using_active_conda_env; then
            echo "  1) conda  (当前环境: $CONDA_DEFAULT_ENV)"
        else
            echo "  1) conda  (自动创建/使用环境: $DEFAULT_CONDA_ENV)"
        fi
        echo "  2) uv     (项目本地 .venv)"
        echo ""
        local choice
        read -r -p "  输入 1/2 [默认 1]: " choice || choice=""
        case "${choice:-1}" in
            1|c|conda)
                FORCE_MODE="conda"
                if ! using_active_conda_env && [ -z "$TARGET_CONDA_ENV" ]; then
                    TARGET_CONDA_ENV="$DEFAULT_CONDA_ENV"
                fi
                ;;
            2|u|uv)
                FORCE_MODE="uv"
                ;;
            *)
                error "未知选择: $choice"
                ;;
        esac
    else
        warn "未找到 conda，使用 uv 路线"
        FORCE_MODE="uv"
    fi
}

activate_or_create_conda_env() {
    local env_name="$1"
    if ! check_conda; then
        error "找不到 conda 命令。请先安装 anaconda/miniforge，或使用 --uv"
    fi
    info "激活 conda 环境: $env_name"
    eval "$(conda shell.bash hook)"
    if ! conda activate "$env_name" 2>/dev/null; then
        info "环境 $env_name 不存在，按 environment.yml 创建..."
        conda env create -f environment.yml -n "$env_name"
        conda activate "$env_name"
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

# 自动检测 PyTorch 安装方式 — 不再询问用户
detect_pytorch_choice() {
    case "$ARCH" in
        aarch64|arm64)
            echo "jetson"
            return
            ;;
    esac
    # x86_64 — 检测 NVIDIA GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        info "检测到 NVIDIA GPU → 安装 GPU 版 PyTorch (CUDA 12.4)" >&2
        echo "gpu"
    else
        info "未检测到 NVIDIA GPU → 安装 CPU 版 PyTorch" >&2
        echo "cpu"
    fi
}

install_pytorch_for() {
    # $1 = pip command (e.g. "pip", "uv pip")
    # $2 = "gpu" | "cpu" | "jetson" | "skip"
    case "$2" in
        gpu)
            $1 install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
            ;;
        cpu)
            $1 install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
            ;;
        jetson)
            $1 install --force-reinstall torch==2.8.0 torchvision==0.23.0 \
                --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
            ;;
        skip)
            warn "跳过 PyTorch 安装"
            ;;
        *)
            error "未知 PyTorch 安装选项: $2"
            ;;
    esac
}

ensure_numpy_compatible() {
    local py="$1"
    local pip_cmd="$2"
    local numpy_ver
    numpy_ver=$("$py" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "")
    if [[ "$numpy_ver" == 2.* ]]; then
        warn "numpy 被某个包升到 $numpy_ver，回退到 <2..."
        $pip_cmd install --force-reinstall "numpy>=1.21,<2"
    fi
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

    info "创建虚拟环境并安装基础依赖..."
    uv sync

    local pt_choice
    pt_choice=$(detect_pytorch_choice)
    install_pytorch_for "uv pip" "$pt_choice"

    info "安装 SAM/YOLO 可选依赖..."
    uv pip install -e ".[sam,yolo]"

    info "切换到 opencv-python-headless 避免 Qt 冲突..."
    uv pip uninstall opencv-python -y 2>/dev/null || true
    uv pip install --force-reinstall \
        "opencv-python-headless>=4.5,<4.12" \
        "numpy>=1.21,<2"

    ensure_numpy_compatible ".venv/bin/python" "uv pip"

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
    if [ -f ".venv/pyvenv.cfg" ] \
        && ! grep -qi "include-system-site-packages = true" ".venv/pyvenv.cfg"; then
        warn ".venv 已存在但未开启 system-site-packages，重建以使用系统 PyQt5..."
        uv venv --clear --system-site-packages --python python3.10 .venv
    else
        uv venv --allow-existing --system-site-packages --python python3.10 .venv
    fi

    info "安装 Jetson 版 PyTorch..."
    install_pytorch_for "uv pip" "jetson"

    info "安装项目依赖..."
    uv pip install -e ".[sam,yolo]"

    info "切换到 opencv-python-headless..."
    uv pip uninstall opencv-python -y 2>/dev/null || true
    uv pip install --force-reinstall \
        "opencv-python-headless>=4.5,<4.12" \
        "numpy>=1.21,<2"

    ensure_numpy_compatible ".venv/bin/python" "uv pip"

    verify_install ".venv/bin/python"

    echo ""
    info "=== 安装完成 ==="
    info "运行方式: bash run.sh"
}

# =============================================================================
# conda 路径 (用户已经在 conda 环境里)
# =============================================================================

# 检测并清理 ~/.local 里的污染包 (之前因为 pip "Defaulting to user installation" 装错地方的残留)
check_user_local_pollution() {
    local found=""
    for pyver in 3.10 3.11 3.12; do
        local d="$HOME/.local/lib/python${pyver}/site-packages"
        if [ -d "$d/torch" ] || [ -d "$d/nvidia" ] || [ -d "$d/triton" ]; then
            found="$found $d"
        fi
    done
    if [ -n "$found" ]; then
        warn "发现 ~/.local 里的 torch/nvidia/triton 残留，可能导致 libcudnn 版本冲突:"
        for d in $found; do
            echo "    $d"
        done
        if [ "$CLEAN_USER_LOCAL" = "1" ]; then
            warn "按 --clean-user-local 清理上述残留包..."
            for d in $found; do
                rm -rf "$d"/torch* "$d"/nvidia* "$d"/triton* 2>/dev/null
            done
            info "已清理 ~/.local 残留包"
        else
            warn "本次不自动删除用户目录包；如确认要清理，请重新运行并加 --clean-user-local"
        fi
    fi
}

ensure_conda_pyqt() {
    local py="$1"
    if "$py" -c "from PyQt5.QtCore import QT_VERSION_STR; print(QT_VERSION_STR)" >/dev/null 2>&1; then
        return
    fi
    if ! check_conda; then
        error "当前 conda 环境缺少 PyQt5，且找不到 conda 命令。请先安装: conda install -c conda-forge 'pyqt>=5.15,<6'"
    fi
    info "当前 conda 环境缺少 PyQt5，使用 conda-forge 安装 pyqt..."
    conda install -y -c conda-forge "pyqt>=5.15,<6"
}

install_conda() {
    info "=== conda 安装路径 (环境: $CONDA_DEFAULT_ENV) ==="

    local py="$CONDA_PREFIX/bin/python"
    if [ ! -x "$py" ]; then
        error "找不到 conda 环境的 python: $py"
    fi

    verify_writable_site_packages "$py"
    check_user_local_pollution
    ensure_conda_pyqt "$py"

    # 强制 pip 不要 fall back 到 ~/.local。即使 site-packages 真的不可写，
    # 我们也要让它失败而不是默默装到错的地方。
    export PYTHONNOUSERSITE=1
    export PIP_USER=0

    # pip 命令缩写
    local PIP="$py -m pip --no-cache-dir"

    info "更新 pip 构建工具..."
    $PIP install --upgrade "pip>=23" "setuptools>=68" wheel

    local pt_choice
    pt_choice=$(detect_pytorch_choice)
    install_pytorch_for "$PIP" "$pt_choice"

    info "安装项目依赖..."
    $PIP install -e ".[sam,yolo]"

    info "切换到 opencv-python-headless (钉版本以避免 numpy 被升到 2.x)..."
    $PIP uninstall opencv-python -y 2>/dev/null || true
    # 同时约束 numpy<2，避免 pip 把 numpy 升到 2.x
    $PIP install --force-reinstall \
        "opencv-python-headless>=4.5,<4.12" \
        "numpy>=1.21,<2"

    info "校验依赖一致性..."
    ensure_numpy_compatible "$py" "$PIP"

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
# 2. 交互式终端询问 conda / uv
# 3. 非交互: 已激活非 base conda 环境 → conda; 否则 → uv
prompt_install_mode

# 选择安装方式
if [ "$FORCE_MODE" = "conda" ]; then
    if [ -n "$TARGET_CONDA_ENV" ]; then
        activate_or_create_conda_env "$TARGET_CONDA_ENV"
    elif ! using_active_conda_env; then
        activate_or_create_conda_env "$DEFAULT_CONDA_ENV"
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

case "$ARCH" in
    x86_64|amd64) install_uv_x86 ;;
    aarch64|arm64) install_uv_jetson ;;
    *) error "不支持的架构: $ARCH" ;;
esac
