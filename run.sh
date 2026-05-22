#!/usr/bin/env bash
# =============================================================================
# YOLO SAM Labeler — 快速启动脚本
# 自动检测 venv (uv / conda / 系统全局) 并启动应用
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Conda 激活会设 QT_QPA_PLATFORM_PLUGIN_PATH 指向 anaconda 的 platforms 目录，
# 覆盖 venv 自己的 PyQt5。我们的代码会修复，但 unset 一下更稳妥。
unset QT_QPA_PLATFORM_PLUGIN_PATH

check_runtime() {
    local py="$1"
    "$py" -c "import yolo_sam_labeler, torch, cv2; from PyQt5 import QtCore, QtWidgets" 2>/dev/null
}

# ----------------------------------------------------------------------------
# 1. 已激活的 conda 环境（用户跑了 conda activate xxx）
#    优先使用它，因为用户显式选择了这个环境。
# ----------------------------------------------------------------------------
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    if check_runtime "$CONDA_PREFIX/bin/python"; then
        exec "$CONDA_PREFIX/bin/python" -m yolo_sam_labeler "$@"
    else
        echo "⚠ 已激活 conda 环境 ($CONDA_DEFAULT_ENV) 但缺少运行依赖。"
        echo ""
        echo "  请运行:"
        echo "    bash install.sh --conda"
        echo "  或者退出 conda 后运行:"
        echo "    bash install.sh --uv"
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# 2. 项目本地 .venv (uv 创建的)
# ----------------------------------------------------------------------------
if [ -x ".venv/bin/python" ]; then
    if check_runtime ".venv/bin/python"; then
        if [ -x ".venv/bin/yolo-sam-label" ]; then
            exec .venv/bin/yolo-sam-label "$@"
        fi
        exec .venv/bin/python -m yolo_sam_labeler "$@"
    else
        echo "⚠ .venv 存在但依赖不全。请重新运行 install.sh。"
        exit 1
    fi
fi

# ----------------------------------------------------------------------------
# 3. uv 兜底（带所有可选依赖）
# ----------------------------------------------------------------------------
if command -v uv &> /dev/null; then
    echo "未找到 venv，使用 uv 创建并启动…"
    exec uv run --extra all yolo-sam-label "$@"
fi

# ----------------------------------------------------------------------------
# 4. 全局安装
# ----------------------------------------------------------------------------
if command -v yolo-sam-label &> /dev/null; then
    exec yolo-sam-label "$@"
fi

echo "✗ 未找到可用的 Python 环境"
echo ""
echo "请先运行安装脚本:"
echo "    bash install.sh"
exit 1
