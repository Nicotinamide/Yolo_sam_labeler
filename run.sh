#!/usr/bin/env bash
# =============================================================================
# YOLO SAM Labeler — 快速启动脚本
# 自动激活 venv 并启动应用，无需记住命令名
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 1. 优先用项目本地 .venv (uv 创建的)
if [ -x ".venv/bin/yolo-sam-label" ]; then
    exec .venv/bin/yolo-sam-label "$@"
fi

# 2. 项目本地 .venv 存在但没装入口脚本 (开发模式没装) — 用 python -m
if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python -m yolo_sam_labeler "$@"
fi

# 3. 用 uv run 兜底
if command -v uv &> /dev/null; then
    exec uv run yolo-sam-label "$@"
fi

# 4. 全局安装
if command -v yolo-sam-label &> /dev/null; then
    exec yolo-sam-label "$@"
fi

# 都不行
echo "✗ 未找到可用的 Python 环境"
echo ""
echo "请先运行安装脚本:"
echo "    bash install.sh"
exit 1
