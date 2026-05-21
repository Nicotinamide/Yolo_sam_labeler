#!/usr/bin/env bash
# 诊断 Qt xcb 插件问题
# 用法: bash diagnose_qt.sh

set +e

echo "=== Qt xcb 插件诊断 ==="
echo ""

# 找到 .venv 里的 PyQt5 路径
PYQT_DIR=$(.venv/bin/python -c "import PyQt5, os; print(os.path.dirname(PyQt5.__file__))" 2>/dev/null)
if [ -z "$PYQT_DIR" ]; then
    echo "✗ 未找到 .venv 中的 PyQt5"
    exit 1
fi
echo "PyQt5 目录: $PYQT_DIR"

XCB_PLUGIN="$PYQT_DIR/Qt5/plugins/platforms/libqxcb.so"
if [ ! -f "$XCB_PLUGIN" ]; then
    XCB_PLUGIN="$PYQT_DIR/plugins/platforms/libqxcb.so"
fi

if [ ! -f "$XCB_PLUGIN" ]; then
    echo "✗ 未找到 libqxcb.so"
    find "$PYQT_DIR" -name "libqxcb.so" 2>/dev/null
    exit 1
fi
echo "xcb 插件: $XCB_PLUGIN"
echo ""

echo "=== 缺失的依赖库 ==="
ldd "$XCB_PLUGIN" | grep "not found" || echo "  无 (所有库齐全)"
echo ""

echo "=== DISPLAY 环境变量 ==="
echo "  DISPLAY=$DISPLAY"
echo "  WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
echo "  XDG_SESSION_TYPE=$XDG_SESSION_TYPE"
echo ""

echo "=== 详细 Qt 调试 (运行应用) ==="
QT_DEBUG_PLUGINS=1 .venv/bin/python -c "
from PyQt5.QtWidgets import QApplication
import sys
app = QApplication(sys.argv)
print('OK: Qt initialized')
" 2>&1 | head -50
