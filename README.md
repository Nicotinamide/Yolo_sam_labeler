# YOLO SAM Labeler

PyQt5 标注工具，SAM 点击分割 + 拖拽画框一体化界面，输出 YOLO 分割 + 检测双格式用于训练。

> **环境要求**：Python **3.10 / 3.11**（不要用 3.12+，PyTorch + PyQt5 + Jetson 兼容性问题）

## 快速安装

### 方式一：一键脚本（推荐）

自动检测平台（x86_64 / Jetson aarch64），自动安装所有依赖：

```bash
git clone https://github.com/Nicotinamide/Yolo_sam_labeler.git
cd Yolo_sam_labeler
bash install.sh
```

脚本会：
1. 检测你的架构（x86_64 / aarch64）
2. 自动安装 uv（如果没有）
3. 创建虚拟环境并安装所有依赖（包括 PyTorch）
4. Jetson 上自动安装系统 PyQt5 并开启 system-site-packages
5. 验证安装是否成功

---

### 方式二：手动安装

<details>
<summary><b>x86_64 — uv（推荐）</b></summary>

前提：安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 1. 安装基础依赖
uv sync --extra all

# 2. 安装 PyTorch（必须单独装，因为需要指定 CUDA 版本）
#    有 NVIDIA GPU:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    无 GPU (纯 CPU):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 运行
uv run yolo-sam-label
```

</details>

<details>
<summary><b>Jetson (aarch64) — uv</b></summary>

前提：安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) + 系统 PyQt5

```bash
# 0. 系统 PyQt5 (apt 安装, pip 上没有 aarch64 wheel)
sudo apt-get install -y python3-pyqt5

# 1. 创建 venv (继承系统包以使用 apt 的 PyQt5)
uv venv --system-site-packages --python python3.10 .venv

# 2. 安装项目依赖 (含 SAM 1 + SAM 2)
uv pip install -e ".[sam,yolo]"

# 3. 安装 Jetson 版 PyTorch
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 4. 运行
uv run yolo-sam-label
```

</details>

<details>
<summary><b>任意平台 — conda</b></summary>

前提：安装 [miniforge](https://github.com/conda-forge/miniforge) 或 Anaconda

```bash
# 1. 创建环境 (包含 Python, NumPy, OpenCV, PyQt5, ultralytics, SAM)
conda env create -f environment.yml
conda activate yolo-sam-labeler

# 2. 安装 PyTorch（根据平台选一个）
#    x86_64 GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    Jetson:
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
#    CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装本项目
pip install -e .

# 4. 运行
yolo-sam-label
```

</details>

---

### 为什么 PyTorch 要单独装？

PyTorch 的安装 URL 取决于你的 CUDA 版本和平台（x86/arm/Mac），无法用标准 `pyproject.toml` 统一描述。这是所有深度学习桌面项目的通行做法。`install.sh` 已经帮你处理了这一步。

### 启动报 "Cannot mix incompatible Qt library" 怎么办？

`ultralytics` 会拉一个带 Qt 的 `opencv-python`，跟 PyQt5 自带的 Qt 冲突。修复：

```bash
# 在 .venv 激活后
uv pip uninstall opencv-python -y
uv pip install --force-reinstall opencv-python-headless
```

`install.sh` 默认已经帮你换成 headless 版了，手动安装时才需要这一步。

### conda 和 uv 冲突？

**只用一个**。`install.sh` 检测到 conda 环境会拒绝运行，避免你创建出"conda 装一半 + uv 装另一半"的破环境。

- 想用 uv：`conda deactivate` 后再 `bash install.sh`
- 想用 conda：跟着上面"任意平台 — conda"小节手动来，**不要用 install.sh**

---

## 使用

最简方式：直接运行启动脚本（自动找 venv）：

```bash
bash run.sh

# 带参数也可以
bash run.sh --image-dir /path/to/images --yolo-weights ./weights/yolo/best.pt
```

或者用命令行入口：

```bash
# 默认路径
yolo-sam-label

# 指定路径
yolo-sam-label \
  --image-dir /path/to/images \
  --label-dir /path/to/labels \
  --sam-checkpoint ./weights/sam/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --yolo-weights ./weights/yolo/best.pt
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image-dir` | 图片目录 | 当前目录 |
| `--label-dir` | 标签保存目录 | `IMAGE_DIR/labels/` |
| `--sam-checkpoint` | SAM 权重路径 | 自动搜索当前目录 |
| `--model-type` | SAM 模型类型：`vit_h` / `vit_l` / `vit_b` | `vit_h` |
| `--yolo-weights` | YOLO 权重路径（可选） | 无 |

- 只指定图片目录时，标签自动保存到 `IMAGE_DIR/labels/`
- 分割 mask → `labels/*.txt`（YOLO seg 格式）
- 检测框 → `labels_detect/*.txt`（YOLO detect 格式）
- 两种格式互不覆盖

## SAM 权重

首次使用时程序会提示自动从 Meta 官方下载 SAM 权重。也可以通过菜单「模型 → SAM 权重管理」浏览所有版本并下载选择。

### SAM 1 (默认，包名 `segment-anything`)

| 模型 | 文件名 | 大小 | 说明 |
|------|--------|------|------|
| ViT-H | sam_vit_h_4b8939.pth | 2.4 GB | 最高精度 |
| ViT-L | sam_vit_l_0b3195.pth | 1.2 GB | 平衡 |
| ViT-B | sam_vit_b_01ec64.pth | 375 MB | 最快 |

### SAM 2.1 (推荐新项目，包名 `sam-2`)

| 模型 | 文件名 | 大小 | 说明 |
|------|--------|------|------|
| Hiera Large | sam2.1_hiera_large.pt | 898 MB | 比 SAM 1 ViT-H 精度更好且更小 |
| Hiera Base+ | sam2.1_hiera_base_plus.pt | 323 MB | 中型 |
| Hiera Small | sam2.1_hiera_small.pt | 184 MB | 小型 |
| Hiera Tiny | sam2.1_hiera_tiny.pt | 156 MB | 最快 |

**SAM 2 vs SAM 1**: 相同精度下 SAM 2 推理快约 6×、文件小得多。两者都已默认安装，可以同时使用。

> Jetson 用户：SAM 2 的 CUDA 扩展可能编译失败（不影响功能，会回退到 PyTorch 实现，速度略慢）。

权重默认下载到项目目录的 `weights/sam/` 文件夹，YOLO 权重默认在 `weights/yolo/`。

## 功能特性

- **Ctrl/Shift + 左键** → SAM 分割（点击物体，生成像素级 mask）
- **左键拖拽** → 画检测框
- **悬停 + 数字键** → 重新标类别（无需删除重画）
- **编辑即保存** → 每次标注修改自动写盘
- **双击类别** → 重命名
- **T 键** → mask 和框互转（mask→外接框，框→SAM mask）
- **YOLO 预标注** → 先用 YOLO 模型批量标，再用 SAM 精修
- **ROI 裁剪** → 画多边形限定 SAM 范围，大图加速
- **邻图预编码** → 切图时后台预加载相邻图的 SAM embedding
- **自动下载 SAM** → 首次使用自动从 Meta CDN 拉取权重

## 快捷键

| 按键 | 功能 |
|------|------|
| `S` | 保存当前图 |
| `N` / `空格` | 保存并下一张 |
| `P` | 保存并上一张 |
| `D` | 跳过（不保存直接下一张） |
| `C` | 清空当前图所有标注 |
| `Del` / `Backspace` | 删除鼠标下方的标注 |
| `R` | 重置缩放 |
| `T` | Mask/框互转 |
| `U` / `Ctrl+Z` | 撤销最近一次标注 |
| `Q` / `E` / `Esc` | 保存并退出 |
| `0`–`9` | 选择类别 0–9 (或重标悬停标注) |
| `Shift+A`–`Shift+Z` | 选择类别 10–35 |
| `[` / `]` | 上/下切换类别 |
| 滚轮 | 以光标为中心缩放 |
| `+` / `-` | 视图中心缩放 |
| 中键拖拽 / Alt+左键拖拽 | 平移画布 |
| 左键拖拽 | 画检测框 |
| Ctrl/Shift + 左键 | SAM 分割 |
| 右键 | 删除光标下的标注 |

## 项目结构

```
yolo_sam_labeler/
├── install.sh                  # 一键安装脚本
├── run.sh                      # 快速启动脚本
├── pyproject.toml              # 项目配置 & 依赖
├── environment.yml             # conda 环境 (任意平台备选)
├── weights/                    # 模型权重 (gitignored)
│   ├── sam/                    # SAM 1/2 权重
│   └── yolo/                   # YOLO 权重
├── src/yolo_sam_labeler/
│   ├── __init__.py             # Qt 插件修复 (Jetson)
│   ├── __main__.py             # 入口
│   ├── app.py                  # 主窗口控制器
│   ├── app_sam.py              # SAM + ROI 逻辑 (mixin)
│   ├── app_input.py            # 鼠标键盘事件 (mixin)
│   ├── models.py               # 数据模型 (Mask, Box, Store)
│   ├── canvas.py               # 渲染 + 坐标变换
│   ├── sidebar.py              # 左侧面板 (SAM/YOLO/ROI)
│   ├── right_panel.py          # 右侧面板 (类别 + 操作)
│   ├── sam_service.py          # SAM 异步加载/编码/预测
│   ├── yolo_service.py         # YOLO 异步推理
│   ├── workers.py              # SAM QThread worker + embedding 缓存
│   ├── weight_manager.py       # SAM 权重管理对话框
│   ├── io_utils.py             # YOLO 格式读写
│   ├── colors.py               # 类别颜色
│   └── backends/               # SAM 1/2 后端抽象
│       ├── base.py             # SamBackend 抽象基类
│       ├── factory.py          # 文件名 → 后端选择
│       ├── sam1.py             # SAM 1 (segment-anything) 后端
│       └── sam2.py             # SAM 2 (sam-2) 后端
├── tests/                      # 134 个单元测试
└── docs/TEST_PLAN.md           # 功能逻辑文档 + 测试计划
```

## 测试

```bash
# 运行全部测试 (无需 display)
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -v
```

## License

MIT
