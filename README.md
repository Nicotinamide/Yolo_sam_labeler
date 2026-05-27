# YOLO SAM Labeler

PyQt5 标注工具，SAM 点击分割 + 拖拽画框一体化界面，输出 YOLO 分割 + 检测双格式用于训练。

> **环境要求**：Python **3.10 / 3.11**（不要用 3.12+，PyTorch + PyQt5 + Jetson 兼容性问题）

## 安装

推荐直接使用项目脚本。脚本会自动检测 Jetson / x86 CUDA / CPU，并在 uv / conda
之间选择安装方式；日志使用统一的 `🔍` / `📦` / `🧪` / `✅` 风格。

### 一键安装

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/Nicotinamide/Yolo_sam_labeler.git
cd Yolo_sam_labeler
bash install.sh
```

常用安装参数：

```bash
bash install.sh --uv                       # 强制 uv（项目本地 .venv）
bash install.sh --conda                    # 强制 conda
bash install.sh --conda-env yolo-labeler   # 指定/创建 conda 环境
bash install.sh --clean-user-local         # 清理 ~/.local 里的残留 torch 包
```

脚本会：

- 自动检测平台（x86_64 / Jetson aarch64）
- 自动选择或使用指定的 uv / conda 安装方式
- 按设备安装 PyTorch
- Jetson 上自动安装系统 PyQt5 并开启 `system-site-packages`
- 切换到 `opencv-python-headless`，避免 Qt 冲突
- 验证 torch、CUDA、OpenCV、PyQt5、SAM 是否可用

`uv` 会缓存下载过的 wheel。第一次下载 Jetson torch 可能较慢，后续删除 `.venv`
重装会直接复用 `~/.cache/uv`，通常很快。

### 手动安装

<details>
<summary><b>x86_64 — uv（推荐）</b></summary>

前提：安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# 1. 安装基础依赖
uv sync

# 2. 安装 PyTorch（必须单独装，因为需要指定 CUDA 版本）
#    有 NVIDIA GPU:
uv pip install --force-reinstall torch torchvision --index https://download.pytorch.org/whl/cu124 --index-strategy first-index
#    无 GPU (纯 CPU):
uv pip install --force-reinstall torch torchvision --index https://download.pytorch.org/whl/cpu --index-strategy first-index

# 3. 安装 SAM/YOLO 依赖
uv pip install -e ".[sam,yolo]"

# 4. 避免 ultralytics 拉入带 Qt 的 opencv-python
uv pip uninstall opencv-python -y
uv pip install --force-reinstall "opencv-python-headless>=4.5,<4.12" "numpy>=1.21,<2"

# 5. 运行
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

# 2. 安装 Jetson 版 PyTorch
uv pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 --index https://pypi.jetson-ai-lab.io/jp6/cu126 --index-strategy first-index

# 3. 安装项目依赖 (含 SAM 1 + SAM 2)
uv pip install -e ".[sam,yolo]"

# 4. 避免 Qt/OpenCV 冲突
uv pip uninstall opencv-python -y
uv pip install --force-reinstall "opencv-python-headless>=4.5,<4.12" "numpy>=1.21,<2"

# 5. 运行
uv run yolo-sam-label
```

</details>

<details>
<summary><b>任意平台 — conda</b></summary>

前提：安装 [miniforge](https://github.com/conda-forge/miniforge) 或 Anaconda

```bash
# 自动创建/激活 yolo-sam-labeler，并安装完整依赖
bash install.sh --conda

# 或者指定环境名（不存在会自动按 environment.yml 创建）
bash install.sh --conda-env my-labeler
```

手动安装时按这个顺序执行：

```bash
# 1. 创建并进入 conda 基础环境
conda env create -f environment.yml
conda activate yolo-sam-labeler

# 2. 安装 PyTorch（根据平台选一个）
#    x86_64 GPU:
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    Jetson:
python -m pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
#    CPU:
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装本项目依赖
python -m pip install -e ".[sam,yolo]"

# 4. 避免 ultralytics 拉入带 Qt 的 opencv-python
python -m pip uninstall opencv-python -y
python -m pip install --force-reinstall "opencv-python-headless>=4.5,<4.12" "numpy>=1.21,<2"

# 5. 运行
yolo-sam-label
```

`environment.yml` 刻意不放 conda `opencv` 和 PyTorch：OpenCV 使用 pip 的 headless 版避免 Qt 冲突，PyTorch 由 `install.sh` 根据 GPU/CPU/Jetson 选择正确安装源。

</details>

---

### 为什么 PyTorch 要单独装？

PyTorch 的安装 URL 取决于你的 CUDA 版本和平台（x86/arm/Mac），无法用标准 `pyproject.toml` 统一描述。这是所有深度学习桌面项目的通行做法。`install.sh` 已经帮你处理了这一步。

### 启动报 "Cannot mix incompatible Qt library" 怎么办？

`ultralytics` 会拉一个带 Qt 的 `opencv-python`，跟 PyQt5 自带的 Qt 冲突。修复：

```bash
# 在 .venv 激活后
uv pip uninstall opencv-python -y
uv pip install --force-reinstall "opencv-python-headless>=4.5,<4.12" "numpy>=1.21,<2"
```

`install.sh` 默认已经帮你换成 headless 版了，手动安装时才需要这一步。

### conda 和 uv 冲突？

`install.sh` 会自动判断默认安装方式：
- 交互式终端 → 让你选择 conda / uv
- 非交互运行 → 已激活非 base 的 conda 环境则用 conda，否则用 uv
- 选择 conda 但未激活环境 → 自动创建/使用 `yolo-sam-labeler`

**强制指定安装方式：**
```bash
bash install.sh --uv                       # 强制 uv，忽略当前 conda
bash install.sh --conda                    # 强制 conda，未激活时自动创建默认环境
bash install.sh --conda-env my-labeler     # 指定 conda 环境名（不存在则创建）
```

`run.sh` 启动时按优先级：active conda → 项目 `.venv` → `uv run` → 全局命令。

**conda 安装失败 ("Defaulting to user installation because normal site-packages is not writeable")**：
你的 anaconda 目录归属 root。修复：

```bash
sudo chown -R $USER:$USER ~/anaconda3
```

或者直接用 uv 路线：
```bash
conda deactivate
bash install.sh --uv
```

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
| `--label-dir` | 标签目录种子（首次启动用） | `IMAGE_DIR/labels/` |
| `--sam-checkpoint` | SAM 权重路径 | 自动搜索当前目录 |
| `--model-type` | SAM 模型类型：`vit_h` / `vit_l` / `vit_b` | `vit_h` |
| `--yolo-weights` | YOLO 权重路径（可选） | 无 |

### 标签目录的工作方式

应用维护两个独立目录：分割（`seg_dir`）和检测（`detect_dir`）。两者可以指向同一物理目录（共享布局），也可以分开（推荐）。

- **菜单 → 文件 → 选择标签目录…**：自动嗅探目录格式（seg / detect / mixed / empty）。混合时弹拆分对话框；空目录两类共用，首次保存时自动定型并自动 seed sibling（`<dir>_seg` 或 `<dir>_detect`）。
- **菜单 → 高级 → 单独指定分割目录… / 单独指定检测目录…**：强制把当前目录设给某一类；另一类未设置时自动 seed sibling。
- **菜单 → 工具 → 整理标签目录…**：当前两类共用同一混合目录时，触发拆分对话框（多数派留原地，少数派移到 sibling）。
- 没有设置任何目录就保存：弹提示「请先选标签目录」，不会丢数据。
- 切换图片目录到新项目时，原先位于旧 image_dir 子树内的标签目录会自动清空，外部路径保留。
- 数据安全：写入空内容时会嗅探现存文件实际格式；格式不匹配时**拒写**并发警告，绝不覆盖异格式数据。

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
├── tests/                      # 176 个单元测试
└── docs/TEST_PLAN.md           # 功能逻辑文档 + 测试计划
```

## 测试

```bash
# 运行全部测试 (无需 display)
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -v
```

## License

MIT
