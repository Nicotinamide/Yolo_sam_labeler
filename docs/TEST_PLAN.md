# YOLO SAM Labeler — 测试计划与功能逻辑文档

## 一、模块功能逻辑梳理

---

### 1. models.py — 数据模型

#### Box (检测框)
- 存储: class_id, x1, y1, x2, y2 (原图像素坐标)
- 计算属性: width, height, center
- 命中检测: contains(x, y) → x1 <= x <= x2 and y1 <= y <= y2

#### Mask (分割掩膜)
- 存储: class_id, data (H×W uint8 二值数组, 0/1)
- 命中检测: contains(x, y) → data[y, x] == 1 (边界检查)

#### ClassRegistry (类别注册表)
- 数据: dict[int, str] (class_id → name)
- 操作:
  - add(name) → 分配 max_id+1, 发射 classes_changed
  - set_names(dict) → 替换全部, 清理空名
  - ensure(id, name?) → 若不存在则添加
  - ensure_ids(iterable) → 批量 ensure, 最多发射一次信号
  - remove(id) → 删除, 不存在返回 False
  - rename(id, name) → 重命名
- 信号: classes_changed (QObject)
- 边界条件:
  - 空注册表时 max_id 返回 -1
  - add 总是用 max+1 (稀疏 ID 不回填)

#### AnnotationStore (标注存储)
- 状态: masks[], boxes[], last_kind, image_width, image_height, label_dir
- 查询:
  - total_count → len(masks) + len(boxes)
  - find_at(x, y) → 从后往前搜索: 先 boxes 再 masks (后添加的在顶层)
- 添加:
  - add_mask(data, class_id) → append, last_kind="mask", emit changed
  - add_box(x1,y1,x2,y2,cid) → append, last_kind="box", emit changed
- 转换:
  - replace_mask_with_box(idx, x1,y1,x2,y2) → pop mask, append box
  - replace_box_with_mask(idx, snapshot, mask, cid) → 验证 snapshot 一致后替换
    - _find_box_snapshot: 先尝试 idx 位置精确匹配, 再全表扫描
- 删除:
  - delete_at(x, y) → find_at + del, 刷新 last_kind
  - undo_last() → 根据 last_kind pop 对应列表最后一项
    - 若 last_kind="box" 且 boxes 非空 → pop boxes
    - 否则 pop masks
    - BUG 风险: 如果 last_kind="mask" 但 masks 空, 返回 False (不会 pop boxes)
- 批量导入:
  - apply_yolo_predictions(masks, mask_cids, boxes, box_cids, replace)
    - replace=True → clear 后追加
- 信号: changed (每次突变后)

---

### 2. io_utils.py — 文件格式读写

#### 图像扫描
- scan_images(dir) → sorted list of abs paths, 过滤 IMAGE_EXTS

#### 图像加载
- load_image_bgr(path):
  - 用 open() + np.frombuffer + cv2.imdecode (兼容非 ASCII 路径)
  - 调用 _apply_exif_orientation 修正拍照方向 (1-8 EXIF 值)
  - 失败返回 None
- load_image_rgb(path): BGR→RGB

#### 类别文件
- load_class_names(path):
  - 支持格式: "name" (行号为 ID), "id name", "id: name"
  - 空行/# 注释跳过
  - next_id 跟踪下一个隐式 ID
- save_class_names(path, dict):
  - 连续 ID (0,1,2,...) → 只写名字
  - 稀疏 ID → "id name" 格式
  - 自动创建父目录

#### YOLO 分割格式
- masks_to_yolo_lines(masks, w, h):
  - 对每个 mask: findContours → 取最大轮廓 → approxPolyDP 简化
  - 面积<50 的跳过, 顶点<3 的跳过
  - 输出: "class_id x1/w y1/h x2/w y2/h ..." (归一化坐标)
- load_masks_from_txt(path, w, h):
  - 每行 ≥7 个 token (class_id + ≥3 个坐标对)
  - fillPoly 重建 mask, sum<30 的跳过

#### YOLO 检测格式
- boxes_to_yolo_lines(boxes, w, h):
  - 输出: "class_id cx/w cy/h bw/w bh/h"
- load_boxes_from_txt(path, w, h):
  - 解析 cx,cy,bw,bh → 转回 x1,y1,x2,y2
  - 宽高<3 的跳过

#### 统一保存/加载
- save_labels(store, stem, w, h):
  - → save_labels_seg: {label_dir}/{stem}.txt
  - → save_labels_detect: {label_dir}_detect/{stem}.txt
  - detect 目录: 若无 boxes 且文件不存在则不创建
- load_labels_for_image(store, image_path, w, h):
  - 清空 store → 加载 seg → 加载 detect → ensure_ids → emit changed

---

### 3. canvas.py — 坐标变换与渲染

#### CoordTransformer
- 模型: fit-to-canvas viewport
  - fit_scale = min(cw/iw, ch/ih) — 全图适配缩放
  - zoom: 1.0~12.0, zoom=1 显示全图
  - view_x1, view_y1: 视图左上角 (图像坐标)
  - view_width = iw/zoom, view_height = ih/zoom
- 坐标转换:
  - canvas_to_image(cx, cy) → 画布像素 → 图像像素 (可能 None)
  - image_to_canvas(ix, iy) → 图像像素 → 画布像素
- 操作:
  - zoom_at(factor, cursor) → 以光标为锚点缩放, clamp
  - pan_by(dx, dy) → 平移 (画布像素除以 scale), clamp
  - update_canvas_size(w, h) → 保持视图中心
  - reset() → zoom=1, origin=(0,0)
- 边界钳位: view 不能超出 [0, iw-view_width] × [0, ih-view_height]

#### render_composite (纯函数)
- 输入: image_bgr, store, hover 状态, draw_state, ROI
- 图层顺序: 原图 → mask 叠加(0.42/0.58) → mask 文字 → hover 轮廓 → box 框 → 拖拽预览 → ROI
- 输出: BGR ndarray

#### composite_to_pixmap
- 裁剪 view 区域 → resize 到 display_size → 贴到 canvas_pix 居中

#### ImageCanvas (QLabel 子类)
- 转发所有事件到回调 (on_wheel, on_mouse_press, ...)
- setMouseTracking(True), StrongFocus

---

### 4. sam_service.py — SAM 服务层

#### SamService 状态机
- 状态: _ready, _gen(generation 计数), _active_key, _cached_keys, _crop_info
- 优先级系统: _priority_lock + _priority_gen + _priority_key (线程安全)
- 生命周期:
  - load(ckpt, type, device) → 创建线程 → worker.do_load
  - shutdown() → quit + wait
- 图像管理:
  - invalidate_image() → gen++, 清除 active/pending
  - drop_cache() → 清除所有缓存
  - encode(rgb, key, crop_info) → 设置优先级 → cmd_encode
  - prefetch(key, path) → 不抬优先级 → cmd_prefetch
- 预测:
  - predict_async(x, y) → 若未编码则暂存 pending_prompt
  - predict_box_async(x1,y1,x2,y2) → 同上
- 信号回调:
  - _on_encode_done: 释放 busy → 加入 cached_keys → 排空 pending_prompt
  - _on_predict_done: 检查 gen, 发射 prediction_ready

#### 下载器 download_sam_checkpoint
- QDialog 带进度条
- urllib.request 分块下载, .part 临时文件
- 取消/失败时清理

---

### 5. workers.py — SAM 推理 Worker

#### SamInferenceWorker
- 拥有: SamPredictor + OrderedDict LRU 缓存
- 缓存容量: 默认 16 (环境变量 SAM_EMBEDDING_CACHE)
- 编码:
  - do_encode(gen, key, rgb):
    - stale check → cache hit → cache miss: _run_encoder + snapshot
    - _run_encoder: set_image (带 autocast, 带 NVML 重试)
    - _snapshot: 保存 features/original_size/input_size
- 预取:
  - do_prefetch(gen, key, path):
    - load_image_rgb → encode → restore 原 active
- 预测:
  - do_predict(gen, x, y, crop_info):
    - crop_info 含 "box" → box prompt
    - 否则 → point prompt (labels=[1])
    - multimask_output=True → argmax scores → threshold 0.5
    - _lift_to_full: 如果 crop 模式, 把小 mask 贴回全图
- 缓存逐出:
  - _evict_to_capacity: 不逐出 _active_key

---

### 6. yolo_service.py — YOLO 推理服务

#### YoloService
- 类似 SamService 的 QThread worker 架构
- load(weights) → worker.do_load → ultralytics.YOLO(path)
- predict(bgr, conf, replace) → worker.do_predict
- busy 标志防止并发请求

#### _build_prediction (结果解析)
- 优先使用 boxes.xyxy (或 obb.xyxy 退化)
- Mask 提取优先级:
  1. masks.data (raw float prototype) → bilinear resize → >0.5 → binary
  2. masks.xy (原图坐标多边形) → fillPoly
  3. masks.xyn (归一化多边形) → 缩放后 fillPoly
- 无 mask 时退化为 box
- sum<30 的 mask 视为无效

---

### 7. app_input.py — 输入处理

- _on_wheel: 1.12x 缩放
- _on_mouse_press:
  - Alt+左/中键 → 平移开始
  - ROI drawing → 添加/撤销顶点
  - Ctrl/Shift+左 → SAM predict
  - 左 → 拖拽画框开始
  - 右 → delete_at
- _on_mouse_move: 平移/拖拽更新/hover检测
- _on_mouse_release: 完成画框 (dx≥5 且 dy≥5)
- _on_key_press: 全部快捷键分发

---

### 8. app_sam.py — SAM+ROI 控制器

- _load_sam: checkpoint 验证 → 下载提示 → 加载
- _auto_load_sam: 启动时自动检测权重
- _encode_current_image: 构建 key(含 ROI bbox) → encode
- _schedule_encode: 防抖 350ms, cache hit 立即
- _schedule_prefetch: 防抖 900ms
- _prefetch_neighbors: 前后各 2 张
- _sam_predict: ROI check → lazy encode → predict_async
- _on_sam_prediction: ROI intersection → sum<30 丢弃 → add_mask 或 replace_box
- ROI 状态机: full → drawing → polygon → full
  - _roi_close: fillPoly → 可选 crop 编码
  - _roi_reset: 恢复全图

---

### 9. app.py — 主窗口

- UI 构建: 菜单栏 + 信息栏 + 三栏 Splitter + 日志
- 导航: _load_directory, _load_current_image, _next/_prev/_skip
- 保存: _save_current (双格式) + _autosave (编辑即存)
- 类别管理: add/delete/rename/relabel
- YOLO: predict → apply_yolo_predictions
- 事件分发 → mixin

---

## 二、测试实施计划（历史规划存档）

> 以下 Phase 1~4 是早期规划，仅 Phase 1（纯逻辑测试）已落地。当前实际测试覆盖见文末「实际测试矩阵」。

### Phase 1: 纯逻辑单元测试 (无 GUI 依赖)

| 测试文件 | 覆盖模块 | 测试要点 |
|----------|----------|----------|
| test_models.py | models.py | 已有 + 补充边界条件 |
| test_io_utils.py | io_utils.py | 已有 + 异常路径 |
| test_yolo_service.py | yolo_service.py | 修复失败 + 补充 |
| test_colors.py | colors.py | 颜色映射一致性 |

### Phase 2: 服务层 mock 测试（未实施）

| 测试文件 | 覆盖模块 | 测试要点 |
|----------|----------|----------|
| test_sam_service.py | sam_service.py | 状态机, cache, pending_prompt |
| test_workers.py | workers.py | 编码/预测/缓存/逐出 |

### Phase 3: GUI 组件测试 (pytest-qt)（未实施）

| 测试文件 | 覆盖模块 | 测试要点 |
|----------|----------|----------|
| test_sidebar.py | sidebar.py | 信号发射 |
| test_right_panel.py | right_panel.py | 类别列表操作 |
| test_canvas_widget.py | canvas.py (widget) | 事件转发 |
| test_app_input.py | app_input.py | 键鼠模拟 |

### Phase 4: 集成测试（部分实施 → `test_app_smoke.py`）

| 测试文件 | 场景 |
|----------|------|
| test_integration.py | 打开目录→标注→保存→重载验证 |

---

## 三、测试运行命令

```bash
# 全量运行
.venv/bin/python -m pytest tests/ -v

# 带覆盖率
.venv/bin/python -m pytest tests/ --cov=yolo_sam_labeler --cov-report=term-missing

# 只跑纯逻辑 (不需要 display)
.venv/bin/python -m pytest tests/ -v -k "not gui and not widget"
```


---

## 四、新增模块（v0.3.0）

### 10. backends/ — SAM 后端抽象层

#### SamBackend (base.py)
- 抽象基类，统一 SAM 1/2/3 的接口
- 核心方法:
  - `load(checkpoint, device)` — 加载权重
  - `set_image(rgb)` — 编码图像
  - `snapshot() / restore()` — 缓存 embedding
  - `predict_point(x, y, multimask)` — 点击预测
  - `predict_box(x1, y1, x2, y2, multimask)` — 框预测
- 属性:
  - `name` — 后端短名 (sam1/sam2)
  - `model_type_label` — 用户可见标签
  - `is_image_set` — 是否已编码
  - `supports_box` — 是否支持框 prompt
  - `supports_autocast(device)` — 是否启用 FP16

#### Sam1Backend (sam1.py)
- 包装 `segment_anything.SamPredictor`
- 文件名映射: sam_vit_h_4b8939 → vit_h, etc.
- `guess_sam1_model_type(path)` 推断 vit_h/l/b
- snapshot 字段: original_size, input_size, features, is_image_set

#### Sam2Backend (sam2.py)
- 包装 `sam2.SAM2ImagePredictor`
- 文件名映射: sam2.1_hiera_large → configs/sam2.1/sam2.1_hiera_l.yaml
- 支持 SAM 2.0 和 2.1 (推荐 2.1)
- snapshot 字段: _features, _orig_hw, _is_image_set, _is_batch

#### factory.py — 后端工厂
- `detect_backend_kind(path)` → "sam1" / "sam2" / "sam3"
  - 基于文件名前缀判断
  - sam2*/sam_2* → SAM 2
  - sam3* → SAM 3 (raise error，未支持)
  - 默认 → SAM 1
- `create_backend(path, model_type_hint)` → 实例化对应后端

### 11. weight_manager.py — 权重管理对话框

#### WeightManagerDialog
- 列出 7 个模型 (3 SAM 1 + 4 SAM 2.1)
- 每行: 名称 | 描述 | 大小 | 状态 | 下载按钮
- 状态判断: 文件存在 + 大小 ≥ 95% expected
- 下载: 非阻塞（QTimer 切片读 chunks）+ 进度条
- 默认目录: `~/.sam_weights/`
- 用户操作: 选行 → "使用选中的权重" → 关闭对话框 → app 加载

#### download_sam_checkpoint (sam_service.py)
- 兼容 SAM 1 和 SAM 2 URL
- URL 字典: SAM_MODEL_URLS + SAM2_MODEL_URLS
- 进度回调: QApplication.processEvents() 让 UI 响应
- 异常处理: 中断时清理 .part 文件

---

## 五、测试运行

```bash
# 全量测试 (176 个，自动 offscreen Qt)
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -v

# 仅纯逻辑 (无 Qt)
python -m pytest tests/test_models.py tests/test_models_extended.py \
                  tests/test_io_utils.py tests/test_io_utils_extended.py \
                  tests/test_yolo_service.py tests/test_backends.py tests/test_colors.py
```

### 实际测试矩阵

| 测试文件 | 数量 | 覆盖范围 |
|----------|------|---------|
| test_models.py | 数据模型基础 | Box / Mask / ClassRegistry / AnnotationStore |
| test_models_extended.py | 边界 + 异常 | 空注册表、稀疏 ID、replace_box_with_mask 验证 |
| test_io_utils.py | IO 基础 | 图像扫描、YOLO 编解码、classes.txt |
| test_io_utils_extended.py | label-storage | sniff、split_mixed、reconcile、cleanup_empty、SaveReport |
| test_canvas.py | CoordTransformer | 视图变换、缩放、平移 |
| test_canvas_extended.py | 渲染 | render_composite 图层顺序、composite_to_pixmap |
| test_colors.py | 颜色映射 | 一致性 |
| test_backends.py | SAM 后端 | factory 选择、模型类型推断 |
| test_yolo_service.py | YOLO 解析 | _build_prediction 多种 mask 表示形式回退 |
| test_app_smoke.py | MainWindow 决策树 | _seed_label_dirs 四分支、_apply_label_dir_choice、image_dir 切换清空 |
| **总计** | **176 个 pytest 用例** | |


---

## 标签目录配置（label-storage 子系统）

### 数据模型

`AnnotationStore` 持两个目录字段：

- `seg_dir`：分割（YOLO seg）标签所在目录。`""` 表示不读不写。
- `detect_dir`：检测（YOLO detect）标签所在目录。`""` 表示不读不写。

两者可以指向同一物理目录（共享布局），加载时按文件首条数据行的 token 数判定（5 列 = detect，1 + 2N ≥ 7 列 = seg）。

### 文件菜单入口

| 菜单项 | 行为 |
|---|---|
| 选择标签目录… | 嗅探 → 自动归类（seg/detect/mixed/empty）。混合时弹拆分对话框；空目录共享种子，首次保存收敛。 |
| 选择分割标签目录… | 强制设置 seg_dir；detect_dir 为空时自动 seed sibling。 |
| 选择检测标签目录… | 对称。 |
| 工具 → 整理标签目录… | 当前 seg_dir == detect_dir 且为混合时，触发拆分对话框。 |

### 关键场景预期

| 场景 | 预期 |
|---|---|
| 选空目录，画 box，保存 | 文件落到所选目录；seg_dir 自动改成 sibling（`<dir>_seg`），日志输出"标签格式已自动定为：检测"。重启后两字段均恢复。 |
| 选 detect-only 目录，跑 SAM 出 mask，保存 | mask 写入 sibling `<dir>_seg/`；detect 文件不变；重启后两类都还在。 |
| 选混合目录，点拆分 | 多数派留原地，少数派移到 sibling；seg_dir / detect_dir 各指一处；空文件 / 不识别 / classes.txt 都保留。 |
| 共享目录 + 同图同时含 seg 和 detect 标签 | 保存时 SaveReport.conflict_shared 置位，UI 弹窗让用户选哪一类留下。 |
| 切换图片目录到新项目 | 旧 image_dir 子树内的 seg/detect_dir 自动清空；外部路径保留。 |
| 没设标签目录就按 S | 弹"无法保存，请先选标签目录"，不创建文件、不丢数据。 |
| 旧版本 paths/label_dir 持久化 | 启动时一次性迁移到 seg_dir/detect_dir，旧键被清除。 |

### 数据安全保护

`_safe_overwrite` 在写入空内容时嗅探现存文件实际格式：

- 同格式 / 未知 → 清空写。
- 异格式 → **拒写**（数据丢失保护），SaveReport.refused_seg / refused_detect 置位，UI 输出警告。

### 测试覆盖

- `tests/test_io_utils.py` / `tests/test_io_utils_extended.py`：IO 单元 + SaveReport 字段 + 共享目录 + 拆分。
- `tests/test_app_smoke.py`：MainWindow 决策树（_seed_label_dirs 四分支、_apply_label_dir_choice 三种 kind、image_dir 切换清空）。
- 全仓库共 176 个 pytest 用例，其中 label-storage 子系统占大头。
