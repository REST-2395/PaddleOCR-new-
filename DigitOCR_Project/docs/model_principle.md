# 数字识别与手势计数原理说明

## 摘要

`DigitOCR_Project` 当前包含两类核心能力：

- 基于 PaddleOCR 的数字识别
- 基于 MediaPipe 的摄像头手势计数

系统入口仍然是桌面 GUI 和 CLI，但摄像头链路已经扩展为三种模式：

- 数字模式
- 黑板模式
- 手势计数模式

其中数字模式和黑板模式走 OCR 运行时，手势计数模式走独立的 `handcount/` 运行时。

## 总体架构

```text
DigitOCR_Project/
  core/                   OCR 服务、图像处理、识别流水线
  camera/                 摄像头 OCR 运行时、ROI、快路径、worker 控制
  handcount/              手势检测、计数运行时、预览叠加
  desktop/controllers/    GUI 控制器
  bootstrap/              运行环境自举
```

职责划分如下：

- `core/` 负责 PaddleOCR 相关识别能力
- `camera/` 负责摄像头数字模式和黑板模式
- `handcount/` 负责摄像头手势计数模式
- `desktop/controllers/` 负责界面交互、状态同步和结果展示

## 输入链路

### 1. 手写识别

- 用户在 Tk 画板上书写数字
- `desktop/controllers/handwriting_controller.py` 将笔迹同步到内存画布
- `DigitOCRService.recognize_handwriting()` 进入手写流水线
- 流水线对前景做分割、合并、旋转重试和必要的连写切分

### 2. 图片识别

- `desktop/controllers/image_controller.py` 负责选图与预览
- `DigitOCRService.recognize_image()` 调用图片流水线
- 流水线先做增强，再进行整图 OCR 与候选块复核

### 3. 摄像头数字模式

- `desktop/controllers/camera_controller.py` 创建 `CameraOCRRuntime`
- 运行时从画面中央取固定 ROI
- ROI 宽高比例可调，但位置始终固定在中央
- 先走快路径候选块识别
- 快路径不足时回退到完整 ROI OCR
- 最终结果经过低置信度过滤、跨帧稳定化后再显示

### 4. 摄像头黑板模式

- 仍由 `CameraOCRRuntime` 驱动
- 使用更高分辨率的采集和更大的 OCR 输入尺寸
- 同样使用中央 ROI
- 对 ROI 内整排数字做序列识别
- 不走数字模式的快路径 worker

### 5. 摄像头手势计数模式

- `desktop/controllers/camera_controller.py` 创建 `HandCountRuntime`
- `HandCountRuntime` 使用独立的采集线程和检测线程
- `handcount/detector.py` 基于 MediaPipe Hands 或 Tasks API 提取关键点
- 只统计中央 ROI 中的手
- 最多保留两只手，超出时返回警告
- 对总数做短历史稳定化后展示

## OCR 链路说明

### 统一服务层

`core/recognition_service.py` 目前主要承担门面职责：

- 创建 `ImageProcessor`
- 创建 `DigitOCREngine`
- 按需组织不同流水线
- 提供统一公开方法

稳定对外接口包括：

- `recognize_image`
- `recognize_handwriting`
- `recognize_camera_frame`
- `recognize_board_frame`

### 图像增强

`core/image_processor.py` 负责通用增强，主要包括：

- 放大
- 去噪
- 对比度增强
- 锐化

不同场景会使用不同参数组合，避免把所有输入都按同一套参数硬处理。

### 数字模式的快路径与回退

数字模式的目标是兼顾实时性和完整性：

1. 从中央 ROI 中提取候选块
2. 将候选块分发给 fast worker 做轻量识别
3. 如果候选数异常、质量不够或结果不完整，则提交 fallback ROI OCR
4. 当 fallback 结果比快路径更完整时，用 fallback 结果覆盖显示

对应模块：

- `camera/roi.py`
- `camera/fast_path.py`
- `camera/digit_loop.py`
- `camera/runtime_worker_control.py`

### 黑板模式的序列识别

黑板模式更关注整排数字：

- 直接对 ROI 做整段 OCR
- 保留多字符结果
- 不强制拆成单个数字块

对应模块：

- `camera/board_loop.py`
- `core/pipelines/board_sequence_pipeline.py`

## 手势计数原理

### 检测

`handcount/detector.py` 负责把 MediaPipe 输出转换成项目内部数据结构：

- 读取 21 个手部关键点
- 计算手框
- 解析左右手标签
- 修正镜像预览导致的左右手标签方向

最新实现已经修复了镜像画面中的 handedness 显示问题。

### 计数

每只手的计数规则是：

- 拇指根据左右手方向判断
- 其余四指通过指尖与中间关节的上下关系判断
- 五指状态求和得到单手数量

运行时输出：

- 每只手的左右手标签
- 每只手的数量
- 总数
- 超过两只手时的警告

### 稳定化

`handcount/runtime.py` 会维护一段短历史：

- 空帧不会立刻把稳定结果清空
- 同一总数重复达到阈值后，才将其作为稳定值发布
- 这样可以减少单帧抖动

### 预览叠加

`handcount/overlay.py` 在预览图上绘制：

- 中央 ROI
- 手框
- 手部骨架连接线
- 左右手标签
- 总数和 FPS

## 摄像头运行时设计

### OCR 运行时

`CameraOCRRuntime` 负责数字模式和黑板模式的生命周期：

- `start`
- `stop`
- `get_snapshot`
- `update_roi_size`

它将实现拆分给多个模块：

- `camera/runtime_lifecycle.py`
- `camera/runtime_loop_facade.py`
- `camera/runtime_worker_control.py`
- `camera/digit_loop.py`
- `camera/board_loop.py`

### 手势计数运行时

`HandCountRuntime` 保持了与 OCR 运行时相似的 facade 风格：

- `start`
- `stop`
- `get_snapshot`
- `update_roi_size`

这样 GUI 控制层可以以较统一的方式管理摄像头模式切换。

## ROI 设计

当前摄像头所有模式都基于居中的固定扫描框：

- ROI 不支持拖拽移动
- 只支持调整宽高比例
- 预览绘制和实际识别共用同一个 ROI 计算函数

对应实现位于 `camera/roi.py`。

数字模式和黑板模式还会先估算 ROI 的前景占比：

- 当前景不足时，主动跳过 OCR
- 减少空跑和误识别

## 结果展示与稳定化

GUI 不会把底层原始结果直接显示给用户，而是做一层控制：

- 低置信度过滤
- 结果摘要生成
- 表格映射
- 预览叠加
- 跨帧稳定化

对应模块主要是：

- `desktop/controllers/camera_controller.py`
- `desktop/controllers/result_panel_controller.py`
- `desktop/messages.py`

## 依赖说明

当前核心依赖包括：

- `paddleocr`
- `paddlepaddle`
- `opencv-python`
- `Pillow`
- `mediapipe`

其中：

- PaddleOCR 用于图片、手写、摄像头数字、黑板序列识别
- MediaPipe 用于手势计数模式

## 测试覆盖

与当前能力直接相关的测试包括：

- `tests.test_image_mode`
- `tests.test_camera_runtime`
- `tests.test_handcount_runtime`
- `tests.test_handcount_detector`
- `tests.test_handcount_overlay`
- `tests.test_characterization_camera_digit`
- `tests.test_characterization_board_sequence`
- `tests.test_characterization_gui_camera`

## 结论

当前版本已经不是单一的“数字 OCR 工具”，而是一个围绕数字识别和手势计数构建的桌面视觉应用：

- 手写和图片链路共享统一 OCR 服务
- 摄像头 OCR 链路拆成数字模式与黑板模式
- 手势计数链路独立运行，但与 GUI 保持一致的使用体验
- 中央 ROI、快路径回退、稳定化和预览叠加共同构成了当前实现的核心
