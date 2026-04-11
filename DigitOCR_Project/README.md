# DigitOCR Project

基于 PaddleOCR 的数字识别工程，提供桌面 GUI 和命令行两种使用方式。

- GUI 支持手写数字识别、图片识别、摄像头数字识别、摄像头黑板识别、摄像头手势计数
- CLI 支持批量处理 `data/input/` 目录下的图片
- 项目内置运行环境自举流程，首次启动会自动准备 `.venv`、依赖和模型缓存

## 适用场景

- 识别纸面或截图中的数字
- 识别手写数字
- 通过摄像头实时识别数字
- 通过摄像头识别扫描框内的一整排数字序列
- 通过摄像头统计双手手指数

## 主要功能

- 手写画板识别
- 本地图片上传识别
- 摄像头数字模式
  - 居中固定扫描框
  - 快路径候选块识别 + 完整 ROI fallback OCR
  - 低置信度过滤与结果稳定化
- 摄像头黑板模式
  - 更高分辨率采集
  - 对扫描框内整排数字做序列识别
- 摄像头手势计数模式
  - 基于 MediaPipe 的双手关键点检测
  - 统计左右手伸出的手指数
  - 超过两只手时给出提示
- 识别结果表格展示
- 结果图片保存
- 识别文本复制到剪贴板

## 环境要求

- Python 3.10 或更高版本
- Windows 推荐直接使用 `launch_gui.bat`
- Linux / macOS 可使用 `launch_gui.sh`

当前运行依赖见 [requirements.txt](./requirements.txt)：

- `numpy`
- `opencv-python`
- `paddlepaddle`
- `paddleocr`
- `Pillow`
- `mediapipe`

## 快速开始

### 方式 1：启动图形界面

Windows：

```powershell
.\launch_gui.bat
```

Linux / macOS：

```bash
bash ./launch_gui.sh
```

也可以直接运行：

```powershell
.\.venv\Scripts\pythonw.exe .\gui_app.pyw
```

说明：

- `gui_app.pyw` 是桌面 GUI 入口
- 首次启动会先检查 `.venv`、安装依赖，并在需要时准备 OCR 运行环境
- 首次启动较慢属于正常现象

### 方式 2：批量处理图片

将要识别的图片放入 `data/input/`，然后运行：

```powershell
.\.venv\Scripts\python.exe .\main.py
```

说明：

- [main.py](./main.py) 是命令行批量识别入口，不会打开 GUI
- 默认读取 `data/input/`
- 默认将带标注结果图写入 `data/output/`

## 图形界面说明

### 1. 手写识别

1. 打开 GUI。
2. 切换到“手写识别”页签。
3. 在白色画板区域书写数字。
4. 点击“识别手写内容”。
5. 在右侧查看摘要、结果表格和预览图。

建议：

- 数字之间尽量保留间距
- 笔画不要过细

### 2. 图片识别

1. 切换到“图片识别”页签。
2. 点击“选择图片”。
3. 选择本地图片文件。
4. 点击“识别上传图片”。
5. 在右侧查看识别摘要、结果表格和预览图。

支持格式：

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`
- `.webp`

### 3. 摄像头模式

1. 切换到“摄像头识别”页签。
2. 选择设备编号。
3. 选择模式。
4. 按需调整扫描框宽高比例。
5. 点击“应用识别框”。
6. 点击“启动摄像头”。
7. 观察实时预览、状态栏和右侧结果区域。
8. 使用结束后点击“停止摄像头”。

当前摄像头共有三种模式：

- 数字模式：适合识别扫描框中的单个或少量数字，使用快路径候选块识别，并在必要时回退到完整 ROI OCR
- 黑板模式：适合识别扫描框中的整排数字，使用更大的输入尺寸与序列 OCR
- 手势计数模式：适合统计最多两只手的手指数，显示左右手标签、关键点和总数

补充说明：

- 扫描框始终位于画面中央，只调整宽高比例，不支持拖拽移动
- 数字模式和黑板模式在扫描框前景不足时会主动跳过 OCR，减少空跑
- 手势计数模式只统计扫描框内的手

## 命令行使用说明

### 默认批量识别

```powershell
.\.venv\Scripts\python.exe .\main.py
```

### 指定输入输出目录

```powershell
.\.venv\Scripts\python.exe .\main.py --input-dir .\data\input --output-dir .\data\output
```

### 查看帮助

```powershell
.\.venv\Scripts\python.exe .\main.py --help
```

常见参数：

- `--input-dir`：输入图片目录
- `--output-dir`：输出目录
- `--dict-path`：数字字典路径
- `--ocr-version`：OCR 模型版本，默认 `PP-OCRv5`
- `--score-threshold`：最低置信度阈值
- `--cpu-threads`：CPU 线程数
- `--use-gpu`：启用 GPU 推理
- `--det-model-dir` / `--rec-model-dir` / `--cls-model-dir`：自定义模型目录

## 目录说明

```text
DigitOCR_Project/
  bootstrap/              启动前环境自举支持
  camera/                 摄像头 OCR 运行时与模式配置
  config/                 字典与环境配置
  core/                   OCR 服务、引擎与识别流水线
  data/
    input/                默认输入图片目录
    output/               默认输出图片目录
  desktop/                GUI 控制器与界面消息
  docs/                   项目文档
  handcount/              手势计数运行时、检测与叠加绘制
  tests/                  单元测试与特征测试
  gui_app.pyw             GUI 入口
  launch_gui.bat          Windows GUI 启动脚本
  launch_gui.sh           Linux / macOS GUI 启动脚本
  main.py                 CLI 批量识别入口
  requirements.txt        Python 依赖
```

## 环境自举说明

项目带有自动环境准备流程，`main.py` 和 `gui_app.pyw` 会先调用 [bootstrap/support.py](./bootstrap/support.py)：

- 自动检查 Python 运行环境
- 自动创建 `.venv`
- 自动安装 `requirements.txt` 中的依赖
- 自动切换到项目 `.venv` 中的解释器

相关文件：

- [bootstrap_env.py](./bootstrap_env.py)
- [bootstrap/support.py](./bootstrap/support.py)
- [config/env_bootstrap.json](./config/env_bootstrap.json)

## 首次运行会发生什么

首次启动时，程序可能会：

- 创建 `.venv`
- 安装 Python 依赖
- 检查运行环境
- 下载 PaddleOCR 相关模型
- 准备 MediaPipe 运行所需资源

因此首次启动较慢属于正常现象。

常见缓存与日志目录：

- 模型缓存：`.runtime/paddlex_cache/`
- 自举日志：`.runtime/bootstrap/logs/`
- 自举报告：`.runtime/bootstrap/reports/`

## 测试

运行全量测试：

```powershell
.\.venv\Scripts\python.exe -m unittest
```

运行主要链路测试：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_image_mode tests.test_camera_runtime tests.test_handcount_runtime tests.test_handcount_detector tests.test_handcount_overlay tests.test_characterization_camera_digit tests.test_characterization_board_sequence tests.test_characterization_gui_camera
```

## 打包

Windows 打包脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

打包配置文件：

- [DigitOCR_GUI.spec](./DigitOCR_GUI.spec)

## 常见问题

### 1. 运行 `main.py` 后没有打开界面

这是正常的。

- [main.py](./main.py) 是命令行批量处理入口
- 图形界面请使用 [launch_gui.bat](./launch_gui.bat) 或 [gui_app.pyw](./gui_app.pyw)

### 2. 双击 `launch_gui.bat` 没反应

优先检查：

- 是否安装了 Python 3.10+
- 项目路径是否受到权限或安全软件限制
- 是否能在 PowerShell 中运行 `.\launch_gui.bat`

仍不行时可直接尝试：

```powershell
.\.venv\Scripts\pythonw.exe .\gui_app.pyw
```

### 3. 首次启动很慢

通常是因为正在：

- 创建虚拟环境
- 安装依赖
- 下载 OCR 模型
- 初始化 MediaPipe / Paddle 相关资源

### 4. 摄像头打不开

请检查：

- 摄像头是否被其他程序占用
- 设备编号是否正确
- 是否授予了系统摄像头权限

### 5. 识别效果不理想

建议优先调整：

- 图像清晰度
- 光照条件
- 数字与背景对比度
- 扫描框宽高比例
- 目标是否位于扫描框中央

### 6. 手势计数不稳定

建议优先检查：

- 画面中是否只保留一到两只手
- 手部是否完整位于扫描框内
- 左右手是否有明显遮挡
- 光照是否足够

## 相关文档

- [docs/model_principle.md](./docs/model_principle.md)
- [docs/refactor_roadmap.md](./docs/refactor_roadmap.md)
- [docs/final_refactor_acceptance.md](./docs/final_refactor_acceptance.md)
- [docs/environment_bootstrap.md](./docs/environment_bootstrap.md)

## 许可

当前仓库未单独声明 License。若需要对外分发，请先补充明确的许可文件。
