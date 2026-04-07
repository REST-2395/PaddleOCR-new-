# DigitOCR Project

基于 PaddleOCR 的数字识别工程，提供两种使用方式：

- 图形界面 GUI：支持手写识别、图片识别、摄像头识别
- 命令行 CLI：支持批量处理文件夹中的图片

项目已经内置运行环境自举逻辑。一般情况下，其他人拿到工程后不需要手动先配虚拟环境，只要按下面的方式启动即可。

## 适用场景

- 识别纸面或图片中的数字
- 识别手写数字
- 通过摄像头实时识别数字
- 黑板模式下识别一整排数字序列

## 主要功能

- 手写画板识别
- 本地图片上传识别
- 摄像头数字模式识别
- 摄像头黑板模式识别
- 识别结果表格展示
- 识别结果图片保存
- 识别文本复制到剪贴板
- 批量图片识别与结果导出

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

- `gui_app.pyw` 才是桌面图形界面入口
- 首次启动会自动检查 `.venv`、安装依赖，并在需要时准备 OCR 运行环境
- 首次启动较慢是正常现象

### 方式 2：批量处理图片

把要识别的图片放到 `data/input/` 目录，然后运行：

```powershell
.\.venv\Scripts\python.exe .\main.py
```

说明：

- [main.py](./main.py) 是命令行批量识别入口，不是 GUI
- 默认读取 `data/input/`
- 默认把带标注结果的图片写到 `data/output/`

## 图形界面使用说明

### 1. 手写识别

1. 打开 GUI
2. 切到“手写识别”页签
3. 在白色画板区域用鼠标按住左键书写数字
4. 点击“识别手写内容”
5. 在右侧查看识别结果、结果表格和预览图
6. 如需重写，可点击“清空画板”

建议：

- 数字之间尽量留一点间距
- 笔画不要太细或太淡

### 2. 图片识别

1. 切到“图片识别”页签
2. 点击“选择图片”
3. 选择本地图片文件
4. 点击“识别上传图片”
5. 在右侧查看结果摘要、表格和预览图

支持格式：

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`
- `.webp`

### 3. 摄像头识别

1. 切到“摄像头识别”页签
2. 选择设备编号
3. 选择识别模式
4. 点击“启动摄像头”
5. 将数字放入识别框内
6. 观察实时预览、识别框和结果表格
7. 使用 ROI 宽高滑条调整识别框大小
8. 点击“应用识别框”使新大小生效
9. 使用结束后点击“停止摄像头”

摄像头模式说明：

- 数字模式：适合识别单行、较清晰的数字
- 黑板模式：适合识别识别框内的一整排数字序列

## 命令行使用说明

### 默认批量识别

```powershell
.\.venv\Scripts\python.exe .\main.py
```

### 指定输入输出目录

```powershell
.\.venv\Scripts\python.exe .\main.py --input-dir .\data\input --output-dir .\data\output
```

### 常用参数

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
  bootstrap/              启动前运行环境自举支持
  camera/                 摄像头识别运行时
  config/                 字典与环境配置
  core/                   OCR 服务、引擎、识别管线
  data/
    input/                默认输入图片目录
    output/               默认输出图片目录
  desktop/                GUI 控制器与桌面媒体逻辑
  docs/                   项目文档与重构文档
  tests/                  单元测试与特征测试
  gui_app.pyw             GUI 入口
  launch_gui.bat          Windows GUI 启动脚本
  launch_gui.sh           Linux / macOS GUI 启动脚本
  main.py                 CLI 批量识别入口
  requirements.txt        Python 依赖
```

## 自举机制说明

项目带有自动环境准备流程，入口会先调用 [bootstrap/support.py](./bootstrap/support.py)：

- 自动检查 Python 运行环境
- 自动创建 `.venv`
- 自动安装 `requirements.txt` 中的依赖
- 自动切换到项目 `.venv` 中的解释器

相关文件：

- [bootstrap_env.py](./bootstrap_env.py)
- [bootstrap/support.py](./bootstrap/support.py)
- [config/env_bootstrap.json](./config/env_bootstrap.json)

如果只是普通使用，通常不需要单独操作这些文件。

## 首次运行会发生什么

首次启动时，程序可能会：

- 创建 `.venv`
- 安装 Python 依赖
- 检查运行环境
- 准备 PaddleOCR 相关模型

因此首次启动可能比较慢，属于正常现象。

模型缓存目录通常位于：

- `.runtime/paddlex_cache/`

运行日志与报告通常位于：

- `.runtime/bootstrap/logs/`
- `.runtime/bootstrap/reports/`

## 测试与验收

运行全量测试：

```powershell
.\.venv\Scripts\python.exe -m unittest
```

运行主要识别链路相关测试：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_image_mode.ImageModeIntegrationTests tests.test_camera_runtime tests.test_characterization_camera_digit tests.test_characterization_board_sequence
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

请优先检查：

- 是否安装了 Python 3.10+
- 项目路径是否包含权限或杀毒软件限制
- 是否能在 PowerShell 中运行 `.\launch_gui.bat`

如果仍不行，可直接尝试：

```powershell
.\.venv\Scripts\pythonw.exe .\gui_app.pyw
```

### 3. 首次启动很慢

这是正常的，通常是因为：

- 正在创建虚拟环境
- 正在安装依赖
- 正在准备 OCR 模型

### 4. 摄像头打不开

请检查：

- 摄像头是否被其他程序占用
- 设备编号是否正确
- 是否给了系统摄像头权限

### 5. 识别效果不理想

建议优先调整：

- 图片清晰度
- 光照条件
- 数字与背景对比度
- 摄像头识别框大小
- 数字是否位于识别框中央

## 给交付使用者的建议

如果是把这个工程交给别人直接使用，推荐这样说明：

1. 先解压整个项目目录，不要只拿单个 `.py` 文件
2. Windows 用户优先双击 `launch_gui.bat`
3. 批量识别图片时运行 `main.py`
4. 不要删除 `.runtime`、`config`、`data` 目录
5. 首次运行慢一点是正常的

## 相关文档

- [docs/refactor_roadmap.md](./docs/refactor_roadmap.md)
- [docs/final_refactor_acceptance.md](./docs/final_refactor_acceptance.md)
- [docs/environment_bootstrap.md](./docs/environment_bootstrap.md)

## 许可证

当前仓库未单独声明许可证。如需对外分发，请先补充明确的 License 文件。
