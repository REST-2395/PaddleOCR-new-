# Contributing

## Scope Rules

- 一个 PR 只做一种类型的改动：只搬代码、只删旧代码、只拆 controller，不混入顺手优化
- 重构 PR 不夹带新功能
- 以下公开入口默认视为稳定接口，改动前必须评估兼容性
  - `main.py`
  - `gui_app.pyw`
  - `DigitOCRService.recognize_image`
  - `DigitOCRService.recognize_camera_frame`
  - `DigitOCRService.recognize_board_frame`
  - `DigitOCRService.recognize_handwriting`
  - `CameraOCRRuntime.start`
  - `CameraOCRRuntime.stop`
  - `CameraOCRRuntime.get_snapshot`
  - `HandCountRuntime.start`
  - `HandCountRuntime.stop`
  - `HandCountRuntime.get_snapshot`

## Placement Rules

- 新功能不要直接堆回以下入口或历史大文件
  - `gui_app.pyw`
  - `camera/runtime.py`
  - `core/recognition_service.py`
- 新增业务实现优先放入以下目录
  - `core/pipelines/`
  - `camera/`
  - `handcount/`
  - `desktop/controllers/`

## Size Guardrails

- 单文件超过 700 行视为失败
- 单类超过 350 行给出警告
- 单函数超过 80 行给出警告
- 提交前运行 `.\.venv\Scripts\python.exe tools/check_module_sizes.py`

## Regression Commands

- 每个 PR 都要在描述中附上实际执行过的回归命令
- 基础回归命令
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_ocr_engine_threads`
- 涉及摄像头手势计数时至少补跑
  - `.\.venv\Scripts\python.exe -m unittest tests.test_handcount_runtime`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_handcount_detector`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_handcount_overlay`
- 涉及摄像头 OCR 行为时建议补跑
  - `.\.venv\Scripts\python.exe -m unittest tests.test_characterization_camera_digit`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_characterization_board_sequence`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_characterization_gui_camera`
- 做结构性变更时还需要
  - `.\.venv\Scripts\python.exe -m unittest`
  - `.\.venv\Scripts\python.exe tools/check_module_sizes.py`

## Migration Notes

- 保留 `main.py` 和 `gui_app.pyw` 作为稳定入口，真实实现统一维护在包内模块
- 若结构调整影响后续开发路径，请同步更新 `README.md`、`docs/model_principle.md` 和相关验收文档
- 摄像头链路目前分为数字模式、黑板模式、手势计数模式，文档和测试都应覆盖这三条路径
