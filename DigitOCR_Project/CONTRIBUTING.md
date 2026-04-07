# Contributing

## Scope Rules

- 一个 PR 只做一种类型的改动：只搬代码、只删死代码、只拆 controller，不能顺手夹带功能优化。
- 重构 PR 不能夹带新功能。
- 公开入口必须保持稳定：
  - `main.py`
  - `gui_app.pyw`
  - `DigitOCRService.recognize_image`
  - `DigitOCRService.recognize_camera_frame`
  - `DigitOCRService.recognize_board_frame`
  - `DigitOCRService.recognize_handwriting`
  - `CameraOCRRuntime.start`
  - `CameraOCRRuntime.stop`
  - `CameraOCRRuntime.get_snapshot`

## Placement Rules

- 新功能不允许直接塞回以下门面入口或历史大文件：
  - `gui_app.pyw`
  - `camera/runtime.py`
  - `core/recognition_service.py`
- 新增业务实现应优先放入：
  - `core/pipelines/`
  - `camera/`
  - `desktop/controllers/`
  - 其他语义清晰的新模块

## Size Guardrails

- 单文件超过 700 行视为失败。
- 单类超过 350 行给出警告。
- 单函数超过 80 行给出警告。
- 使用 `.\.venv\Scripts\python.exe tools/check_module_sizes.py` 在提交前检查。

## Regression Commands

- 每个 PR 必须在描述中附带实际执行过的回归命令。
- 基础回归命令：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_ocr_engine_threads`
- 工程化或 Phase 8 相关改动还需要：
  - `.\.venv\Scripts\python.exe -m unittest`
  - `.\.venv\Scripts\python.exe tools/check_module_sizes.py`

## Migration Notes

- 保留 `main.py` 和 `gui_app.pyw` 作为稳定入口，真实实现统一维护在包内模块。
- 如果结构调整影响后续开发路径，请同步更新 `docs/phase_8_summary.md` 与最终验收文档。
