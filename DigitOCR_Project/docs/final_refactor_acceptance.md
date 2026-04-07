# 重构最终验收

本文记录 `docs/refactor_roadmap.md` 中 Phase 之外的最终收尾动作，也就是删除顶层重复 shim、确认 canonical 导入路径、并补齐最终验收记录。

## 已完成动作

- 删除顶层 shim 文件：
  - `camera_runtime.py`
  - `camera_config.py`
  - `camera_state.py`
  - `gui_media.py`
  - `bootstrap_support.py`
- 保留的稳定入口：
  - `main.py`
  - `gui_app.pyw`
- 保留的真实实现路径：
  - `camera/runtime.py`
  - `camera/config.py`
  - `camera/state.py`
  - `desktop/media.py`
  - `bootstrap/support.py`

## 验收命令

- 导入扫描：
  - `rg -n --glob '*.py' --glob '*.pyw' --glob '!dist/**' --glob '!.venv/**' --glob '!build/**' --glob '!__pycache__/**' "from (camera_runtime|camera_config|camera_state|gui_media|bootstrap_support) import|import (camera_runtime|camera_config|camera_state|gui_media|bootstrap_support)"`
- 全量测试：
  - `.\.venv\Scripts\python.exe -m unittest`
- 工程检查：
  - `.\.venv\Scripts\python.exe tools/check_module_sizes.py`
- Windows 打包验证：
  - `powershell -ExecutionPolicy Bypass -File .\build_windows.ps1`

## 当前结论

- 代码路径已经统一到 package 内真实实现。
- 顶层重复 shim 已删除，不再保留“假包真顶层”结构。
- 如后续继续做结构收口，应直接在 package 模块内推进，不再恢复根目录 shim。

## 本次验收结果

- 验收日期：
  - `2026-03-21`
- 导入扫描：
  - 代码文件中对 `camera_runtime`、`camera_config`、`camera_state`、`gui_media`、`bootstrap_support` 的导入命中数为 `0`
- 入口检查：
  - `gui_app.pyw` 已完成源码级导入验证
  - `main.py --help` 正常输出参数说明
- 全量测试：
  - `.\.venv\Scripts\python.exe -m unittest`
  - 结果：`Ran 120 tests ... OK`
- 工程检查：
  - `.\.venv\Scripts\python.exe tools/check_module_sizes.py`
  - 结果：`Failures: none`
  - 说明：仍有若干类/函数长度 warning，但不影响当前最终验收通过
- Windows 打包验证：
  - `powershell -ExecutionPolicy Bypass -File .\build_windows.ps1`
  - 结果：打包成功，输出目录为 `dist/DigitOCR_GUI`
