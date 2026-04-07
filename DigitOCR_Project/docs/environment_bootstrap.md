# 运行环境自举工具说明

## 文档目标

本项目内置了一套“启动前自动环境配置工具”。在主程序真正启动之前，它会自动识别当前操作系统、准备 `.venv` 虚拟环境、解析依赖清单、安装运行依赖、写入详细日志，并在失败时生成结构化错误报告。

这套工具的目标是尽量把“手动配环境”的步骤收敛为一次启动动作，让项目在干净环境中也能更顺畅地完成部署。

## 已交付文件

- `bootstrap_env.py`：跨平台环境自举主入口。
- `bootstrap/support.py`：供 `main.py` 和 `gui_app.pyw` 调用的启动接管辅助模块。
- `config/env_bootstrap.json`：可配置的依赖清单与系统依赖定义文件。
- `launch_gui.bat`：Windows 图形界面启动脚本，启动前会先执行环境检查。
- `launch_gui.sh`：Linux/macOS 图形界面启动脚本，启动前会先执行环境检查。
- `launch_gui.command`：macOS 双击启动包装脚本。

## 当前支持能力

1. 自动识别 Windows、macOS、Linux，并选择当前系统可用的包管理器来处理可选系统依赖。
2. 自动解析 `config/env_bootstrap.json` 中声明的 `requirements.txt` 和 `package.json`。
3. 下载和安装命令内置失败重试与递增退避机制。
4. 在 `.runtime/cache` 下保留本地缓存，网络异常时可回退使用已有 pip 缓存。
5. 支持 `--force-reinstall`、`--skip-installed`、`--offline`、`--skip-system-deps`、`--skip-bootstrap-packages`、`--dry-run` 等命令行参数。
6. 日志输出到 `.runtime/bootstrap/logs/`，结构化 JSON 报告输出到 `.runtime/bootstrap/reports/`。

## 启动前执行流程

`main.py` 和 `gui_app.pyw` 会在导入 `cv2`、`numpy`、`Pillow`、PaddleOCR 等运行依赖之前，先调用 `bootstrap.support.ensure_runtime_ready()`。

该流程会执行以下动作：

1. 调用 `bootstrap_env.py` 检查当前环境。
2. 如果项目虚拟环境 `.venv` 不存在，则自动创建。
3. 根据依赖清单自动安装或更新需要的运行依赖。
4. 如果当前解释器不是项目 `.venv` 中的解释器，则自动切换到 `.venv` 解释器重新启动主程序。

这意味着在干净环境里，也可以直接从下面这些入口启动，而不需要先手动创建虚拟环境：

- `python main.py`
- `launch_gui.bat`
- `bash launch_gui.sh`

## 配置文件说明

环境自举配置文件位于 `config/env_bootstrap.json`。

当前默认配置如下：

- `requirements.txt`：作为必需的 Python 运行时依赖清单。
- `package.json`：作为可选的 Node 运行时依赖清单。

如果后续项目新增前端、辅助工具链或系统层依赖，可以继续在该配置文件里扩展更多 manifest 或平台包映射。对于被标记为可选的清单文件，如果文件本身不存在，自举工具会自动跳过，不会阻塞启动。

## 常用命令

初始化或刷新项目运行环境：

```bash
python bootstrap_env.py --project-root .
```

仅在依赖清单发生变化时才重新安装：

```bash
python bootstrap_env.py --project-root . --skip-installed
```

强制重新安装所有 manifest 中声明的依赖：

```bash
python bootstrap_env.py --project-root . --force-reinstall
```

仅使用本地缓存安装：

```bash
python bootstrap_env.py --project-root . --offline
```

仅预览执行动作，不真正执行安装：

```bash
python bootstrap_env.py --project-root . --dry-run --skip-system-deps --skip-bootstrap-packages
```

## 日志与报告

- 日志文件：`.runtime/bootstrap/logs/bootstrap-<时间戳>.log`
- JSON 报告：`.runtime/bootstrap/reports/bootstrap-<时间戳>.json`
- 状态缓存：`.runtime/bootstrap/state.json`

其中 `state.json` 会保存依赖清单指纹，用来支持 `--skip-installed` 的快速跳过逻辑。

## 性能目标说明

这套工具为了提升重复启动速度，主要依赖以下机制：

- 复用已有 `.venv`
- 基于 manifest 指纹跳过未变更依赖
- 复用本地 pip/npm 缓存
- 下载失败后先重试再退出

在普通宽带网络和 CPU 版本依赖的前提下，首次部署通常应能在数分钟内完成；后续再次启动时，如果依赖清单没有变化，整体耗时会显著缩短。
