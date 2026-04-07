# 项目去屎山化完整阶段路线图

可以，下面给你一份“完整、彻底、详细，而且以不影响核心业务为第一原则”的整项目去屎山化路线图。

这份方案不是“重写项目”，而是“分阶段拆雷区”。核心方法是：

1. 先把现在能跑的业务行为锁死。
2. 再把大文件拆成稳定边界。
3. 每一阶段只做一种类型的重构。
4. 不在结构重构阶段顺手改算法、阈值、模型和 UI 产品逻辑。

注：路线图里提到的 `camera_runtime.py`、`camera_config.py`、`camera_state.py`、`gui_media.py`、`bootstrap_support.py` 是分阶段执行时的历史顶层文件名。当前仓库的真实实现已经收口到 `camera/runtime.py`、`camera/config.py`、`camera/state.py`、`desktop/media.py`、`bootstrap/support.py`。

## 总体目标

- 保持现有入口不变：[main.py](D:/PaddleOCR/DigitOCR_Project/main.py)、[gui_app.pyw](D:/PaddleOCR/DigitOCR_Project/gui_app.pyw)
- 保持现有核心业务方法签名不变：[recognition_service.py](D:/PaddleOCR/DigitOCR_Project/core/recognition_service.py#L155)、[recognition_service.py](D:/PaddleOCR/DigitOCR_Project/core/recognition_service.py#L202)、[recognition_service.py](D:/PaddleOCR/DigitOCR_Project/core/recognition_service.py#L217)、[recognition_service.py](D:/PaddleOCR/DigitOCR_Project/core/recognition_service.py#L394)
- 最终把当前三个核心屎山热点拆开：
  - [recognition_service.py](D:/PaddleOCR/DigitOCR_Project/core/recognition_service.py)
  - [runtime.py](D:/PaddleOCR/DigitOCR_Project/camera/runtime.py)
  - [gui_app.pyw](D:/PaddleOCR/DigitOCR_Project/gui_app.pyw)
- 最终去掉黑板模式历史遗留的自动定位旧链路
- 最终把“识别策略”“相机运行时”“GUI 控制器”“纯工具函数”分层

## 最终结构目标

```text
DigitOCR_Project/
  core/
    recognition_service.py        # 只保留门面和组装
    ocr_engine.py
    image_processor.py
    pipelines/
      image_pipeline.py
      handwriting_pipeline.py
      camera_digit_pipeline.py
      board_sequence_pipeline.py
    geometry.py
    result_mapping.py
  camera/
    __init__.py
    config.py
    state.py
    protocol.py
    roi.py
    overlay.py
    fast_path.py
    digit_loop.py
    board_loop.py
    worker_process.py
    runtime.py                    # 对外 facade
  desktop/
    media.py
    controllers/
      camera_controller.py
      image_controller.py
      handwriting_controller.py
      result_panel_controller.py
  bootstrap/
    support.py
  gui_app.pyw                     # 只保留入口、布局、装配
  main.py                         # 保留入口
```

## 分阶段路线

### Phase 0：建立安全网，不允许裸拆

- 目标：把当前业务行为固化，防止“重构成功，功能悄悄变了”。
- 任务：
  - 补一组完整的行为锁定测试，不测实现细节，只测输入输出和关键状态。
  - 锁定四条主链：图片识别、手写识别、摄像头数字模式、摄像头黑板模式。
  - 锁定 ROI 调整行为、结果展示格式、保存逻辑、黑板模式整排数字输出。
  - 建一个简单的回归命令清单，固定每阶段都要跑。
- 必跑测试：
  - `tests.test_camera_runtime`
  - `tests.test_image_mode`
  - `tests.test_save_behaviors`
  - `tests.test_ocr_engine_threads`
  - `tests.test_bootstrap_env`
  - `tests.test_bootstrap_support`
- 交付标准：
  - 当前所有主路径都有可复现测试覆盖。
  - 后续任何阶段只要测试全绿，就视为“核心业务未受影响”。
- 时间建议：2 到 3 天。

### Phase 1：先立边界，不先搬业务

- 目标：把未来的拆分骨架建起来，但不改行为。
- 任务：
  - 在 `core/pipelines/`、`camera/`、`desktop/controllers/` 下创建目标模块壳。
  - 把纯 dataclass、协议对象、纯工具函数优先迁出。
  - 先建立 facade 模式：旧入口保留，新模块先空转接。
  - 明确依赖方向：
    - GUI 只能依赖 runtime/service
    - runtime 只能依赖 service/engine/state
    - pipeline 不能反向依赖 GUI
- 交付标准：
  - 新目录结构存在。
  - 老入口和调用关系不变。
  - 代码行为零变化。
- 时间建议：1 到 2 天。

### Phase 2：拆 `DigitOCRService`，这是第一刀

- 目标：把识别策略从单个 God object 中拆开。
- 当前问题点：
  - 一个类同时处理图片、相机、黑板、手写、切分、几何映射、结构化面板。
- 任务：
  - 抽出纯几何和映射工具到 `core/geometry.py` / `core/result_mapping.py`
  - 抽出手写链路到 `core/pipelines/handwriting_pipeline.py`
  - 抽出图片链路到 `core/pipelines/image_pipeline.py`
  - 抽出摄像头数字链路到 `core/pipelines/camera_digit_pipeline.py`
  - 抽出黑板整排序列链路到 `core/pipelines/board_sequence_pipeline.py`
  - 保留 `DigitOCRService` 作为统一门面，只做：
    - 初始化 `engine`、`processor`
    - 分发到对应 pipeline
    - 返回现有结果对象
- 拆分顺序：
  - 先搬纯工具函数
  - 再搬手写
  - 再搬图片
  - 最后搬相机数字和黑板模式
- 严格不做：
  - 不改阈值
  - 不改 OCR 参数
  - 不改结果结构
  - 不改 public method
- 交付标准：
  - `DigitOCRService` 只剩门面逻辑
  - 原测试全部通过
  - 业务方法签名不变
- 时间建议：4 到 6 天。

### Phase 3：拆 `camera_runtime.py`，这是第二刀

- 目标：把实时相机的多职责大文件拆开。
- 当前问题点：
  - 一个文件混了 ROI、快路径、黑板模式、IPC、预览叠框、状态发布、相机后端选择。
- 任务：
  - 抽协议对象到 `camera/protocol.py`
    - `CameraOCRWorkerConfig`
    - `CameraOCRTask`
    - `CameraOCRWorkerResult`
    - `PendingFastFrame`
    - `BoardFramePlan`
  - 抽 ROI 和裁剪逻辑到 `camera/roi.py`
  - 抽预览叠框到 `camera/overlay.py`
  - 抽数字模式快路径和候选块逻辑到 `camera/fast_path.py`
  - 抽 worker 进程入口到 `camera/worker_process.py`
  - 抽数字模式 loop 到 `camera/digit_loop.py`
  - 抽黑板模式 loop 到 `camera/board_loop.py`
  - 保留 `CameraOCRRuntime` 为 facade，对 GUI 提供原接口：
    - `start`
    - `stop`
    - `get_snapshot`
    - `update_roi_size`
- 严格不做：
  - 不改 GUI 调用方式
  - 不改现有 `CameraSnapshot` / `CameraInferenceResult` 结构
  - 不顺手优化识别算法
- 交付标准：
  - `camera_runtime.py` 不再承载全部实现
  - GUI 完全无感知
  - 相机模式行为与现有一致
- 时间建议：5 到 7 天。

### Phase 4：清理黑板模式历史死代码

- 目标：把已经不参与主链路的旧自动黑板逻辑彻底清掉。
- 当前现状：
  - 黑板模式已经改成手动 ROI，但旧自动黑板函数还在。
- 清理对象重点在 [runtime.py](D:/PaddleOCR/DigitOCR_Project/camera/runtime.py)：
  - `detect_camera_board_polygon`
  - `smooth_camera_polygon`
  - `camera_polygon_iou`
  - `camera_polygon_shift_ratio`
  - `warp_camera_polygon_for_ocr`
  - `camera_blur_score`
  - `camera_frame_change_ratio`
  - `_publish_board_reuse_result`
  - `_record_board_success`
  - 相关 `scene_polygon` 旧路径依赖
- 做法：
  - 先 `rg` 确认引用
  - 删除主链路不再调用的函数
  - 更新测试，移除旧自动黑板断言
- 交付标准：
  - 黑板模式只剩“固定 ROI + 序列 OCR”一条主链
  - 不再同时存在一整套旧自动定位分支
- 时间建议：1 到 2 天。

### Phase 5：再拆 GUI，最后动它

- 目标：把单类超级控制器拆成多个 controller。
- 当前问题点：
  - [gui_app.pyw](D:/PaddleOCR/DigitOCR_Project/gui_app.pyw) 同时处理布局、状态、线程、相机、手写、图片、结果面板。
- 任务：
  - 抽相机控制器到 `desktop/controllers/camera_controller.py`
  - 抽图片上传控制器到 `desktop/controllers/image_controller.py`
  - 抽手写控制器到 `desktop/controllers/handwriting_controller.py`
  - 抽结果面板控制器到 `desktop/controllers/result_panel_controller.py`
  - `DigitOCRGuiApp` 保留为：
    - 创建 Tk 变量
    - 建 UI
    - 注入 controller
    - 绑定回调
- 拆分顺序：
  - 先相机 controller
  - 再图片 controller
  - 再手写 controller
  - 最后结果面板 controller
- 严格不做：
  - 不重做 UI
  - 不改现有窗口布局
  - 不重构视觉样式
- 交付标准：
  - `gui_app.pyw` 从“全能控制器”降为“装配入口”
  - 相机/图片/手写各自独立
- 时间建议：4 到 6 天。

### Phase 6：包结构归位，消除“假包真顶层”

- 目标：把当前一些“包装模块反向导入顶层实现”的结构理顺。
- 当前现状：
  - `camera/runtime.py`、`camera/config.py`、`camera/state.py`、`desktop/media.py`、`bootstrap/support.py` 是兼容包装层，真实实现却在顶层。
- 任务：
  - 把真实实现移回对应 package 内
  - 顶层文件改成兼容 shim，或者在最后阶段删掉
  - 新代码统一只从 package 路径导入
- 原则：
  - 这一步一定要在 Phase 2、3 稳定后再做
  - 不能一上来就动 import 路径全局大换血
- 交付标准：
  - 代码结构和目录结构语义一致
  - 顶层文件不再承载核心实现
- 时间建议：2 到 3 天。

### Phase 7：统一配置、状态文案和错误处理

- 目标：把散落常量、状态文案、模式差异逻辑收口。
- 任务：
  - 统一摄像头模式文案和状态提示来源
  - 把 ROI、FPS、worker 参数、黑板/数字模式差异规则整理成更可读的配置层
  - 统一错误消息构造方式
  - 统一结果摘要格式
- 说明：
  - 这一步不是改业务，是让后续维护的人能看懂。
- 交付标准：
  - 模式差异不再靠到处 scattered if 判断
  - 文案和状态收口到少量位置
- 时间建议：2 到 3 天。

### Phase 8：最后才做工程化收尾

- 目标：防止再次屎山化。
- 任务：
  - 增加简单的静态检查门槛
  - 建立“文件长度”和“函数长度”红线
  - 建立 PR 规则：重构 PR 不夹带新功能
  - 保留迁移说明，并在最终验收时删除顶层兼容层
- 推荐红线：
  - 单文件尽量不超过 700 行
  - 单类尽量不超过 350 行
  - 单函数尽量不超过 80 行
  - 新功能不得直接塞回 `gui_app.pyw`、`camera/runtime.py`、`recognition_service.py`
- 时间建议：1 到 2 天。

## 全流程中的硬性保护规则

- 每个阶段都必须保持这些方法签名不变：
  - `DigitOCRService.recognize_image`
  - `DigitOCRService.recognize_camera_frame`
  - `DigitOCRService.recognize_board_frame`
  - `DigitOCRService.recognize_handwriting`
  - `CameraOCRRuntime.start`
  - `CameraOCRRuntime.stop`
  - `CameraOCRRuntime.get_snapshot`
  - `DigitOCRGuiApp` 入口行为
- 每个阶段都必须跑固定回归：
  - `tests.test_camera_runtime`
  - `tests.test_image_mode`
  - `tests.test_save_behaviors`
  - `tests.test_ocr_engine_threads`
- 每个阶段只允许一种类型的改动：
  - 只搬代码
  - 或只删死代码
  - 或只拆 controller
  - 不允许“顺手优化”
- 任何阶段只要发现输出变了，但没有产品需求支撑，就回退。

## 明确的任务切分建议

- Phase 0 到 Phase 1 可以一组做
- Phase 2 单独做一组
- Phase 3 单独做一组
- Phase 4 单独做一组
- Phase 5 单独做一组
- Phase 6 到 Phase 8 可以按精力拆两组

## 建议执行顺序

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8

## 最重要的判断

- 这个项目不能靠“一次性大重构”解决。
- 只能靠“测试先行 + 门面保留 + 内部分层替换”解决。
- 真正的主战场不是 OCR 模型，而是三个大文件的职责拆解。

下面是具体的可执行清单请你参考

## 可执行任务清单版

这一版按“阶段 -> 步骤 -> 文件 -> 函数 -> 测试”组织，目标是可以直接开工，不再需要二次决策。

### 统一执行规则

- 每个阶段单独建分支，命名格式建议：
  - `refactor/phase-0-safety-net`
  - `refactor/phase-2-service-split`
  - `refactor/phase-3-camera-runtime-split`
- 每个阶段结束前，不允许把“结构重构”和“业务行为变更”混在同一个提交里。
- 每个阶段必须保留以下公开入口不变：
  - `main.py`
  - `gui_app.pyw`
  - `DigitOCRService.recognize_image`
  - `DigitOCRService.recognize_camera_frame`
  - `DigitOCRService.recognize_board_frame`
  - `DigitOCRService.recognize_handwriting`
  - `CameraOCRRuntime.start`
  - `CameraOCRRuntime.stop`
  - `CameraOCRRuntime.get_snapshot`
- 每个阶段固定回归命令：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_ocr_engine_threads`

### Phase 0：安全网与行为锁定

#### 先新建文件

- `tests/test_characterization_camera_digit.py`
- `tests/test_characterization_board_sequence.py`
- `tests/test_characterization_gui_camera.py`
- `tests/test_characterization_service_contracts.py`

#### 任务步骤

1. 锁定 `DigitOCRService` 对外行为
- 在 `tests/test_characterization_service_contracts.py` 中补以下场景：
  - `recognize_image` 返回 `RecognitionOutput`
  - `recognize_camera_frame` 返回 `list[OCRResult]`
  - `recognize_board_frame` 允许多字符结果
  - `recognize_handwriting` 返回 `RecognitionOutput`
- 不 mock 返回结构本身，只 mock OCR 引擎结果，验证输出形状和排序规则。
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_characterization_service_contracts`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode tests.test_ocr_engine_threads`

2. 锁定数字模式相机行为
- 在 `tests/test_characterization_camera_digit.py` 中补以下场景：
  - 固定 ROI 裁剪位置不变
  - `4-6` 位数字在数字模式下可识别
  - `7+` 位数字触发拥挤保护
  - fallback 结果仅在更完整时覆盖快路径结果
  - ROI 更新后旧 generation 结果丢弃
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_characterization_camera_digit`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

3. 锁定黑板模式相机行为
- 在 `tests/test_characterization_board_sequence.py` 中补以下场景：
  - 黑板模式使用固定手动 ROI
  - 黑板模式保留多字符结果
  - 黑板模式输出整排数字序列
  - 黑板模式不启动 fast worker
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_characterization_board_sequence`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

4. 锁定 GUI 相机交互行为
- 在 `tests/test_characterization_gui_camera.py` 中补以下场景：
  - 黑板模式 ROI 滑条和应用按钮可用
  - 黑板模式应用 ROI 后 summary/status 正确
  - 数字模式与黑板模式提示语不串线
  - 保存结果、切换 tab、关闭窗口不破坏当前逻辑
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_characterization_gui_camera`
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`

#### Phase 0 完成标准

- 新增 characterization 测试全绿。
- 现有测试全绿。
- 后续所有重构以这些测试为“业务未变”的判据。

### Phase 1：建立模块骨架，不搬业务

#### 先新建文件

- `core/pipelines/__init__.py`
- `core/pipelines/image_pipeline.py`
- `core/pipelines/handwriting_pipeline.py`
- `core/pipelines/camera_digit_pipeline.py`
- `core/pipelines/board_sequence_pipeline.py`
- `core/geometry.py`
- `core/result_mapping.py`
- `camera/protocol.py`
- `camera/roi.py`
- `camera/overlay.py`
- `camera/fast_path.py`
- `camera/digit_loop.py`
- `camera/board_loop.py`
- `camera/worker_process.py`
- `desktop/controllers/__init__.py`
- `desktop/controllers/camera_controller.py`
- `desktop/controllers/image_controller.py`
- `desktop/controllers/handwriting_controller.py`
- `desktop/controllers/result_panel_controller.py`
- `desktop/controllers/recognition_controller.py`

#### 任务步骤

1. 只创建模块与最小导出
- 每个新文件只写模块注释、占位类/函数、必要 import。
- 不修改现有调用点。
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime tests.test_image_mode`

2. 建立 `__all__` 与导入边界
- `core/pipelines/__init__.py` 暂时只导出占位 pipeline。
- `camera/protocol.py` 暂时只定义注释，不搬 dataclass。
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_ocr_engine_threads tests.test_save_behaviors`

#### Phase 1 完成标准

- 新目录结构存在。
- 原入口不变。
- 代码行为零变化。

### Phase 2：拆 `DigitOCRService`

#### 先新建/启用文件

- `core/geometry.py`
- `core/result_mapping.py`
- `core/pipelines/handwriting_pipeline.py`
- `core/pipelines/image_pipeline.py`
- `core/pipelines/camera_digit_pipeline.py`
- `core/pipelines/board_sequence_pipeline.py`

#### 搬迁顺序与函数清单

1. 先搬纯几何函数到 `core/geometry.py`
- 从 `core/recognition_service.py` 搬出：
  - `_combine_boxes`
  - `_expand_region_box`
  - `_region_box_to_polygon`
  - `_polygon_to_region_box`
  - `_box_width`
  - `_box_height`
  - `_box_area`
  - `_box_center`
  - `_intersection_area`
  - `_box_iou`
- 做法：
  - 新模块实现纯函数
  - 原类内函数先改成调用新模块，保持旧名字不删
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`

2. 再搬结果映射函数到 `core/result_mapping.py`
- 搬出：
  - `_remap_results`
  - `_sort_results`
- 做法：
  - 先抽成纯函数
  - 原类方法保留薄包装，调用新模块
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode tests.test_camera_runtime`

3. 拆手写识别链到 `core/pipelines/handwriting_pipeline.py`
- 第一批搬出直接入口：
  - `_recognize_handwriting_regions`
  - `_resolve_handwriting_blocks`
  - `_extract_handwriting_regions`
  - `_merge_component_boxes`
- 第二批搬出所有手写专属 helper，规则是全部移动这些命名前缀函数：
  - `_build_handwriting_*`
  - `_split_handwriting_*`
  - `_normalize_handwriting_*`
  - `_should_retry_handwriting_*`
  - `_handwriting_result_*`
  - `_validate_handwriting_content`
- `DigitOCRService.recognize_handwriting` 保留，内部改为调用 `HandwritingPipeline.run(...)`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`

4. 拆图片识别链到 `core/pipelines/image_pipeline.py`
- 第一批搬出图片主入口相关：
  - `_resolve_image_digit_results`
  - `_collect_image_candidates`
  - `_resolve_image_candidates`
  - `_resolve_image_candidate`
  - `_resolve_image_results_with_ocr_fallback`
- 第二批搬出结构化图像相关：
  - `_resolve_structured_photo_results`
  - 全部 `_resolve_structured_*`
  - 全部 `_collect_structured_*`
  - 全部 `_filter_structured_*`
  - 全部 `_group_region_boxes_*`
  - 全部 `_should_retry_structured_*`
- 第三批搬出图片候选复核相关：
  - `_estimate_projection_segment_count`
  - `_split_image_candidate_block`
  - 全部 `_build_image_*`
  - 全部 `_review_image_candidate_*`
  - `_aggregate_image_review_results`
  - `_determine_image_split_count`
- `DigitOCRService.recognize_image` 保留，内部改为调用 `ImageRecognitionPipeline.run(...)`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`

5. 拆摄像头数字链到 `core/pipelines/camera_digit_pipeline.py`
- 搬出：
  - `_resolve_camera_fast_path`
  - `_collect_camera_fast_candidate_boxes`
  - `_resolve_camera_fast_candidate`
  - `_resolve_camera_fallback_path`
- `DigitOCRService.recognize_camera_frame` 和 `_recognize_camera_frame_internal` 保留，对应改为调用 `CameraDigitPipeline`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime tests.test_image_mode`

6. 拆黑板整排序列链到 `core/pipelines/board_sequence_pipeline.py`
- 搬出：
  - `recognize_board_frame` 的主体实现
- `DigitOCRService.recognize_board_frame` 保留，对应改为调用 `BoardSequencePipeline.run(...)`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

#### Phase 2 完成标准

- `DigitOCRService` 只剩门面、装配和极少量兼容包装。
- 核心识别逻辑进入 4 个 pipeline 文件。
- 所有对外方法签名不变。

### Phase 3：拆 `camera_runtime.py`

#### 先新建/启用文件

- `camera/protocol.py`
- `camera/roi.py`
- `camera/fast_path.py`
- `camera/overlay.py`
- `camera/worker_process.py`
- `camera/digit_loop.py`
- `camera/board_loop.py`

#### 搬迁顺序与函数清单

1. 先搬协议对象到 `camera/protocol.py`
- 搬出 dataclass / state model：
  - `CameraTrack`
  - `CameraOCRWorkerConfig`
  - `CameraOCRTask`
  - `CameraOCRWorkerResult`
  - `PendingFastFrame`
  - `BoardFramePlan`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

2. 搬 ROI 与基础裁剪到 `camera/roi.py`
- 搬出：
  - `camera_roi_box`
  - `crop_camera_roi`
  - `camera_roi_foreground_ratio`
  - `camera_roi_has_foreground`
  - `_resize_for_ocr`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

3. 搬快路径候选块与检测结果处理到 `camera/fast_path.py`
- 搬出：
  - `extract_camera_fast_candidates`
  - `_extract_fast_component_boxes`
  - `_split_camera_fast_box`
  - `build_camera_detections_from_results`
  - `filter_camera_detections`
  - `stabilize_camera_detections`
  - `stable_camera_sequence`
  - `camera_detection_signature`
  - `camera_result_is_fresh`
- 如果需要一并搬辅助函数：
  - `_result_to_camera_box`
  - `_bbox_iou`
  - `_camera_box_area`
  - `_bbox_overlap_ratio`
  - `_camera_boxes_equivalent`
  - `_bbox_center_distance`
  - `_blend_camera_detection`
  - `_match_camera_track`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

4. 搬预览叠框到 `camera/overlay.py`
- 搬出：
  - `resize_camera_frame_for_preview`
  - `overlay_camera_detections`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime tests.test_save_behaviors`

5. 搬 worker 进程入口到 `camera/worker_process.py`
- 搬出：
  - `_apply_camera_worker_env_limits`
  - `camera_ocr_worker_main`
- `CameraOCRRuntime._start_ocr_worker` 改为从新模块调用
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

6. 搬数字模式 runtime loop 到 `camera/digit_loop.py`
- 第一批搬出内部任务调度：
  - `_handle_completed_fast_frame`
  - `_build_fallback_task`
  - `_submit_fast_frame`
  - `_replace_fast_task`
  - `_build_ocr_task`
  - `_apply_ocr_worker_result`
  - `_publish_empty_camera_result`
  - `_drain_ocr_worker_results`
  - `_submit_ocr_task`
- 第二批搬出数字模式执行循环：
  - `_local_inference_loop`
  - `_process_inference_loop`
  - `_run_ocr`
  - `_set_worker_error`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

7. 搬黑板模式 runtime loop 到 `camera/board_loop.py`
- 搬出：
  - `_submit_board_task`
  - `_prepare_board_frame_plan`
  - `_local_board_inference_loop`
  - `_process_board_inference_loop`
  - `_run_board_ocr`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

8. 收口 `CameraOCRRuntime`
- `camera/runtime.py` 中仅保留：
  - `CameraOCRRuntime`
  - 生命周期方法
  - `get_snapshot`
  - 相机打开/关闭相关最外层方法
- 内部通过委派对象调用 `digit_loop` / `board_loop`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime tests.test_save_behaviors`

#### Phase 3 完成标准

- `camera/runtime.py` 不再承载全部实现。
- 数字模式和黑板模式 loop 分居不同文件。
- GUI 不需要改调用方式。

### Phase 4：清理黑板模式旧自动定位死代码

#### 先清引用，再删文件内旧函数

1. 先 `rg` 验证无主链路引用
- 检查这些函数只剩 0 个或仅测试引用：
  - `detect_camera_board_polygon`
  - `smooth_camera_polygon`
  - `camera_polygon_iou`
  - `camera_polygon_shift_ratio`
  - `warp_camera_polygon_for_ocr`
  - `camera_blur_score`
  - `camera_frame_change_ratio`
  - `_order_camera_polygon`
  - `_polygon_array_to_tuple`
  - `_matrix_to_tuple`

2. 删除旧 board 自动定位函数
- 从相应模块中删除上面所有函数。
- 删除不再使用的 `_publish_board_reuse_result`、`_record_board_success`
- 删除不再使用的 board 状态字段：
  - `_board_scene_polygon`
  - `_board_last_detected_polygon`
  - `_board_last_success_polygon`
  - `_board_last_success_frame`
  - `_board_pending_reference_frame`
  - `_board_pending_frame_id`

3. 更新黑板模式测试
- 删除旧的自动黑板 polygon/stability 测试。
- 保留并强化：
  - 固定 ROI 裁剪测试
  - 多字符结果保留测试
  - board worker 独立测试

#### 本阶段测试

- `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`
- `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`

#### Phase 4 完成标准

- 代码库中只存在手动 ROI 黑板链路。
- 黑板模式无历史自动定位残留状态。

### Phase 5：拆 GUI 控制器

#### 先新建/启用文件

- `desktop/controllers/camera_controller.py`
- `desktop/controllers/image_controller.py`
- `desktop/controllers/handwriting_controller.py`
- `desktop/controllers/result_panel_controller.py`
- `desktop/controllers/recognition_controller.py`

#### 搬迁顺序与函数清单

1. 先拆 `recognition_controller.py`
- 搬出：
  - `_submit_recognition`
  - `_handle_recognition_success`
  - `_handle_recognition_error`
  - `_get_or_create_service`
  - `_toggle_actions`
  - `_set_status`
  - `_queue_status_update`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`

2. 拆 `camera_controller.py`
- 搬出：
  - `_build_camera_tab`
  - `_apply_camera_roi_size`
  - `_handle_camera_started`
  - `_update_board_camera_results`
  - `_set_camera_controls_state`
  - `_start_camera_session`
  - `_stop_camera_session`
  - `_schedule_camera_poll`
  - `_poll_camera_snapshot`
  - `_update_camera_results`
  - `_populate_camera_result_table`
- `DigitOCRGuiApp` 保留 widget 挂载和 controller 注入，不再承载全部 camera 逻辑
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors tests.test_camera_runtime`

3. 拆 `image_controller.py`
- 搬出：
  - `_build_upload_tab`
  - `_choose_image`
  - `_clear_uploaded_image`
  - `_handle_image_loaded`
  - `_handle_image_load_error`
  - `_recognize_uploaded_image`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors tests.test_gui_media`

4. 拆 `handwriting_controller.py`
- 搬出：
  - `_build_handwriting_tab`
  - `_reset_handwriting_surface`
  - `_handle_handwriting_canvas_configure`
  - `_sync_handwriting_surface_to_widget`
  - `_sync_handwriting_surface_to_size`
  - `_scale_stroke_history`
  - `_rebuild_handwriting_surface`
  - `_render_full_stroke`
  - `_draw_line_segment`
  - `_clamp_point_to_surface`
  - `_start_drawing`
  - `_continue_drawing`
  - `_stop_drawing`
  - `_draw_dot`
  - `_clear_canvas`
  - `_recognize_handwriting`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`

5. 拆 `result_panel_controller.py`
- 搬出：
  - `_copy_result_text`
  - `_restore_result_panel_from_current_output`
  - `_populate_result_table`
  - `_show_result_preview`
  - `_schedule_result_preview`
  - `_show_result_preview_when_idle`
  - `_reset_result_preview`
  - `_show_bgr_image`
  - `_save_result_image`
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`

6. 留在 `gui_app.pyw` 的函数
- 暂时只保留：
  - `__init__`
  - `_configure_styles`
  - `_build_ui`
  - `_handle_notebook_tab_changed`
  - `_handle_close`
  - `main`

#### Phase 5 完成标准

- `gui_app.pyw` 退化为装配层。
- 相机、图片、手写、结果面板逻辑分别进入 controller。

### Phase 6：包结构归位

#### 先新建/调整目标文件

- `camera/runtime.py` 改为真实实现 facade
- `camera/config.py` 改为真实配置实现
- `camera/state.py` 改为真实 state 实现
- `desktop/media.py` 改为真实实现
- `bootstrap/support.py` 改为真实实现

#### 任务步骤

1. 把真实实现迁回 package
- 迁移：
  - 顶层 `camera_runtime.py` -> `camera/runtime.py`
  - 顶层 `camera_config.py` -> `camera/config.py`
  - 顶层 `camera_state.py` -> `camera/state.py`
  - 顶层 `gui_media.py` -> `desktop/media.py`
  - 顶层 `bootstrap_support.py` -> `bootstrap/support.py`

2. 顶层文件改成兼容 shim
- 顶层旧文件只保留 re-export：
  - `from camera.runtime import *`
  - `from camera.config import *`
  - `from camera.state import *`
  - `from desktop.media import *`
  - `from bootstrap.support import *`

3. 新代码导入路径统一
- 所有新文件只允许导入 package 路径，不再导入顶层兼容文件。

#### 本阶段测试

- `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime tests.test_save_behaviors tests.test_gui_media`
- `.\.venv\Scripts\python.exe -m unittest tests.test_bootstrap_env tests.test_bootstrap_support`

#### Phase 6 完成标准

- 目录结构和实现位置语义一致。
- 顶层文件不再承载核心实现。

### Phase 7：统一配置、状态文案和错误处理

#### 先新建文件

- `camera/mode_profiles.py`
- `desktop/messages.py`
- `core/messages.py`

#### 任务步骤

1. 收口摄像头模式配置到 `camera/mode_profiles.py`
- 集中定义：
  - 数字模式与黑板模式的 `cpu_threads`
  - `enable_mkldnn`
  - `use_textline_orientation`
  - 识别间隔
  - OCR 输入尺寸
- 替换 GUI 和 runtime 中的 scattered mode-specific if。
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`

2. 收口 GUI 状态文案到 `desktop/messages.py`
- 统一：
  - 相机启动成功提示
  - 相机等待提示
  - ROI 应用提示
  - 黑板模式和数字模式的识别框提示
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`

3. 收口服务层和 OCR 错误文案到 `core/messages.py`
- 统一：
  - “未识别到数字”
  - 图片/手写/黑板模式警告文本
- 跑测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode tests.test_ocr_engine_threads`

#### Phase 7 完成标准

- 模式差异集中到 profile。
- 状态文案集中到消息模块。
- 新增逻辑不再需要到处复制字符串。

### Phase 8：工程化收尾与防回退

#### 先新建文件

- `CONTRIBUTING.md`
- `pyproject.toml`
- `tools/check_module_sizes.py`

#### 任务步骤

1. 建立静态规则
- 在 `pyproject.toml` 中增加最小规则：
  - 行长
  - 简单复杂度限制
  - import 排序
- 不启用会大面积重写旧文件的规则。

2. 建立文件尺寸检查
- `tools/check_module_sizes.py` 负责检查：
  - 单文件 > 700 行时失败
  - 单函数 > 80 行时给警告
  - 单类 > 350 行时给警告

3. 建立协作规则
- `CONTRIBUTING.md` 明确：
  - 重构 PR 不能夹带新功能
  - 新功能不允许直接塞回：
    - `gui_app.pyw`
    - `camera_runtime.py`
    - `core/recognition_service.py`
  - 每个 PR 必须附带回归命令

#### 本阶段测试

- 运行全量单元测试：
  - `.\.venv\Scripts\python.exe -m unittest`
- 运行工程检查：
  - `.\.venv\Scripts\python.exe tools/check_module_sizes.py`

#### Phase 8 完成标准

- 结构性红线被写进仓库。
- 后续开发很难再把逻辑直接堆回老大文件。

## 推荐提交粒度

- Phase 0：4 个提交
- Phase 1：2 个提交
- Phase 2：6 个提交
- Phase 3：8 个提交
- Phase 4：2 个提交
- Phase 5：6 个提交
- Phase 6：3 个提交
- Phase 7：3 个提交
- Phase 8：3 个提交

## 每阶段验收结论模板

每阶段结束时固定写 4 行：

- 本阶段是否改了公开接口：`是/否`
- 本阶段是否改了业务行为：`是/否`
- 本阶段新增文件列表：`...`
- 本阶段回归结果：`通过的测试命令列表`

## 一句话执行策略

先把测试补成业务护栏，再按“服务层 -> 相机运行时 -> 黑板死代码 -> GUI -> 包结构 -> 工程规则”的顺序拆，不要跳步，不要夹带功能优化。

## 重构完成后的代码文件删除清单

这一节定义的是“整个路线图全部完成后”的最终删除清单，不是中途就删除。

### 可以删除的代码文件

以下文件在 **Phase 6 完成且所有导入已切换到 package 路径之后** 可以删除：

- `camera_runtime.py`
- `camera_config.py`
- `camera_state.py`
- `gui_media.py`
- `bootstrap_support.py`

### 删除原因

- 这些文件当前是顶层实现文件，路线图的目标是把真实实现迁回各自 package：
  - `camera/runtime.py`
  - `camera/config.py`
  - `camera/state.py`
  - `desktop/media.py`
  - `bootstrap/support.py`
- 如果 Phase 6 结束后还保留顶层同名文件，它们就只剩“兼容 shim”意义。
- 一旦确认仓库内代码、测试、启动脚本、打包脚本都已经不再依赖这些顶层 shim，就应删除，避免再次出现“假包真顶层”的结构倒挂。

### 删除前必须满足的条件

删除上述文件前，必须同时满足以下条件：

1. 仓库内所有 Python 导入已切换到 package 路径：
- 不再有新代码导入：
  - `camera_runtime`
  - `camera_config`
  - `camera_state`
  - `gui_media`
  - `bootstrap_support`

2. 以下入口已验证可正常工作：
- `main.py`
- `gui_app.pyw`
- `launch_gui.bat`
- `launch_gui.sh`
- `DigitOCR_GUI.spec`

3. 以下测试必须全绿：
- `.\.venv\Scripts\python.exe -m unittest tests.test_camera_runtime`
- `.\.venv\Scripts\python.exe -m unittest tests.test_image_mode`
- `.\.venv\Scripts\python.exe -m unittest tests.test_save_behaviors`
- `.\.venv\Scripts\python.exe -m unittest tests.test_gui_media`
- `.\.venv\Scripts\python.exe -m unittest tests.test_bootstrap_env`
- `.\.venv\Scripts\python.exe -m unittest tests.test_bootstrap_support`

4. 打包流程已验证：
- `build_windows.ps1` 可正常执行
- `DigitOCR_GUI.spec` 中引用的模块路径已全部指向 package 内真实实现

### 不在“删除文件”范围内的对象

以下内容应当清理，但不属于“删除整个代码文件”的范围：

- `camera_runtime.py` / 后续 `camera/runtime.py` 中已废弃的自动黑板检测函数
- `core/recognition_service.py` 中被拆走后的兼容包装方法
- 旧测试中针对自动黑板定位逻辑的断言

这些属于“删除死代码函数/测试分支”，应在对应阶段完成，不等到最终删文件。

### 不能删除的代码文件

以下文件在路线图完成后仍然应该保留：

- `main.py`
- `gui_app.pyw`
- `core/ocr_engine.py`
- `core/image_processor.py`
- `core/recognition_service.py`
- `camera/runtime.py`
- `camera/config.py`
- `camera/state.py`
- `desktop/media.py`
- `bootstrap/support.py`

原因是它们分别承担：

- 稳定入口
- GUI 入口
- OCR 引擎封装
- 图像预处理
- 统一服务门面
- package 内真实运行时实现

### 推荐删除顺序

最终删顶层重复实现文件时，按以下顺序执行：

1. 删除 `gui_media.py`
2. 删除 `bootstrap_support.py`
3. 删除 `camera_config.py`
4. 删除 `camera_state.py`
5. 最后删除 `camera_runtime.py`

原因：

- `camera_runtime.py` 牵涉最广，风险最高，应最后删。
- `gui_media.py` 和 `bootstrap_support.py` 依赖面相对窄，适合先验证 package 化迁移是否彻底。

### 最终验收要求

删除完顶层重复实现文件后，再执行一次全量检查：

- `rg -n "import camera_runtime|from camera_runtime|import camera_config|from camera_config|import camera_state|from camera_state|import gui_media|from gui_media|import bootstrap_support|from bootstrap_support" .`
- `.\.venv\Scripts\python.exe -m unittest`
- Windows 打包流程验证一次

只有这三项全部通过，才视为“顶层重复实现文件删除完成”。
