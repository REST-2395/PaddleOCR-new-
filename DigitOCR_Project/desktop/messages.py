"""Shared GUI-facing copy for the desktop application."""

from __future__ import annotations

from camera.config import CAMERA_MODE_BOARD, CAMERA_MODE_DIGIT, CAMERA_MODE_HAND_COUNT


APP_READY_STATUS = "已就绪。你可以手写数字，或者上传图片进行识别。"
RESULT_PLACEHOLDER_TEXT = "识别结果会显示在这里。"
CAMERA_IDLE_STATUS = "摄像头空闲。"
CAMERA_IDLE_RUNTIME_STATUS = "摄像头空闲"
CAMERA_STOPPING_STATUS = "正在停止摄像头..."
CAMERA_PREVIEW_PLACEHOLDER_TEXT = "此处将显示摄像头实时预览。"
CAMERA_OPENING_PREVIEW_TEXT = "正在打开摄像头..."
CAMERA_ROI_APPLIED_STATUS = "已应用新的识别框大小。"
CAMERA_START_FAILED_STATUS = "启动摄像头失败。"
CAMERA_STOP_FAILED_STATUS = "停止摄像头失败。"
CAMERA_TEXT_COPIED_STATUS = "识别文本已复制到剪贴板。"
CAMERA_WARNING_RUNTIME_STATUS = "摄像头警告"

CAMERA_MODE_LABELS = {
    CAMERA_MODE_DIGIT: "数字模式",
    CAMERA_MODE_BOARD: "黑板模式",
    CAMERA_MODE_HAND_COUNT: "手势计数模式",
}
CAMERA_MODE_BY_LABEL = {label: mode for mode, label in CAMERA_MODE_LABELS.items()}


def camera_mode_label(mode: str) -> str:
    return CAMERA_MODE_LABELS.get(mode, CAMERA_MODE_LABELS[CAMERA_MODE_DIGIT])


def camera_prompt_text(mode: str, *, trailing_period: bool = True) -> str:
    if mode == CAMERA_MODE_BOARD:
        prompt = "请将黑板上的数字放入识别框"
    elif mode == CAMERA_MODE_HAND_COUNT:
        prompt = "请将手放入计数框内"
    else:
        prompt = "请将数字放入识别框"
    return f"{prompt}。" if trailing_period else prompt


def camera_empty_summary(mode: str) -> str:
    if mode == CAMERA_MODE_BOARD:
        return "未检测到识别框内的黑板数字。"
    if mode == CAMERA_MODE_HAND_COUNT:
        return "未检测到手"
    return camera_prompt_text(mode)


def camera_started_status(device_index: int, mode: str) -> str:
    if mode in {CAMERA_MODE_BOARD, CAMERA_MODE_HAND_COUNT}:
        return f"已启动摄像头 {device_index}，当前为{camera_mode_label(mode)}。"
    return f"已启动摄像头 {device_index}。"


def _camera_device_label(device_index: int, backend_name: str) -> str:
    backend_suffix = f" ({backend_name})" if backend_name else ""
    return f"摄像头 {device_index}{backend_suffix}"


def _with_trailing_period(text: str, *, trailing_period: bool) -> str:
    if trailing_period and not text.endswith("。"):
        return f"{text}。"
    return text


def camera_starting_status(device_index: int) -> str:
    return f"正在启动摄像头 {device_index}..."


def camera_running_status(device_index: int, backend_name: str, *, trailing_period: bool = True) -> str:
    return _with_trailing_period(
        f"{_camera_device_label(device_index, backend_name)} 运行中",
        trailing_period=trailing_period,
    )


def camera_waiting_status(mode: str, device_index: int, backend_name: str, *, trailing_period: bool = False) -> str:
    if mode == CAMERA_MODE_BOARD:
        message = f"{_camera_device_label(device_index, backend_name)} 等待识别框内黑板数字"
    elif mode == CAMERA_MODE_HAND_COUNT:
        message = f"{_camera_device_label(device_index, backend_name)} 等待计数框内的手势"
    else:
        message = f"{_camera_device_label(device_index, backend_name)} 等待识别框内数字"
    return _with_trailing_period(message, trailing_period=trailing_period)


def camera_detection_status(
    mode: str,
    device_index: int,
    backend_name: str,
    *,
    has_detections: bool,
    trailing_period: bool = False,
) -> str:
    if mode == CAMERA_MODE_BOARD:
        state_text = "已识别黑板数字" if has_detections else "未识别到黑板数字"
        return _with_trailing_period(
            f"{_camera_device_label(device_index, backend_name)} {state_text}",
            trailing_period=trailing_period,
        )
    if mode == CAMERA_MODE_HAND_COUNT:
        state_text = "已检测到手势" if has_detections else "未检测到手势"
        return _with_trailing_period(
            f"{_camera_device_label(device_index, backend_name)} {state_text}",
            trailing_period=trailing_period,
        )
    if has_detections:
        return camera_running_status(device_index, backend_name, trailing_period=trailing_period)
    return camera_waiting_status(mode, device_index, backend_name, trailing_period=trailing_period)


def camera_parallel_status(device_index: int, backend_name: str, *, trailing_period: bool = False) -> str:
    return _with_trailing_period(
        f"{_camera_device_label(device_index, backend_name)} 正在并行识别",
        trailing_period=trailing_period,
    )


def camera_recognizing_status(mode: str, device_index: int, backend_name: str, *, trailing_period: bool = False) -> str:
    if mode == CAMERA_MODE_BOARD:
        return _with_trailing_period(
            f"{_camera_device_label(device_index, backend_name)} 正在识别黑板",
            trailing_period=trailing_period,
        )
    if mode == CAMERA_MODE_HAND_COUNT:
        return _with_trailing_period(
            f"{_camera_device_label(device_index, backend_name)} 正在计数手势",
            trailing_period=trailing_period,
        )
    return camera_parallel_status(device_index, backend_name, trailing_period=trailing_period)


def camera_waiting_task_status(device_index: int, backend_name: str, *, trailing_period: bool = False) -> str:
    return _with_trailing_period(
        f"{_camera_device_label(device_index, backend_name)} 等待识别任务",
        trailing_period=trailing_period,
    )


def camera_opening_status(device_index: int) -> str:
    return f"正在打开摄像头 {device_index}..."


def camera_roi_label_text(width_ratio: float, height_ratio: float) -> str:
    width_percent = int(round(float(width_ratio) * 100))
    height_percent = int(round(float(height_ratio) * 100))
    return f"识别框大小：宽 {width_percent}% / 高 {height_percent}%"


def camera_hidden_low_confidence_summary(visible_count: int, hidden_count: int) -> str:
    return f"实时显示 {visible_count} 个数字，已隐藏 {hidden_count} 个低置信度结果。"


def camera_speed_label(mode: str) -> str:
    return "计数速度" if mode == CAMERA_MODE_HAND_COUNT else "OCR速度"


def camera_hand_count_summary(total_count: int) -> str:
    return f"总数：{total_count}"


def camera_too_many_hands_summary() -> str:
    return "请仅保留两只手在画面中"


__all__ = [
    "APP_READY_STATUS",
    "CAMERA_IDLE_STATUS",
    "CAMERA_IDLE_RUNTIME_STATUS",
    "CAMERA_MODE_BY_LABEL",
    "CAMERA_MODE_LABELS",
    "CAMERA_OPENING_PREVIEW_TEXT",
    "CAMERA_PREVIEW_PLACEHOLDER_TEXT",
    "CAMERA_ROI_APPLIED_STATUS",
    "CAMERA_START_FAILED_STATUS",
    "CAMERA_STOP_FAILED_STATUS",
    "CAMERA_STOPPING_STATUS",
    "CAMERA_TEXT_COPIED_STATUS",
    "CAMERA_WARNING_RUNTIME_STATUS",
    "RESULT_PLACEHOLDER_TEXT",
    "camera_detection_status",
    "camera_empty_summary",
    "camera_hand_count_summary",
    "camera_hidden_low_confidence_summary",
    "camera_mode_label",
    "camera_opening_status",
    "camera_parallel_status",
    "camera_prompt_text",
    "camera_recognizing_status",
    "camera_roi_label_text",
    "camera_running_status",
    "camera_speed_label",
    "camera_starting_status",
    "camera_started_status",
    "camera_too_many_hands_summary",
    "camera_waiting_status",
    "camera_waiting_task_status",
]
