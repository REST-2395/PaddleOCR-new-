"""Camera workflow controller for the staged GUI refactor."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from tkinter import messagebox, ttk

from camera.mode_profiles import get_camera_mode_profile
from camera.runtime import (
    CAMERA_EMPTY_RESET_FRAMES,
    CAMERA_MODE_BOARD,
    CAMERA_MODE_DIGIT,
    CAMERA_MODE_HAND_COUNT,
    CAMERA_PREVIEW_INTERVAL_MS,
    CameraDetection,
    CameraOCRRuntime,
    camera_detection_signature,
    camera_result_is_fresh,
    filter_camera_detections,
    group_camera_detections,
    overlay_camera_detections,
    stabilize_camera_detections,
    stable_camera_sequence,
    summarize_camera_detections,
)
from handcount.constants import HAND_CONNECTIONS
from handcount.overlay import overlay_hand_count_frame
from handcount.runtime import HandCountRuntime
from handcount.types import HandCountPayload
from core.recognition_service import DigitOCRService
from desktop.messages import (
    CAMERA_IDLE_STATUS,
    CAMERA_MODE_BY_LABEL,
    CAMERA_MODE_LABELS,
    CAMERA_OPENING_PREVIEW_TEXT,
    CAMERA_PREVIEW_PLACEHOLDER_TEXT,
    CAMERA_ROI_APPLIED_STATUS,
    CAMERA_START_FAILED_STATUS,
    CAMERA_STOP_FAILED_STATUS,
    CAMERA_STOPPING_STATUS,
    RESULT_PLACEHOLDER_TEXT,
    camera_empty_summary,
    camera_hand_count_summary,
    camera_hidden_low_confidence_summary,
    camera_opening_status,
    camera_prompt_text,
    camera_roi_label_text,
    camera_running_status,
    camera_speed_label,
    camera_started_status,
    camera_too_many_hands_summary,
)


class CameraController:
    """Own camera tab assembly and runtime interaction state."""

    def __init__(self, app: Any, recognition_controller: Any, result_panel_controller: Any) -> None:
        self.app = app
        self.recognition_controller = recognition_controller
        self.result_panel_controller = result_panel_controller

    def _build_camera_tab(self) -> None:
        self.app.camera_tab = ttk.Frame(self.app.notebook, padding=20)
        self.app.camera_tab.columnconfigure(0, weight=1)
        self.app.camera_tab.rowconfigure(2, weight=1)
        self.app.notebook.add(self.app.camera_tab, text="摄像头识别")
        _build_camera_controls(self)
        _build_camera_status_and_preview(self)

    def _apply_camera_roi_size(self) -> None:
        width_ratio = float(np.clip(self.app.camera_roi_width_var.get(), 0.20, 0.90))
        height_ratio = float(np.clip(self.app.camera_roi_height_var.get(), 0.15, 0.85))
        self.app.camera_roi_width_var.set(width_ratio)
        self.app.camera_roi_height_var.set(height_ratio)
        self.app.camera_roi_label_var.set(self._camera_roi_label_text())

        session = self.app.camera_session
        if session is not None and session.is_running:
            session.update_roi_size(width_ratio, height_ratio)
            self.app.current_camera_detections = ()
            self.app.current_camera_payload = None
            self.app.camera_tracks = []
            self.app.camera_result_signature = None
            self.app.camera_last_render_signature = None
            self.app.camera_last_result_completed_at = 0.0
            self.app.camera_last_consumed_result_id = -1
            self.app.camera_sequence_history.clear()
            prompt_mode = self._selected_camera_mode()
            self.app.summary_var.set(camera_prompt_text(prompt_mode))
            self._populate_camera_result_table((), mode=prompt_mode)

        self.recognition_controller._set_status(CAMERA_ROI_APPLIED_STATUS)

    def _handle_camera_started(self, runtime: Any, token: int) -> None:
        if token != self.app.camera_operation_token:
            self.recognition_controller._run_background_task(worker=runtime.stop)
            return

        self.app.camera_session = runtime
        self.app.camera_starting = False
        self.app.camera_stopping = False
        self._reset_camera_display_state()
        self.app.current_camera_result_mode = runtime.camera_mode
        self.app.camera_status_var.set(camera_running_status(runtime.device_index, runtime.backend_name))
        self._set_camera_controls_state(start_enabled=False, stop_enabled=True, device_enabled=False)
        self._populate_camera_result_table((), mode=runtime.camera_mode)
        self.app.summary_var.set(camera_prompt_text(runtime.camera_mode))
        self.recognition_controller._set_status(camera_started_status(runtime.device_index, runtime.camera_mode))
        self._schedule_camera_poll()

    def _update_board_camera_results(self, detections: tuple[CameraDetection, ...]) -> None:
        signature = (
            "board",
            camera_detection_signature(self._camera_detection_text(), detections) if detections else None,
        )
        if signature == self.app.camera_result_signature:
            return

        self.app.camera_empty_streak = 0
        if detections:
            self.app.summary_var.set(summarize_camera_detections(detections, mode=CAMERA_MODE_BOARD))
        else:
            self.app.summary_var.set(camera_empty_summary(CAMERA_MODE_BOARD))
        self._populate_camera_result_table(detections)
        self.app.camera_result_signature = signature

    def _set_camera_controls_state(
        self,
        *,
        start_enabled: bool,
        stop_enabled: bool,
        device_enabled: bool,
    ) -> None:
        roi_enabled = device_enabled or stop_enabled
        if self.app.camera_start_button is not None:
            self.app.camera_start_button.configure(state="normal" if start_enabled else "disabled")
        if self.app.camera_stop_button is not None:
            self.app.camera_stop_button.configure(state="normal" if stop_enabled else "disabled")
        if self.app.camera_device_combo is not None:
            self.app.camera_device_combo.configure(state="readonly" if device_enabled else "disabled")
        if self.app.camera_mode_combo is not None:
            self.app.camera_mode_combo.configure(state="readonly" if device_enabled else "disabled")
        for widget in (self.app.camera_roi_width_scale, self.app.camera_roi_height_scale, self.app.camera_apply_roi_button):
            if widget is not None:
                widget.configure(state="normal" if roi_enabled else "disabled")

    def _start_camera_session(self) -> None:
        if self.app.busy:
            messagebox.showinfo("正在处理中", "请等待当前识别任务完成后再启动摄像头。")
            return

        if self.app.image_loading:
            messagebox.showinfo("图片加载中", "请等待图片加载完成后再启动摄像头。")
            return

        if self.app.camera_starting or self.app.camera_stopping:
            return

        if self.app.camera_session is not None and self.app.camera_session.is_running:
            return

        try:
            device_index = int(self.app.camera_device_var.get())
        except ValueError:
            messagebox.showerror("摄像头设备无效", "请选择有效的摄像头编号。")
            return

        self.app.camera_operation_token += 1
        current_token = self.app.camera_operation_token
        self.app.camera_starting = True
        self._reset_camera_display_state()
        self._set_camera_controls_state(start_enabled=False, stop_enabled=False, device_enabled=False)
        if self.app.camera_preview_label is not None:
            self.app.camera_preview_label.configure(image="", text=CAMERA_OPENING_PREVIEW_TEXT)
        self.app.camera_preview_photo = None
        self.app.camera_status_var.set(camera_opening_status(device_index))
        self.recognition_controller._set_status(camera_opening_status(device_index))

        runtime = self._build_camera_session(device_index)

        self.recognition_controller._run_background_task(
            worker=lambda: self._open_camera_session(runtime, device_index),
            on_success=lambda active_runtime: self._handle_camera_started(active_runtime, current_token),
            on_error=lambda error: self._handle_camera_start_error(error, current_token),
        )

    def _stop_camera_session(self, *, reset_preview: bool = True) -> None:
        if self.app.camera_poll_after_id is not None:
            try:
                self.app.after_cancel(self.app.camera_poll_after_id)
            except Exception:
                pass
            self.app.camera_poll_after_id = None

        runtime = self.app.camera_session
        self.app.camera_session = None
        self.app.camera_operation_token += 1
        current_token = self.app.camera_operation_token
        was_running = runtime is not None

        self.app.camera_starting = False
        self.app.camera_stopping = was_running
        self._reset_camera_display_state()

        if reset_preview and self.app.camera_preview_label is not None:
            self.app.camera_preview_label.configure(image="", text=CAMERA_PREVIEW_PLACEHOLDER_TEXT)
            self.app.camera_preview_photo = None

        self.result_panel_controller._restore_result_panel_from_current_output()
        if was_running:
            self.app.camera_status_var.set(CAMERA_STOPPING_STATUS)
            self._set_camera_controls_state(start_enabled=False, stop_enabled=False, device_enabled=False)
            self.recognition_controller._run_background_task(
                worker=runtime.stop,
                on_success=lambda _: self._handle_camera_stopped(current_token),
                on_error=lambda error: self._handle_camera_stop_error(error, current_token),
            )
            return

        self.app.camera_status_var.set(CAMERA_IDLE_STATUS)
        self._set_camera_controls_state(start_enabled=True, stop_enabled=False, device_enabled=True)

    def _schedule_camera_poll(self) -> None:
        if self.app.camera_poll_after_id is None:
            self.app.camera_poll_after_id = self.app.after(CAMERA_PREVIEW_INTERVAL_MS, self._poll_camera_snapshot)

    def _poll_camera_snapshot(self) -> None:
        self.app.camera_poll_after_id = None
        if self.app.camera_session is None:
            return

        snapshot = self.app.camera_session.get_snapshot(last_frame_id=self.app.camera_last_snapshot_frame_id)
        self.app.camera_status_var.set(
            f"{snapshot.status_text}  帧率 {snapshot.capture_fps:.1f}  "
            f"{camera_speed_label(self.app.current_camera_result_mode)} {snapshot.ocr_fps:.1f}"
        )

        if snapshot.error_message is not None and snapshot.error_message != self.app.camera_error_message:
            self.app.camera_error_message = snapshot.error_message
            self._stop_camera_session(reset_preview=True)
            messagebox.showerror("摄像头错误", snapshot.error_message)
            return

        preview_frame = snapshot.frame_bgr if snapshot.has_new_frame else None
        preview_frame_id = snapshot.frame_id
        latest_result = snapshot.latest_result

        if preview_frame is not None:
            self.app.current_camera_preview = preview_frame
            self.app.camera_last_snapshot_frame_id = preview_frame_id

        if latest_result is not None and latest_result.frame_id != self.app.camera_last_consumed_result_id:
            _consume_camera_snapshot_result(self, latest_result)

        frame_image = preview_frame if preview_frame is not None else self.app.current_camera_preview
        _render_camera_snapshot_preview(self, snapshot, preview_frame_id, frame_image)

        if self.app.camera_session is not None and self.app.camera_session.is_running:
            self._schedule_camera_poll()
        elif self.app.camera_session is not None:
            self._stop_camera_session(reset_preview=False)

    def _update_camera_results(
        self,
        detections: tuple[CameraDetection, ...],
        *,
        filter_state: dict[str, float | int | str] | None,
    ) -> None:
        if filter_state is not None and filter_state["reason"] in {"average_low", "all_low"}:
            self.app.camera_empty_streak = 0
            self.app.camera_sequence_history.clear()
            signature = (
                "filtered",
                filter_state["reason"],
                round(float(filter_state["average_confidence"]), 2),
                int(filter_state["total_count"]),
            )
            if signature == self.app.camera_result_signature:
                return

            self.app.summary_var.set(camera_empty_summary(CAMERA_MODE_DIGIT))
            self._populate_camera_result_table(())
            self.app.camera_result_signature = signature
            return

        if not detections:
            self.app.camera_empty_streak += 1
            if (
                self.app.camera_empty_streak >= CAMERA_EMPTY_RESET_FRAMES
                and self.app.camera_result_signature != "empty"
            ):
                self.app.summary_var.set(camera_empty_summary(CAMERA_MODE_DIGIT))
                self._populate_camera_result_table(())
                self.app.camera_result_signature = "empty"
            return

        self.app.camera_empty_streak = 0
        current_sequence = "".join(item.text for item in detections)
        self.app.camera_sequence_history.append(current_sequence)
        stable_sequence = stable_camera_sequence(self.app.camera_sequence_history, current_sequence)
        hidden_count = int(filter_state["hidden_count"]) if filter_state is not None else 0
        signature = (
            camera_detection_signature(stable_sequence, detections),
            hidden_count,
        )
        if signature == self.app.camera_result_signature:
            return

        if hidden_count:
            self.app.summary_var.set(camera_hidden_low_confidence_summary(len(detections), hidden_count))
        else:
            self.app.summary_var.set(summarize_camera_detections(detections))
        self._populate_camera_result_table(detections)
        self.app.camera_result_signature = signature

    def _update_hand_count_results(self, payload: HandCountPayload | None) -> None:
        signature = _hand_payload_signature(payload)
        if signature == self.app.camera_result_signature:
            return

        if payload is None or (not payload.items and not payload.too_many_hands):
            self.app.summary_var.set(camera_empty_summary(CAMERA_MODE_HAND_COUNT))
        elif payload.too_many_hands:
            self.app.summary_var.set(camera_too_many_hands_summary())
        else:
            self.app.summary_var.set(camera_hand_count_summary(payload.total_count))
        self._populate_hand_count_result_table(payload)
        self.app.camera_result_signature = signature

    def _populate_camera_result_table(
        self,
        detections: tuple[CameraDetection, ...],
        *,
        mode: str | None = None,
    ) -> None:
        active_mode = self._selected_camera_mode() if mode is None else mode
        if active_mode == CAMERA_MODE_HAND_COUNT:
            self._populate_hand_count_result_table(self._current_hand_payload())
            return

        for item_id in self.app.result_table.get_children():
            self.app.result_table.delete(item_id)

        ordered_items = tuple(detections)
        for item in ordered_items:
            x0, y0, x1, y1 = item.box
            self.app.result_table.insert(
                "",
                "end",
                values=(item.text, f"{item.score:.2f}", f"({x0},{y0}) ({x1},{y1})"),
            )

        if not ordered_items:
            self.app.result_table.insert("", "end", values=("-", "-", "无实时数字"))

    def _populate_hand_count_result_table(self, payload: HandCountPayload | None) -> None:
        for item_id in self.app.result_table.get_children():
            self.app.result_table.delete(item_id)

        if payload is not None and not payload.too_many_hands and payload.items:
            for item in payload.items:
                x0, y0, x1, y1 = item.box
                self.app.result_table.insert(
                    "",
                    "end",
                    values=(f"{item.handedness}: {item.count}", f"{item.score:.2f}", f"({x0},{y0}) ({x1},{y1})"),
                )
            return

        placeholder = "未检测到手"
        if payload is not None and payload.too_many_hands:
            placeholder = camera_too_many_hands_summary()
        self.app.result_table.insert("", "end", values=("-", "-", placeholder))

    def _camera_roi_label_text(self) -> str:
        return camera_roi_label_text(self.app.camera_roi_width_var.get(), self.app.camera_roi_height_var.get())

    def _selected_camera_mode(self) -> str:
        camera_mode_var = getattr(self.app, "camera_mode_var", None)
        if camera_mode_var is None:
            return CAMERA_MODE_DIGIT
        return CAMERA_MODE_BY_LABEL.get(camera_mode_var.get(), CAMERA_MODE_DIGIT)

    def _camera_mode_is_board(self) -> bool:
        return self._selected_camera_mode() == CAMERA_MODE_BOARD

    def _camera_mode_is_hand_count(self) -> bool:
        return self._selected_camera_mode() == CAMERA_MODE_HAND_COUNT

    def _handle_camera_mode_changed(self) -> None:
        self.app.camera_roi_label_var.set(self._camera_roi_label_text())
        if self.app.camera_session is None or not self.app.camera_session.is_running:
            self._set_camera_controls_state(start_enabled=True, stop_enabled=False, device_enabled=True)
        if self.app.current_output is None:
            self.app.summary_var.set(RESULT_PLACEHOLDER_TEXT)
            self._populate_camera_result_table((), mode=self._selected_camera_mode())

    def _reset_camera_display_state(self) -> None:
        self.app.current_camera_preview = None
        self.app.current_camera_frame = None
        self.app.current_camera_detections = ()
        self.app.current_camera_payload = None
        self.app.current_camera_result_mode = CAMERA_MODE_DIGIT
        self.app.camera_tracks = []
        self.app.camera_next_track_id = 1
        self.app.camera_empty_streak = 0
        self.app.camera_filter_state = None
        self.app.camera_result_signature = None
        self.app.camera_last_render_signature = None
        self.app.camera_last_snapshot_frame_id = -1
        self.app.camera_last_result_completed_at = 0.0
        self.app.camera_last_consumed_result_id = -1
        self.app.camera_sequence_history.clear()
        self.app.camera_error_message = None

    def _handle_camera_roi_slider_changed(self) -> None:
        self.app.camera_roi_label_var.set(self._camera_roi_label_text())

    def _open_camera_session(self, runtime: Any, device_index: int) -> Any:
        runtime.start(device_index=device_index)
        return runtime

    def _build_camera_session(self, device_index: int) -> Any:
        if self._camera_mode_is_hand_count():
            return self._build_hand_count_session(device_index)

        selected_mode = self._selected_camera_mode()
        profile = get_camera_mode_profile(selected_mode)
        return CameraOCRRuntime(
            service_factory=self._build_local_camera_service_factory(selected_mode),
            camera_mode=selected_mode,
            device_index=device_index,
            capture_size=profile.capture_size,
            ocr_interval_seconds=profile.ocr_interval_seconds,
            max_ocr_side=profile.max_ocr_side,
            roi_width_ratio=float(self.app.camera_roi_width_var.get()),
            roi_height_ratio=float(self.app.camera_roi_height_var.get()),
        )

    def _build_hand_count_session(self, device_index: int) -> HandCountRuntime:
        return HandCountRuntime(
            device_index=device_index,
            roi_width_ratio=float(self.app.camera_roi_width_var.get()),
            roi_height_ratio=float(self.app.camera_roi_height_var.get()),
        )

    def _build_local_camera_service_factory(self, selected_mode: str) -> Any:
        profile = get_camera_mode_profile(selected_mode)
        service_cache: dict[str, DigitOCRService] = {}

        def get_service() -> DigitOCRService:
            service = service_cache.get("service")
            if service is None:
                service = DigitOCRService(
                    dict_path=str(self.app.dict_path),
                    ocr_version="PP-OCRv5",
                    score_threshold=0.3,
                    use_gpu=False,
                    cpu_threads=profile.cpu_threads,
                    enable_mkldnn=profile.enable_mkldnn,
                    use_textline_orientation=profile.use_textline_orientation,
                    language="en",
                )
                service_cache["service"] = service
            return service

        return get_service

    def _handle_camera_start_error(self, error: Exception, token: int) -> None:
        if token != self.app.camera_operation_token:
            return

        self.app.camera_starting = False
        self.app.camera_stopping = False
        self.app.camera_session = None
        self.app.camera_status_var.set(CAMERA_START_FAILED_STATUS)
        self._set_camera_controls_state(start_enabled=True, stop_enabled=False, device_enabled=True)
        if self.app.camera_preview_label is not None:
            self.app.camera_preview_label.configure(image="", text=CAMERA_PREVIEW_PLACEHOLDER_TEXT)
        self.app.camera_preview_photo = None
        self.recognition_controller._set_status(CAMERA_START_FAILED_STATUS)
        messagebox.showerror("摄像头启动失败", str(error))

    def _handle_camera_stopped(self, token: int) -> None:
        if token != self.app.camera_operation_token:
            return

        self.app.camera_stopping = False
        self.app.camera_status_var.set(CAMERA_IDLE_STATUS)
        self._set_camera_controls_state(start_enabled=True, stop_enabled=False, device_enabled=True)

    def _handle_camera_stop_error(self, error: Exception, token: int) -> None:
        if token != self.app.camera_operation_token:
            return

        self.app.camera_stopping = False
        self.app.camera_status_var.set(CAMERA_STOP_FAILED_STATUS)
        self._set_camera_controls_state(start_enabled=True, stop_enabled=False, device_enabled=True)
        self.recognition_controller._set_status(f"摄像头停止时发生错误：{error}")

    def _preview_detections(self) -> tuple[CameraDetection, ...]:
        if not camera_result_is_fresh(self.app.camera_last_result_completed_at, now=time.perf_counter()):
            return ()
        return self.app.current_camera_detections

    def _current_hand_payload(self) -> HandCountPayload | None:
        payload = getattr(self.app, "current_camera_payload", None)
        return payload if isinstance(payload, HandCountPayload) else None

    def _preview_hand_payload(self) -> HandCountPayload | None:
        if not camera_result_is_fresh(self.app.camera_last_result_completed_at, now=time.perf_counter()):
            return None
        return self._current_hand_payload()

    def _camera_detection_text(self) -> str:
        if self.app.current_camera_result_mode == CAMERA_MODE_HAND_COUNT:
            return self._hand_count_text()
        if not self.app.current_camera_detections:
            return ""
        rows = group_camera_detections(self.app.current_camera_detections)
        if self.app.current_camera_result_mode == CAMERA_MODE_BOARD:
            return "\n".join(" ".join(item.text for item in row) for row in rows)
        return " ".join(item.text for row in rows for item in row)

    def _hand_count_text(self) -> str:
        payload = self._current_hand_payload()
        if payload is None:
            return ""
        if payload.too_many_hands:
            return camera_too_many_hands_summary()
        if not payload.items:
            return camera_empty_summary(CAMERA_MODE_HAND_COUNT)
        return camera_hand_count_summary(payload.total_count)

    def _is_camera_active(self) -> bool:
        return self.app.camera_session is not None and self.app.camera_session.is_running


__all__ = ["CAMERA_MODE_BOARD", "CAMERA_MODE_DIGIT", "CAMERA_MODE_HAND_COUNT", "CameraController"]


def _build_camera_selector_row(controller: CameraController, controls: ttk.Frame) -> None:
    app = controller.app
    ttk.Label(controls, text="设备编号").grid(row=0, column=0, sticky="w", padx=(0, 10))
    app.camera_device_combo = ttk.Combobox(
        controls,
        textvariable=app.camera_device_var,
        values=("0", "1", "2", "3"),
        width=8,
        state="readonly",
    )
    app.camera_device_combo.grid(row=0, column=1, sticky="w")

    ttk.Label(controls, text="识别模式").grid(row=0, column=2, sticky="w", padx=(20, 10))
    app.camera_mode_combo = ttk.Combobox(
        controls,
        textvariable=app.camera_mode_var,
        values=tuple(CAMERA_MODE_LABELS.values()),
        width=12,
        state="readonly",
    )
    app.camera_mode_combo.grid(row=0, column=3, sticky="w")
    app.camera_mode_combo.bind("<<ComboboxSelected>>", lambda _event: controller._handle_camera_mode_changed())

    app.camera_start_button = ttk.Button(
        controls,
        text="启动摄像头",
        command=controller._start_camera_session,
        style="Accent.TButton",
    )
    app.camera_start_button.grid(row=0, column=4, sticky="w", padx=(20, 10))

    app.camera_stop_button = ttk.Button(
        controls,
        text="停止摄像头",
        command=controller._stop_camera_session,
        state="disabled",
    )
    app.camera_stop_button.grid(row=0, column=5, sticky="w")


def _build_camera_roi_row(controller: CameraController, controls: ttk.Frame) -> None:
    app = controller.app
    ttk.Label(controls, text="框宽").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(14, 0))
    app.camera_roi_width_scale = ttk.Scale(
        controls,
        from_=0.20,
        to=0.90,
        variable=app.camera_roi_width_var,
        orient="horizontal",
        command=lambda _value: controller._handle_camera_roi_slider_changed(),
    )
    app.camera_roi_width_scale.grid(row=1, column=1, sticky="ew", pady=(14, 0), padx=(0, 20))

    ttk.Label(controls, text="框高").grid(row=1, column=2, sticky="w", padx=(0, 10), pady=(14, 0))
    app.camera_roi_height_scale = ttk.Scale(
        controls,
        from_=0.15,
        to=0.85,
        variable=app.camera_roi_height_var,
        orient="horizontal",
        command=lambda _value: controller._handle_camera_roi_slider_changed(),
    )
    app.camera_roi_height_scale.grid(row=1, column=3, sticky="ew", pady=(14, 0), padx=(0, 20))

    app.camera_apply_roi_button = ttk.Button(
        controls,
        text="应用识别框",
        command=controller._apply_camera_roi_size,
    )
    app.camera_apply_roi_button.grid(row=1, column=4, sticky="w", pady=(14, 0))

    ttk.Label(
        controls,
        textvariable=app.camera_roi_label_var,
        foreground="#6b7280",
    ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(12, 0))


def _build_camera_controls(controller: CameraController) -> ttk.Frame:
    controls = ttk.Frame(controller.app.camera_tab)
    controls.grid(row=0, column=0, sticky="ew", pady=(0, 20))
    controls.columnconfigure(1, weight=1)
    controls.columnconfigure(3, weight=1)
    _build_camera_selector_row(controller, controls)
    _build_camera_roi_row(controller, controls)
    return controls


def _build_camera_status_and_preview(controller: CameraController) -> None:
    ttk.Label(
        controller.app.camera_tab,
        textvariable=controller.app.camera_status_var,
        wraplength=620,
        foreground="#6b7280",
    ).grid(row=1, column=0, sticky="w", pady=(0, 10))

    controller.app.camera_preview_label = ttk.Label(
        controller.app.camera_tab,
        text="此处将显示摄像头实时预览。",
        anchor="center",
        relief="flat",
        background="#e5e7eb",
        foreground="#6b7280",
    )
    controller.app.camera_preview_label.grid(row=2, column=0, sticky="nsew")


def _consume_camera_snapshot_result(controller: CameraController, latest_result: Any) -> None:
    controller.app.current_camera_result_mode = latest_result.mode
    if latest_result.mode == CAMERA_MODE_HAND_COUNT:
        controller.app.camera_filter_state = None
        controller.app.camera_tracks = []
        controller.app.current_camera_detections = ()
        controller.app.current_camera_payload = (
            latest_result.payload if isinstance(latest_result.payload, HandCountPayload) else None
        )
        controller.app.camera_last_result_completed_at = latest_result.completed_at
        controller.app.camera_last_consumed_result_id = latest_result.frame_id
        controller._update_hand_count_results(controller._current_hand_payload())
        if latest_result.warnings:
            controller.recognition_controller._set_status("; ".join(latest_result.warnings))
        return

    if latest_result.mode == CAMERA_MODE_BOARD:
        controller.app.camera_filter_state = None
        controller.app.camera_tracks = []
        controller.app.current_camera_detections = latest_result.detections
        controller.app.current_camera_payload = None
        controller.app.camera_last_result_completed_at = latest_result.completed_at
        controller.app.camera_last_consumed_result_id = latest_result.frame_id
        controller._update_board_camera_results(latest_result.detections)
        if latest_result.warnings:
            controller.recognition_controller._set_status("; ".join(latest_result.warnings))
        return

    controller.app.current_camera_payload = None
    filtered_detections, filter_state = filter_camera_detections(latest_result.detections)
    controller.app.camera_filter_state = filter_state
    if filtered_detections:
        detections, controller.app.camera_tracks, controller.app.camera_next_track_id = stabilize_camera_detections(
            filtered_detections,
            controller.app.camera_tracks,
            next_track_id=controller.app.camera_next_track_id,
            allow_missed_tracks=False,
        )
    else:
        controller.app.camera_tracks = []
        detections = ()

    if detections:
        controller.app.current_camera_detections = detections
        controller.app.camera_last_result_completed_at = latest_result.completed_at
    controller.app.camera_last_consumed_result_id = latest_result.frame_id
    controller._update_camera_results(detections, filter_state=filter_state)


def _camera_roi_ratios(controller: CameraController) -> tuple[float, float]:
    session = controller.app.camera_session
    width_ratio = session.roi_width_ratio if session is not None else float(controller.app.camera_roi_width_var.get())
    height_ratio = session.roi_height_ratio if session is not None else float(controller.app.camera_roi_height_var.get())
    return width_ratio, height_ratio


def _render_camera_snapshot_preview(
    controller: CameraController,
    snapshot: Any,
    preview_frame_id: int,
    frame_image: np.ndarray | None,
) -> None:
    overlay_detections = controller._preview_detections()
    hand_payload = controller._preview_hand_payload()
    render_signature = (
        preview_frame_id,
        controller.app.current_camera_result_mode,
        _hand_payload_signature(hand_payload)
        if controller.app.current_camera_result_mode == CAMERA_MODE_HAND_COUNT
        else (camera_detection_signature("preview", overlay_detections) if overlay_detections else None),
    )
    if (
        frame_image is None
        or render_signature == controller.app.camera_last_render_signature
        or controller.app.camera_preview_label is None
    ):
        return

    roi_width_ratio, roi_height_ratio = _camera_roi_ratios(controller)
    if controller.app.current_camera_result_mode == CAMERA_MODE_HAND_COUNT:
        overlay_frame = overlay_hand_count_frame(
            frame_image,
            hand_payload,
            capture_fps=snapshot.capture_fps,
            count_fps=snapshot.ocr_fps,
            prompt_text=camera_prompt_text(controller.app.current_camera_result_mode, trailing_period=False),
            roi_width_ratio=roi_width_ratio,
            roi_height_ratio=roi_height_ratio,
            connections=HAND_CONNECTIONS,
        )
    else:
        overlay_frame = overlay_camera_detections(
            frame_image,
            overlay_detections,
            capture_fps=snapshot.capture_fps,
            ocr_fps=snapshot.ocr_fps,
            prompt_text=camera_prompt_text(controller.app.current_camera_result_mode, trailing_period=False),
            camera_mode=controller.app.current_camera_result_mode,
            roi_width_ratio=roi_width_ratio,
            roi_height_ratio=roi_height_ratio,
        )
    controller.app.current_camera_frame = overlay_frame
    controller.result_panel_controller._show_bgr_image(
        controller.app.camera_preview_label,
        overlay_frame,
        max_size=None,
        photo_slot="camera_preview_photo",
    )
    controller.app.camera_last_render_signature = render_signature


def _hand_payload_signature(payload: HandCountPayload | None) -> tuple[object, ...] | None:
    if payload is None:
        return None
    item_signature = tuple(
        (item.handedness, item.count, round(item.score, 3), item.box)
        for item in payload.items
    )
    return (
        payload.total_count,
        payload.too_many_hands,
        round(payload.fps, 2),
        payload.warnings,
        item_signature,
    )
