"""Recognition workflow controller for the staged GUI refactor."""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

import numpy as np
from tkinter import messagebox

from core.messages import coalesce_result_text, format_recognition_status
from core.recognition_service import DigitOCRService, RecognitionOutput
from desktop.media import prepare_result_preview_image


RecognitionWorker = Callable[[DigitOCRService], RecognitionOutput]


class RecognitionController:
    """Coordinate OCR service lifecycle, status updates, and recognition jobs."""

    def __init__(self, app: Any, result_panel_controller: Any) -> None:
        self.app = app
        self.result_panel_controller = result_panel_controller

    def _submit_recognition(
        self,
        *,
        task_name: str,
        worker: RecognitionWorker,
    ) -> None:
        if self.app.busy:
            messagebox.showinfo("正在处理中", "识别任务正在运行，请稍候。")
            return

        if self.app.image_loading:
            messagebox.showinfo("图片加载中", "请等待图片加载完成后再执行识别。")
            return

        if self.app.camera_starting or self.app.camera_stopping:
            messagebox.showinfo("摄像头处理中", "请等待摄像头启动或停止完成后再执行识别。")
            return

        if self.app.camera_session is not None and self.app.camera_session.is_running:
            messagebox.showinfo("摄像头运行中", "请先停止摄像头识别，再执行手写或图片识别。")
            return

        self.app.busy = True
        self._toggle_actions(enabled=False)
        self._set_status(task_name)

        def run() -> None:
            try:
                service = self._get_or_create_service()
                output = worker(service)
                preview_image = prepare_result_preview_image(output.annotated_image)
            except Exception as exc:  # pragma: no cover - UI error reporting
                error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                self.app.after(0, lambda: self._handle_recognition_error(error_message))
                return

            self.app.after(0, lambda: self._handle_recognition_success(output, preview_image))

        threading.Thread(target=run, daemon=True).start()

    def _handle_recognition_success(self, output: RecognitionOutput, preview_image: np.ndarray) -> None:
        self.app.current_output = output
        self.app.summary_var.set(coalesce_result_text(output.combined_text))
        self.result_panel_controller._populate_result_table(output)
        self._toggle_actions(enabled=True)
        self.app.busy = False

        self._set_status(
            format_recognition_status(
                output.source_name,
                output.summary_text,
                output.warnings,
                has_results=bool(output.results),
            )
        )
        self.result_panel_controller._schedule_result_preview(preview_image)

    def _handle_recognition_error(self, error_message: str) -> None:
        self._toggle_actions(enabled=True)
        self.app.busy = False
        self._set_status("识别失败。")
        messagebox.showerror("OCR 错误", error_message)

    def _get_or_create_service(self) -> DigitOCRService:
        if self.app.service is not None:
            return self.app.service

        with self.app.worker_lock:
            if self.app.service is None:
                self.app.after(0, lambda: self._set_status("正在加载 OCR 引擎和本地模型..."))
                self.app.service = DigitOCRService(
                    dict_path=self.app.dict_path,
                    ocr_version="PP-OCRv5",
                    score_threshold=0.3,
                    use_gpu=False,
                )
        return self.app.service

    def _toggle_actions(self, *, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (
            getattr(self.app, "clear_canvas_button", None),
            getattr(self.app, "recognize_canvas_button", None),
            getattr(self.app, "choose_image_button", None),
            getattr(self.app, "clear_image_button", None),
            getattr(self.app, "recognize_image_button", None),
            getattr(self.app, "save_result_button", None),
            getattr(self.app, "copy_result_button", None),
        ):
            if widget is not None:
                widget.configure(state=state)

    def _set_status(self, message: str) -> None:
        self.app.status_var.set(message)

    def _queue_status_update(self, message: str) -> None:
        self.app.after(0, lambda value=message: self._set_status(value))

    def _run_background_task(
        self,
        *,
        worker,
        on_success=None,
        on_error=None,
    ) -> None:
        def run() -> None:
            try:
                result = worker()
            except Exception as exc:  # pragma: no cover - UI error reporting
                if on_error is not None and self.app.winfo_exists():
                    self.app.after(0, lambda error=exc: on_error(error))
                return

            if on_success is not None and self.app.winfo_exists():
                self.app.after(0, lambda value=result: on_success(value))

        threading.Thread(target=run, daemon=True).start()

    def _start_service_warmup(self) -> None:
        if self.app.service_warmup_started or self.app.service is not None:
            return

        self.app.service_warmup_started = True
        self._set_status("正在后台预热 OCR 引擎...")
        self._run_background_task(
            worker=self._get_or_create_service,
            on_success=lambda _: self._handle_service_warmup_success(),
            on_error=lambda error: self._handle_service_warmup_error(error),
        )

    def _handle_service_warmup_success(self) -> None:
        self.app.service_warmup_finished = True
        if not self.app.busy and not self.app.image_loading and not self.app.camera_starting:
            self._set_status("已完成 OCR 预热。")

    def _handle_service_warmup_error(self, error: Exception) -> None:
        self.app.service_warmup_started = False
        if not self.app.busy and not self.app.image_loading and not self.app.camera_starting:
            self._set_status(f"OCR 预热失败：{error}")


__all__ = ["RecognitionController"]
