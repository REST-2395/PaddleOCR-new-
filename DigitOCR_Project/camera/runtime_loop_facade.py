"""Loop and delegation helpers for the camera runtime facade."""

from __future__ import annotations

import time

import numpy as np

from . import config
from .overlay import resize_camera_frame_for_preview
from .protocol import BoardFramePlan, CameraOCRTask
from desktop.messages import camera_running_status
from core.ocr_engine import OCRResult


class RuntimeLoopFacadeMixin:
    """Delegate loop-specific work out of the runtime facade class."""

    def _capture_loop(self) -> None:
        last_published_at = 0.0
        publish_interval = max(self._preview_interval_seconds, 1.0 / self._target_capture_fps)
        while not self._stop_event.is_set():
            capture = self._capture
            if capture is None:
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                self._set_worker_error(f"Failed to read frames from camera {self._device_index}.")
                time.sleep(self._retry_seconds)
                continue

            now = time.perf_counter()
            elapsed_since_publish = now - last_published_at if last_published_at > 0.0 else publish_interval
            if elapsed_since_publish < publish_interval:
                time.sleep(min(self._idle_seconds, publish_interval - elapsed_since_publish))
                continue

            preview_frame = resize_camera_frame_for_preview(frame, max_dimension=self._max_preview_size)
            published_fps = self._target_capture_fps if last_published_at <= 0.0 else (1.0 / max(1e-6, elapsed_since_publish))
            last_published_at = now
            with self._lock:
                self._latest_frame_id += 1
                self._latest_frame = preview_frame
                self._worker_error = None
                self._capture_fps = (
                    published_fps
                    if self._capture_fps <= 0.0
                    else ((self._capture_fps * 0.8) + (published_fps * 0.2))
                )
                self._status_text = camera_running_status(
                    self._device_index,
                    self._backend_name,
                    trailing_period=False,
                )

    def _inference_loop(self) -> None:
        if self._worker_config is not None:
            if self._camera_mode == config.CAMERA_MODE_BOARD:
                self._process_board_inference_loop()
                return
            self._process_inference_loop()
            return

        if self._camera_mode == config.CAMERA_MODE_BOARD:
            self._local_board_inference_loop()
            return
        self._local_inference_loop()

    def _prepare_board_frame_plan(self, frame_id: int, frame: np.ndarray) -> BoardFramePlan:
        from .board_loop import prepare_board_frame_plan

        return prepare_board_frame_plan(self, frame_id, frame)

    def _local_board_inference_loop(self) -> None:
        from .board_loop import local_board_inference_loop

        local_board_inference_loop(self)

    def _process_board_inference_loop(self) -> None:
        from .board_loop import process_board_inference_loop

        process_board_inference_loop(self)

    def _local_inference_loop(self) -> None:
        from .digit_loop import local_inference_loop

        local_inference_loop(self)

    def _process_inference_loop(self) -> None:
        from .digit_loop import process_inference_loop

        process_inference_loop(self)

    def _run_ocr(
        self,
        frame: np.ndarray,
        *,
        allow_fallback: bool = True,
    ) -> tuple[list[OCRResult], bool]:
        from .digit_loop import run_ocr

        return run_ocr(self, frame, allow_fallback=allow_fallback)

    def _run_board_ocr(
        self,
        task: CameraOCRTask,
        *,
        return_warnings: bool = False,
    ) -> list[OCRResult] | tuple[list[OCRResult], list[str]]:
        from .board_loop import run_board_ocr

        return run_board_ocr(self, task, return_warnings=return_warnings)

    def _set_worker_error(self, message: str) -> None:
        from .digit_loop import set_worker_error

        set_worker_error(self, message)
