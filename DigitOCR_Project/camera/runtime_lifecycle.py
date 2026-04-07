"""Lifecycle helpers for the camera runtime facade."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from . import config
from .runtime_support import drain_task_queue, open_camera_capture
from .state import CameraSnapshot
from desktop.messages import CAMERA_IDLE_RUNTIME_STATUS, camera_starting_status


class RuntimeLifecycleMixin:
    """Manage capture lifecycle and lightweight runtime state."""

    def update_roi_size(self, width_ratio: float, height_ratio: float) -> None:
        self._roi_width_ratio = float(np.clip(width_ratio, 0.20, 0.90))
        self._roi_height_ratio = float(np.clip(height_ratio, 0.15, 0.85))
        self._roi_generation += 1
        self._fast_pending_frames = {}
        self._fast_last_submitted_frame_id = -1
        drain_task_queue(self._ocr_task_queue)
        for task_queue in self._fast_task_queues:
            drain_task_queue(task_queue)
        self._ocr_task_inflight = False
        self._ocr_task_started_at = 0.0
        self._ocr_task_frame_id = -1
        self._fast_task_inflight = [False for _ in self._fast_task_inflight]

    def start(self, *, device_index: int | None = None) -> None:
        if self._running:
            self.stop()

        if device_index is not None:
            self._device_index = device_index

        capture, device_index, backend_name, open_detail = open_camera_capture(
            cv2,
            preferred_index=self._device_index,
            capture_size=self._capture_size,
            target_capture_fps=self._target_capture_fps,
            warmup_frames=config.CAMERA_WARMUP_FRAMES,
        )
        if capture is None or device_index is None:
            raise RuntimeError(f"Failed to open camera device {self._device_index}. {open_detail}")

        self.start_with_capture(capture, device_index=device_index, backend_name=backend_name)

    def start_with_capture(
        self,
        capture: cv2.VideoCapture,
        *,
        device_index: int | None = None,
        backend_name: str = "manual",
    ) -> None:
        if self._running:
            self.stop()

        if device_index is not None:
            self._device_index = device_index

        self._capture = capture
        self._backend_name = backend_name
        self._stop_event.clear()
        with self._lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._latest_result = None
            self._worker_error = None
            self._capture_fps = 0.0
            self._ocr_fps = 0.0
            self._status_text = camera_starting_status(self._device_index)
            self._inference_resume_at = time.perf_counter() + self._startup_grace_seconds
        self._last_fallback_completed_at = 0.0
        self._fast_pending_frames = {}
        self._fast_last_submitted_frame_id = -1
        self._roi_generation = 1
        try:
            self._start_ocr_worker()
        except Exception:
            if self._capture is not None:
                self._capture.release()
            self._capture = None
            raise
        self._running = True

        self._capture_thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._worker_thread = threading.Thread(target=self._inference_loop, name="camera-inference", daemon=True)
        self._capture_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        for thread in (self._capture_thread, self._worker_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(config.CAMERA_THREAD_JOIN_TIMEOUT_MS, 1) / 1000.0)

        if self._capture is not None:
            self._capture.release()
        self._stop_ocr_worker()

        self._capture = None
        self._capture_thread = None
        self._worker_thread = None
        self._running = False
        with self._lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._latest_result = None
            self._worker_error = None
            self._capture_fps = 0.0
            self._ocr_fps = 0.0
            self._status_text = CAMERA_IDLE_RUNTIME_STATUS
            self._inference_resume_at = 0.0
            self._last_fallback_completed_at = 0.0
            self._fast_pending_frames = {}
            self._fast_last_submitted_frame_id = -1
            self._roi_generation = 0

    def get_snapshot(self, last_frame_id: int = -1) -> CameraSnapshot:
        with self._lock:
            has_new_frame = self._latest_frame is not None and self._latest_frame_id != last_frame_id
            return CameraSnapshot(
                has_new_frame=has_new_frame,
                frame_bgr=self._latest_frame if has_new_frame else None,
                frame_id=self._latest_frame_id,
                latest_result=self._latest_result,
                running=self._running,
                device_index=self._device_index,
                backend_name=self._backend_name,
                capture_fps=self._capture_fps,
                ocr_fps=self._ocr_fps,
                status_text=self._status_text,
                error_message=self._worker_error,
            )

    def _camera_fallback_allowed(self, *, now: float | None = None) -> bool:
        current_time = time.perf_counter() if now is None else now
        interval_seconds = max(0.0, config.CAMERA_FALLBACK_INTERVAL_MS / 1000.0)
        return (current_time - self._last_fallback_completed_at) >= interval_seconds

    def _record_camera_fallback(self, *, completed_at: float) -> None:
        self._last_fallback_completed_at = completed_at

    def _fast_workers_available(self) -> bool:
        return bool(self._fast_task_inflight) and any(not inflight for inflight in self._fast_task_inflight)
