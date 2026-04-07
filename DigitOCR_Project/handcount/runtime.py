"""Standalone runtime for the hand-count camera mode."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import replace
import threading
import time

import cv2
import numpy as np

from camera import config
from camera.overlay import resize_camera_frame_for_preview
from camera.roi import camera_roi_box
from camera.runtime_support import open_camera_capture
from camera.state import CameraInferenceResult, CameraSnapshot
from desktop.messages import camera_detection_status, camera_running_status, camera_waiting_status

from .constants import HAND_EMPTY_SUMMARY, HAND_TOO_MANY_WARNING
from .detector import HandDetector
from .types import HandCountItem, HandCountPayload


class HandCountRuntime:
    """Two-thread runtime that mirrors the camera OCR facade API."""

    def __init__(
        self,
        *,
        detector_factory: Callable[[], HandDetector] | None = None,
        device_index: int = config.CAMERA_INDEX,
        capture_size: tuple[int, int] = (config.CAMERA_FRAME_WIDTH, config.CAMERA_FRAME_HEIGHT),
        preview_interval_seconds: float = config.CAMERA_PREVIEW_INTERVAL_MS / 1000.0,
        inference_interval_seconds: float = config.CAMERA_HAND_INFERENCE_INTERVAL_MS / 1000.0,
        startup_grace_seconds: float = config.CAMERA_STARTUP_GRACE_MS / 1000.0,
        idle_seconds: float = config.CAMERA_WORKER_IDLE_MS / 1000.0,
        retry_seconds: float = config.CAMERA_CAPTURE_RETRY_MS / 1000.0,
        target_capture_fps: float = config.CAMERA_TARGET_FPS,
        max_preview_size: int = config.CAMERA_MAX_PREVIEW_SIZE,
        roi_width_ratio: float = config.CAMERA_ROI_WIDTH_RATIO,
        roi_height_ratio: float = config.CAMERA_ROI_HEIGHT_RATIO,
    ) -> None:
        self._detector_factory = detector_factory or (lambda: HandDetector())
        self._device_index = device_index
        self._capture_size = capture_size
        self._preview_interval_seconds = max(preview_interval_seconds, 0.01)
        self._inference_interval_seconds = max(inference_interval_seconds, 0.01)
        self._startup_grace_seconds = max(startup_grace_seconds, 0.0)
        self._idle_seconds = max(idle_seconds, 0.001)
        self._retry_seconds = max(retry_seconds, 0.001)
        self._target_capture_fps = max(target_capture_fps, 1.0)
        self._max_preview_size = max_preview_size
        self._roi_width_ratio = float(np.clip(roi_width_ratio, 0.20, 0.90))
        self._roi_height_ratio = float(np.clip(roi_height_ratio, 0.15, 0.85))

        self._capture: cv2.VideoCapture | None = None
        self._detector: HandDetector | None = None
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: np.ndarray | None = None
        self._latest_frame_id = 0
        self._latest_result: CameraInferenceResult | None = None
        self._worker_error: str | None = None
        self._capture_fps = 0.0
        self._ocr_fps = 0.0
        self._backend_name = ""
        self._status_text = HAND_EMPTY_SUMMARY
        self._inference_resume_at = 0.0
        self._running = False

        self._history: deque[HandCountPayload] = deque(maxlen=config.CAMERA_HAND_STABLE_HISTORY)
        self._stable_payload: HandCountPayload | None = None
        self._empty_frame_streak = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def camera_mode(self) -> str:
        return config.CAMERA_MODE_HAND_COUNT

    @property
    def roi_width_ratio(self) -> float:
        return self._roi_width_ratio

    @property
    def roi_height_ratio(self) -> float:
        return self._roi_height_ratio

    def update_roi_size(self, width_ratio: float, height_ratio: float) -> None:
        self._roi_width_ratio = float(np.clip(width_ratio, 0.20, 0.90))
        self._roi_height_ratio = float(np.clip(height_ratio, 0.15, 0.85))
        self._history.clear()
        self._stable_payload = None
        self._empty_frame_streak = 0
        with self._lock:
            self._latest_result = None
            self._ocr_fps = 0.0

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
        self._history.clear()
        self._stable_payload = None
        self._empty_frame_streak = 0
        with self._lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._latest_result = None
            self._worker_error = None
            self._capture_fps = 0.0
            self._ocr_fps = 0.0
            self._status_text = camera_running_status(
                self._device_index,
                self._backend_name,
                trailing_period=False,
            )
            self._inference_resume_at = time.perf_counter() + self._startup_grace_seconds

        try:
            self._detector = self._detector_factory()
        except Exception:
            capture.release()
            self._capture = None
            raise

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, name="handcount-capture", daemon=True)
        self._worker_thread = threading.Thread(target=self._inference_loop, name="handcount-inference", daemon=True)
        self._capture_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        for thread in (self._capture_thread, self._worker_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=max(config.CAMERA_THREAD_JOIN_TIMEOUT_MS, 1) / 1000.0)

        if self._capture is not None:
            self._capture.release()
        if self._detector is not None:
            self._detector.close()

        self._capture = None
        self._detector = None
        self._capture_thread = None
        self._worker_thread = None
        self._history.clear()
        self._stable_payload = None
        self._empty_frame_streak = 0
        self._running = False
        with self._lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._latest_result = None
            self._worker_error = None
            self._capture_fps = 0.0
            self._ocr_fps = 0.0
            self._status_text = HAND_EMPTY_SUMMARY
            self._inference_resume_at = 0.0

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

            mirrored_frame = cv2.flip(frame, 1)
            preview_frame = resize_camera_frame_for_preview(
                mirrored_frame,
                max_dimension=self._max_preview_size,
            )
            published_fps = (
                self._target_capture_fps
                if last_published_at <= 0.0
                else (1.0 / max(1e-6, elapsed_since_publish))
            )
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
                if not self._status_text:
                    self._status_text = camera_running_status(
                        self._device_index,
                        self._backend_name,
                        trailing_period=False,
                    )

    def _inference_loop(self) -> None:
        last_processed_frame_id = -1
        last_completed_at = 0.0

        while not self._stop_event.is_set():
            with self._lock:
                frame_id = self._latest_frame_id
                source_frame = None if self._latest_frame is None else self._latest_frame.copy()
                inference_resume_at = self._inference_resume_at

            if source_frame is None or frame_id == last_processed_frame_id:
                time.sleep(self._idle_seconds)
                continue

            now = time.perf_counter()
            if now < inference_resume_at:
                time.sleep(min(self._idle_seconds, inference_resume_at - now))
                continue

            if last_completed_at > 0.0:
                elapsed = now - last_completed_at
                if elapsed < self._inference_interval_seconds:
                    time.sleep(min(self._idle_seconds, self._inference_interval_seconds - elapsed))
                    continue

            detector = self._detector
            if detector is None:
                self._set_worker_error("Hand detector is unavailable.")
                time.sleep(self._idle_seconds)
                continue

            detection_started_at = time.perf_counter()
            try:
                raw_items = detector.detect(source_frame)
                completed_at = time.perf_counter()
            except Exception as exc:
                self._set_worker_error(f"Hand count failed: {exc}")
                last_completed_at = time.perf_counter()
                time.sleep(self._idle_seconds)
                continue

            current_count_fps = 1.0 / max(1e-6, completed_at - detection_started_at)
            payload, status_text = self._build_payload(
                raw_items,
                frame_shape=source_frame.shape,
                count_fps=current_count_fps,
            )
            self._publish_payload(
                frame_id,
                payload,
                completed_at=completed_at,
                current_count_fps=current_count_fps,
                status_text=status_text,
            )
            last_processed_frame_id = frame_id
            last_completed_at = completed_at

    def _build_payload(
        self,
        raw_items: tuple[HandCountItem, ...],
        *,
        frame_shape: tuple[int, ...],
        count_fps: float,
    ) -> tuple[HandCountPayload, str]:
        roi_box = camera_roi_box(
            frame_shape,
            width_ratio=self._roi_width_ratio,
            height_ratio=self._roi_height_ratio,
        )
        roi_items = tuple(item for item in raw_items if _box_center_in_roi(item.box, roi_box=roi_box))
        ordered_items = _order_hand_items(roi_items)

        if len(ordered_items) > config.CAMERA_HAND_MAX_HANDS:
            payload = HandCountPayload(
                items=ordered_items[: config.CAMERA_HAND_MAX_HANDS],
                total_count=self._stable_payload.total_count if self._stable_payload is not None else 0,
                too_many_hands=True,
                fps=count_fps,
                warnings=(HAND_TOO_MANY_WARNING,),
            )
            return payload, HAND_TOO_MANY_WARNING

        if not ordered_items:
            return self._build_empty_payload(count_fps=count_fps)

        self._empty_frame_streak = 0
        candidate = HandCountPayload(
            items=ordered_items,
            total_count=sum(item.count for item in ordered_items),
            too_many_hands=False,
            fps=count_fps,
            warnings=(),
        )
        published_payload = self._stable_or_candidate_payload(candidate, count_fps=count_fps)
        return published_payload, camera_detection_status(
            config.CAMERA_MODE_HAND_COUNT,
            self._device_index,
            self._backend_name,
            has_detections=True,
            trailing_period=False,
        )

    def _build_empty_payload(self, *, count_fps: float) -> tuple[HandCountPayload, str]:
        self._empty_frame_streak += 1
        if self._empty_frame_streak >= config.CAMERA_HAND_EMPTY_RESET_FRAMES:
            self._history.clear()
            self._stable_payload = None
            payload = HandCountPayload(
                items=(),
                total_count=0,
                too_many_hands=False,
                fps=count_fps,
                warnings=(),
            )
        elif self._stable_payload is not None:
            payload = replace(self._stable_payload, fps=count_fps, warnings=())
        else:
            payload = HandCountPayload(
                items=(),
                total_count=0,
                too_many_hands=False,
                fps=count_fps,
                warnings=(),
            )
        status_text = camera_waiting_status(
            config.CAMERA_MODE_HAND_COUNT,
            self._device_index,
            self._backend_name,
            trailing_period=False,
        )
        return payload, status_text

    def _stable_or_candidate_payload(
        self,
        candidate: HandCountPayload,
        *,
        count_fps: float,
    ) -> HandCountPayload:
        self._history.append(candidate)
        total_counts = [payload.total_count for payload in self._history]
        top_total, top_frequency = Counter(total_counts).most_common(1)[0]
        if top_frequency >= config.CAMERA_HAND_STABLE_MIN_COUNT:
            for payload in reversed(self._history):
                if payload.total_count == top_total:
                    self._stable_payload = payload
                    break

        if self._stable_payload is not None:
            return replace(self._stable_payload, fps=count_fps, warnings=())
        return candidate

    def _publish_payload(
        self,
        frame_id: int,
        payload: HandCountPayload,
        *,
        completed_at: float,
        current_count_fps: float,
        status_text: str,
    ) -> None:
        with self._lock:
            self._latest_result = CameraInferenceResult(
                frame_id=frame_id,
                detections=(),
                average_score=0.0,
                completed_at=completed_at,
                mode=config.CAMERA_MODE_HAND_COUNT,
                warnings=payload.warnings,
                payload=payload,
            )
            self._worker_error = None
            self._ocr_fps = (
                current_count_fps
                if self._ocr_fps <= 0.0
                else ((self._ocr_fps * 0.7) + (current_count_fps * 0.3))
            )
            self._status_text = status_text

    def _set_worker_error(self, message: str) -> None:
        with self._lock:
            self._worker_error = message
            self._status_text = message


def _box_center_in_roi(
    box: tuple[int, int, int, int],
    *,
    roi_box: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    roi_x0, roi_y0, roi_x1, roi_y1 = roi_box
    return roi_x0 <= center_x <= roi_x1 and roi_y0 <= center_y <= roi_y1


def _order_hand_items(items: tuple[HandCountItem, ...]) -> tuple[HandCountItem, ...]:
    order = {"Left": 0, "Right": 1, "Unknown": 2}
    return tuple(sorted(items, key=lambda item: (order.get(item.handedness, 3), item.box[0], item.box[1])))


__all__ = ["HandCountRuntime"]
