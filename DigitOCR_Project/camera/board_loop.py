"""Board-mode runtime loop helpers for live camera OCR."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .fast_path import _map_perspective_results, _scale_ocr_results, build_camera_detections_from_results
from .protocol import BoardFramePlan, CameraOCRTask
from .roi import _resize_for_ocr, camera_roi_has_foreground, crop_camera_roi
from .state import CameraInferenceResult
from core.ocr_engine import OCRResult
from desktop.messages import camera_detection_status, camera_recognizing_status, camera_waiting_task_status

if TYPE_CHECKING:
    from .runtime import CameraOCRRuntime


def submit_board_task(
    runtime: "CameraOCRRuntime",
    task: CameraOCRTask,
) -> bool:
    return runtime._submit_ocr_task(task)


def prepare_board_frame_plan(runtime: "CameraOCRRuntime", frame_id: int, frame: np.ndarray) -> BoardFramePlan:
    roi_frame, roi_box = crop_camera_roi(
        frame,
        width_ratio=runtime._roi_width_ratio,
        height_ratio=runtime._roi_height_ratio,
    )
    ocr_frame, scale_x, scale_y = _resize_for_ocr(roi_frame, max_side=runtime._max_ocr_side)
    task = CameraOCRTask(
        frame_id=frame_id,
        task_kind="board_frame",
        ocr_frame=ocr_frame,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=roi_box[0],
        offset_y=roi_box[1],
        allow_fallback=False,
        generation=runtime._roi_generation,
        inverse_matrix=None,
        camera_mode=runtime._camera_mode,
    )
    return BoardFramePlan(
        task=task,
        status_text=camera_recognizing_status(
            runtime._camera_mode,
            runtime._device_index,
            runtime._backend_name,
            trailing_period=False,
        ),
    )


def local_board_inference_loop(runtime: "CameraOCRRuntime") -> None:
    """Run board-mode OCR without child processes."""
    last_processed_frame_id = -1
    last_completed_at = 0.0

    while not runtime._stop_event.is_set():
        with runtime._lock:
            frame_id = runtime._latest_frame_id
            source_frame = None if runtime._latest_frame is None else runtime._latest_frame.copy()
            inference_resume_at = runtime._inference_resume_at

        if source_frame is None or frame_id == last_processed_frame_id:
            time.sleep(runtime._idle_seconds)
            continue

        now = time.perf_counter()
        if now < inference_resume_at:
            time.sleep(min(runtime._idle_seconds, inference_resume_at - now))
            continue

        if last_completed_at > 0.0:
            elapsed = now - last_completed_at
            if elapsed < runtime._ocr_interval_seconds:
                time.sleep(min(runtime._idle_seconds, runtime._ocr_interval_seconds - elapsed))
                continue

        if not camera_roi_has_foreground(
            source_frame,
            width_ratio=runtime._roi_width_ratio,
            height_ratio=runtime._roi_height_ratio,
        ):
            completed_at = time.perf_counter()
            runtime._publish_empty_camera_result(frame_id, completed_at=completed_at)
            last_processed_frame_id = frame_id
            last_completed_at = completed_at
            time.sleep(runtime._idle_seconds)
            continue

        board_plan = runtime._prepare_board_frame_plan(frame_id, source_frame)
        ocr_started_at = time.perf_counter()
        try:
            assert board_plan.task is not None
            ocr_output, warnings = runtime._run_board_ocr(board_plan.task, return_warnings=True)
            completed_at = time.perf_counter()
        except Exception as exc:
            completed_at = time.perf_counter()
            last_completed_at = completed_at
            runtime._set_worker_error(f"Camera OCR failed: {exc}")
            time.sleep(runtime._idle_seconds)
            continue

        detections = build_camera_detections_from_results(
            ocr_output,
            frame_shape=source_frame.shape,
            allow_multi_char=True,
        )
        average_score = (
            sum(item.score for item in detections) / float(len(detections))
            if detections
            else 0.0
        )
        ocr_delta = max(1e-6, completed_at - ocr_started_at)
        current_ocr_fps = 1.0 / ocr_delta
        with runtime._lock:
            runtime._latest_result = CameraInferenceResult(
                frame_id=frame_id,
                detections=detections,
                average_score=average_score,
                completed_at=completed_at,
                mode=runtime._camera_mode,
                warnings=tuple(warnings),
            )
            runtime._worker_error = None
            runtime._ocr_fps = (
                current_ocr_fps
                if runtime._ocr_fps <= 0.0
                else ((runtime._ocr_fps * 0.7) + (current_ocr_fps * 0.3))
            )
            runtime._status_text = camera_detection_status(
                runtime._camera_mode,
                runtime._device_index,
                runtime._backend_name,
                has_detections=bool(detections),
                trailing_period=False,
            )
        last_processed_frame_id = frame_id
        last_completed_at = completed_at


def process_board_inference_loop(runtime: "CameraOCRRuntime") -> None:
    """Submit board-mode OCR to the child process using the fixed manual ROI."""
    last_processed_frame_id = -1
    last_completed_at = 0.0

    while not runtime._stop_event.is_set():
        processed_frame_id, completed_at = runtime._drain_ocr_worker_results()
        if processed_frame_id == -1:
            time.sleep(runtime._idle_seconds)
            continue

        if processed_frame_id is not None:
            last_processed_frame_id = max(last_processed_frame_id, processed_frame_id)
            last_completed_at = completed_at

        if runtime._ocr_worker_exited():
            with runtime._lock:
                existing_error = runtime._worker_error
            if existing_error is None:
                runtime._set_worker_error("Camera OCR worker exited unexpectedly.")
            time.sleep(runtime._idle_seconds)
            continue

        with runtime._lock:
            frame_id = runtime._latest_frame_id
            source_frame = None if runtime._latest_frame is None else runtime._latest_frame.copy()
            inference_resume_at = runtime._inference_resume_at

        if source_frame is None or runtime._ocr_task_inflight or frame_id == last_processed_frame_id:
            time.sleep(runtime._idle_seconds)
            continue

        now = time.perf_counter()
        if now < inference_resume_at:
            time.sleep(min(runtime._idle_seconds, inference_resume_at - now))
            continue

        if last_completed_at > 0.0:
            elapsed = now - last_completed_at
            if elapsed < runtime._ocr_interval_seconds:
                time.sleep(min(runtime._idle_seconds, runtime._ocr_interval_seconds - elapsed))
                continue

        if not camera_roi_has_foreground(
            source_frame,
            width_ratio=runtime._roi_width_ratio,
            height_ratio=runtime._roi_height_ratio,
        ):
            completed_at = time.perf_counter()
            runtime._publish_empty_camera_result(frame_id, completed_at=completed_at)
            last_processed_frame_id = frame_id
            last_completed_at = completed_at
            time.sleep(runtime._idle_seconds)
            continue

        board_plan = runtime._prepare_board_frame_plan(frame_id, source_frame)
        try:
            assert board_plan.task is not None
            submitted = runtime._submit_board_task(board_plan.task)
            if submitted:
                runtime._status_text = board_plan.status_text
            else:
                runtime._status_text = camera_waiting_task_status(
                    runtime._device_index,
                    runtime._backend_name,
                    trailing_period=False,
                )
        except Exception as exc:
            runtime._set_worker_error(f"Camera OCR worker submission failed: {exc}")
            time.sleep(runtime._idle_seconds)


def run_board_ocr(
    runtime: "CameraOCRRuntime",
    task: CameraOCRTask,
    *,
    return_warnings: bool = False,
) -> list[OCRResult] | tuple[list[OCRResult], list[str]]:
    service = runtime._service_factory()
    assert task.ocr_frame is not None
    board_output = service.recognize_board_frame(
        task.ocr_frame,
        source_name=f"Camera {runtime._device_index}",
        return_warnings=return_warnings,
    )
    if return_warnings:
        assert isinstance(board_output, tuple)
        ocr_results, warnings = board_output
    else:
        assert isinstance(board_output, list)
        ocr_results, warnings = board_output, []
    if task.inverse_matrix is not None:
        mapped_results = _map_perspective_results(
            ocr_results,
            inverse_matrix=task.inverse_matrix,
            to_shape=runtime._latest_frame.shape if runtime._latest_frame is not None else task.ocr_frame.shape,
        )
        return (mapped_results, warnings) if return_warnings else mapped_results
    mapped_results = _scale_ocr_results(
        ocr_results,
        scale_x=task.scale_x,
        scale_y=task.scale_y,
        offset_x=task.offset_x,
        offset_y=task.offset_y,
    )
    return (mapped_results, warnings) if return_warnings else mapped_results


__all__ = [
    "local_board_inference_loop",
    "prepare_board_frame_plan",
    "process_board_inference_loop",
    "run_board_ocr",
    "submit_board_task",
]
