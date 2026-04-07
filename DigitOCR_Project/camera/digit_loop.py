"""Digit-mode runtime loop helpers for live camera OCR."""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from . import config
from .fast_path import (
    _map_perspective_results,
    _scale_ocr_results,
    build_camera_detections_from_results,
    extract_camera_fast_candidates,
)
from .protocol import CameraOCRTask, CameraOCRWorkerResult, PendingFastFrame
from .roi import _resize_for_ocr, camera_roi_has_foreground, crop_camera_roi
from .state import CameraInferenceResult
from core.ocr_engine import OCRResult
from desktop.messages import (
    CAMERA_WARNING_RUNTIME_STATUS,
    camera_detection_status,
    camera_parallel_status,
    camera_waiting_status,
)

if TYPE_CHECKING:
    from .state import CameraBox, CameraDetection
    from .runtime import CameraOCRRuntime


def handle_completed_fast_frame(
    runtime: "CameraOCRRuntime",
    pending: PendingFastFrame,
    *,
    completed_at: float,
) -> None:
    fast_results = tuple(pending.results)
    if fast_results:
        runtime._apply_ocr_worker_result(
            CameraOCRWorkerResult(
                frame_id=pending.frame_id,
                task_kind="fast_candidate",
                results=fast_results,
                generation=pending.generation,
                started_at=pending.started_at,
            ),
            completed_at=completed_at,
        )

    should_fallback = (
        pending.total_candidates == 0
        or len(fast_results) == 0
        or len(fast_results) < pending.total_candidates
        or (
            fast_results
            and (sum(float(item.score) for item in fast_results) / float(len(fast_results)))
            < config.CAMERA_FAST_MIN_REVIEW_SCORE
        )
    )
    if (
        should_fallback
        and pending.fallback_task is not None
        and not runtime._ocr_task_inflight
        and runtime._camera_fallback_allowed(now=completed_at)
    ):
        try:
            runtime._submit_ocr_task(pending.fallback_task)
        except Exception as exc:
            runtime._set_worker_error(f"Camera OCR worker submission failed: {exc}")


def build_fallback_task(runtime: "CameraOCRRuntime", frame_id: int, frame: np.ndarray) -> CameraOCRTask:
    roi_frame, roi_box = crop_camera_roi(
        frame,
        width_ratio=runtime._roi_width_ratio,
        height_ratio=runtime._roi_height_ratio,
    )
    ocr_frame, scale_x, scale_y = _resize_for_ocr(roi_frame, max_side=runtime._max_ocr_side)
    return CameraOCRTask(
        frame_id=frame_id,
        task_kind="fallback_roi",
        ocr_frame=ocr_frame,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=roi_box[0],
        offset_y=roi_box[1],
        allow_fallback=False,
        generation=runtime._roi_generation,
    )


def submit_fast_frame(
    runtime: "CameraOCRRuntime",
    frame_id: int,
    frame: np.ndarray,
    *,
    allow_fallback: bool,
    fast_candidate_extractor: Callable[..., tuple[tuple[np.ndarray, ...], tuple["CameraBox", ...]]] | None = None,
) -> bool:
    extractor = extract_camera_fast_candidates if fast_candidate_extractor is None else fast_candidate_extractor
    candidate_images, candidate_boxes = extractor(
        frame,
        width_ratio=runtime._roi_width_ratio,
        height_ratio=runtime._roi_height_ratio,
    )
    candidate_count = len(candidate_boxes)
    fallback_task = runtime._build_fallback_task(frame_id, frame) if allow_fallback else None

    if candidate_count == 0 or candidate_count > config.CAMERA_FAST_MAX_CANDIDATES:
        if fallback_task is not None and not runtime._ocr_task_inflight and runtime._camera_fallback_allowed():
            fallback_submitted = runtime._submit_ocr_task(fallback_task)
            if not fallback_submitted:
                runtime._publish_empty_camera_result(frame_id, completed_at=time.perf_counter())
        else:
            runtime._publish_empty_camera_result(frame_id, completed_at=time.perf_counter())
        return False

    available_workers = [index for index, inflight in enumerate(runtime._fast_task_inflight) if not inflight]
    if not available_workers:
        return False

    batch_count = min(len(available_workers), candidate_count)
    worker_indexes = available_workers[:batch_count]
    batched_candidate_indexes = [list(batch) for batch in np.array_split(np.arange(candidate_count), batch_count)]
    started_at = time.perf_counter()
    pending = PendingFastFrame(
        frame_id=frame_id,
        expected_batches=0,
        total_candidates=candidate_count,
        fallback_task=fallback_task,
        generation=runtime._roi_generation,
        started_at=started_at,
    )
    runtime._fast_pending_frames[frame_id] = pending

    for worker_index, batch_indexes in zip(worker_indexes, batched_candidate_indexes):
        selected_indexes = [int(item) for item in batch_indexes.tolist()] if hasattr(batch_indexes, "tolist") else list(batch_indexes)
        task = CameraOCRTask(
            frame_id=frame_id,
            task_kind="fast_candidate",
            candidate_images=tuple(candidate_images[index] for index in selected_indexes),
            candidate_boxes=tuple(candidate_boxes[index] for index in selected_indexes),
            allow_fallback=allow_fallback,
            generation=runtime._roi_generation,
            started_at=started_at,
        )
        if runtime._replace_fast_task(worker_index, task):
            pending.expected_batches += 1

    if pending.expected_batches == 0:
        runtime._fast_pending_frames.pop(frame_id, None)
        if fallback_task is not None and not runtime._ocr_task_inflight and runtime._camera_fallback_allowed():
            fallback_submitted = runtime._submit_ocr_task(fallback_task)
            if not fallback_submitted:
                runtime._publish_empty_camera_result(frame_id, completed_at=time.perf_counter())
        else:
            runtime._publish_empty_camera_result(frame_id, completed_at=time.perf_counter())
        return False

    runtime._fast_last_submitted_frame_id = frame_id
    return True


def replace_fast_task(runtime: "CameraOCRRuntime", worker_index: int, task: CameraOCRTask) -> bool:
    task_queue = runtime._fast_task_queues[worker_index]
    submitted = runtime._replace_queue_item_latest_only(task_queue, task)
    if submitted:
        runtime._fast_task_inflight[worker_index] = True
    return submitted


def build_ocr_task(
    runtime: "CameraOCRRuntime",
    frame_id: int,
    frame: np.ndarray,
    *,
    allow_fallback: bool = True,
) -> CameraOCRTask:
    roi_frame, roi_box = crop_camera_roi(
        frame,
        width_ratio=runtime._roi_width_ratio,
        height_ratio=runtime._roi_height_ratio,
    )
    ocr_frame, scale_x, scale_y = _resize_for_ocr(roi_frame, max_side=runtime._max_ocr_side)
    return CameraOCRTask(
        frame_id=frame_id,
        ocr_frame=ocr_frame,
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=roi_box[0],
        offset_y=roi_box[1],
        allow_fallback=allow_fallback,
        generation=runtime._roi_generation,
        camera_mode=runtime._camera_mode,
    )


def apply_ocr_worker_result(
    runtime: "CameraOCRRuntime",
    worker_result: CameraOCRWorkerResult,
    *,
    completed_at: float,
) -> int:
    if worker_result.worker_error is not None:
        set_worker_error(runtime, worker_result.worker_error)
        return -1

    with runtime._lock:
        frame_shape = runtime._latest_frame.shape if runtime._latest_frame is not None else (1, 1, 3)
        current_generation = runtime._roi_generation

    effective_generation = worker_result.generation if worker_result.generation > 0 else current_generation
    if effective_generation != current_generation:
        return worker_result.frame_id

    mapped_results = _map_camera_worker_results(worker_result, frame_shape=frame_shape)
    detections, average_score = _build_camera_worker_detections(
        worker_result,
        mapped_results,
        frame_shape=frame_shape,
    )
    current_ocr_fps = _compute_worker_ocr_fps(worker_result, completed_at=completed_at)
    published_frame_id = _publish_camera_worker_result(
        runtime,
        worker_result,
        detections=detections,
        average_score=average_score,
        completed_at=completed_at,
        current_ocr_fps=current_ocr_fps,
        effective_generation=effective_generation,
    )
    if published_frame_id < 0:
        return worker_result.frame_id
    if worker_result.fallback_used:
        runtime._record_camera_fallback(completed_at=completed_at)
    return published_frame_id


def _map_camera_worker_results(
    worker_result: CameraOCRWorkerResult,
    *,
    frame_shape: tuple[int, ...],
) -> list[OCRResult]:
    if worker_result.inverse_matrix is not None:
        return _map_perspective_results(
            worker_result.results,
            inverse_matrix=worker_result.inverse_matrix,
            to_shape=frame_shape,
        )
    return _scale_ocr_results(
        worker_result.results,
        scale_x=worker_result.scale_x,
        scale_y=worker_result.scale_y,
        offset_x=worker_result.offset_x,
        offset_y=worker_result.offset_y,
    )


def _build_camera_worker_detections(
    worker_result: CameraOCRWorkerResult,
    mapped_results: list[OCRResult],
    *,
    frame_shape: tuple[int, ...],
) -> tuple[tuple[CameraDetection, ...], float]:
    detections = build_camera_detections_from_results(
        mapped_results,
        frame_shape=frame_shape,
        allow_multi_char=worker_result.camera_mode == config.CAMERA_MODE_BOARD,
    )
    average_score = sum(item.score for item in detections) / float(len(detections)) if detections else 0.0
    return detections, average_score


def _compute_worker_ocr_fps(
    worker_result: CameraOCRWorkerResult,
    *,
    completed_at: float,
) -> float:
    if worker_result.started_at <= 0.0:
        return 0.0
    ocr_delta = max(1e-6, completed_at - worker_result.started_at)
    return 1.0 / ocr_delta


def _publish_camera_worker_result(
    runtime: "CameraOCRRuntime",
    worker_result: CameraOCRWorkerResult,
    *,
    detections: tuple[CameraDetection, ...],
    average_score: float,
    completed_at: float,
    current_ocr_fps: float,
    effective_generation: int,
) -> int:
    published_frame_id = worker_result.frame_id
    with runtime._lock:
        latest_result = runtime._latest_result
        if effective_generation != runtime._roi_generation:
            return -1
        if latest_result is not None and worker_result.frame_id < latest_result.frame_id:
            if not runtime._should_accept_stale_fallback_result(
                worker_result,
                latest_result=latest_result,
                detections=detections,
                average_score=average_score,
            ):
                return -1
            published_frame_id = latest_result.frame_id
        runtime._latest_result = CameraInferenceResult(
            frame_id=published_frame_id,
            detections=detections,
            average_score=average_score,
            completed_at=completed_at,
            mode=worker_result.camera_mode,
            warnings=worker_result.warnings,
        )
        runtime._worker_error = None
        if current_ocr_fps > 0.0:
            runtime._ocr_fps = (
                current_ocr_fps
                if runtime._ocr_fps <= 0.0
                else ((runtime._ocr_fps * 0.7) + (current_ocr_fps * 0.3))
            )
        runtime._status_text = camera_detection_status(
            worker_result.camera_mode,
            runtime._device_index,
            runtime._backend_name,
            has_detections=bool(detections),
            trailing_period=False,
        )
    return published_frame_id


def publish_empty_camera_result(
    runtime: "CameraOCRRuntime",
    frame_id: int,
    *,
    completed_at: float,
) -> None:
    """Publish one empty result so the GUI can hold then clear stale digits."""
    with runtime._lock:
        if runtime._latest_result is not None and frame_id < runtime._latest_result.frame_id:
            return
        runtime._latest_result = CameraInferenceResult(
            frame_id=frame_id,
            detections=(),
            average_score=0.0,
            completed_at=completed_at,
            mode=runtime._camera_mode,
        )
        runtime._worker_error = None
        runtime._status_text = camera_waiting_status(
            runtime._camera_mode,
            runtime._device_index,
            runtime._backend_name,
            trailing_period=False,
        )


def drain_ocr_worker_results(runtime: "CameraOCRRuntime") -> tuple[int | None, float]:
    if runtime._ocr_result_queue is None:
        return None, 0.0

    processed_frame_id: int | None = None
    completed_at = 0.0
    while True:
        try:
            worker_result = runtime._ocr_result_queue.get_nowait()
        except queue.Empty:
            break

        completed_at = time.perf_counter()
        result_frame_id = apply_ocr_worker_result(runtime, worker_result, completed_at=completed_at)
        processed_frame_id = result_frame_id if processed_frame_id is None else max(processed_frame_id, result_frame_id)
        runtime._ocr_task_inflight = False
        runtime._ocr_task_started_at = 0.0
        runtime._ocr_task_frame_id = -1
    return processed_frame_id, completed_at


def submit_ocr_task(runtime: "CameraOCRRuntime", task: CameraOCRTask) -> bool:
    if runtime._ocr_task_queue is None:
        raise RuntimeError("Camera OCR task queue is unavailable.")

    if task.started_at <= 0.0:
        task.started_at = time.perf_counter()
    task.generation = runtime._roi_generation
    submitted = runtime._replace_queue_item_latest_only(runtime._ocr_task_queue, task)
    if not submitted:
        return False
    runtime._ocr_task_inflight = True
    runtime._ocr_task_started_at = task.started_at
    runtime._ocr_task_frame_id = task.frame_id
    return True


def local_inference_loop(runtime: "CameraOCRRuntime") -> None:
    """Consume only the latest unseen preview frame and refresh OCR at a fixed interval."""
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

        if not camera_roi_has_foreground(source_frame):
            completed_at = time.perf_counter()
            runtime._publish_empty_camera_result(frame_id, completed_at=completed_at)
            last_processed_frame_id = frame_id
            last_completed_at = completed_at
            time.sleep(runtime._idle_seconds)
            continue

        ocr_started_at = time.perf_counter()
        completed_at = 0.0
        allow_fallback = runtime._camera_fallback_allowed(now=now)
        try:
            ocr_output = runtime._run_ocr(source_frame, allow_fallback=allow_fallback)
            completed_at = time.perf_counter()
        except Exception as exc:
            completed_at = time.perf_counter()
            last_completed_at = completed_at
            runtime._set_worker_error(f"Camera OCR failed: {exc}")
            time.sleep(runtime._idle_seconds)
            continue

        if isinstance(ocr_output, tuple):
            ocr_results, fallback_used = ocr_output
        else:
            ocr_results, fallback_used = ocr_output, False
        if fallback_used:
            runtime._record_camera_fallback(completed_at=completed_at)

        detections = build_camera_detections_from_results(ocr_results, frame_shape=source_frame.shape)
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
                warnings=(),
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


def process_inference_loop(runtime: "CameraOCRRuntime") -> None:
    """Submit live OCR work to the isolated child process and consume its results."""
    last_processed_frame_id = -1
    last_completed_at = 0.0

    while not runtime._stop_event.is_set():
        fast_processed_frame_id, fast_completed_at = runtime._drain_fast_worker_results()
        processed_frame_id, completed_at = drain_ocr_worker_results(runtime)
        if processed_frame_id == -1:
            time.sleep(runtime._idle_seconds)
            continue

        if fast_completed_at > completed_at:
            completed_at = fast_completed_at
            if fast_processed_frame_id is not None:
                processed_frame_id = fast_processed_frame_id

        if processed_frame_id is not None:
            last_processed_frame_id = max(last_processed_frame_id, processed_frame_id)
            last_completed_at = completed_at

        if runtime._fast_worker_exited():
            set_worker_error(runtime, "Camera fast OCR worker exited unexpectedly.")
            time.sleep(runtime._idle_seconds)
            continue

        if runtime._ocr_worker_exited():
            with runtime._lock:
                existing_error = runtime._worker_error
            if existing_error is None:
                set_worker_error(runtime, "Camera OCR worker exited unexpectedly.")
            time.sleep(runtime._idle_seconds)
            continue

        with runtime._lock:
            frame_id = runtime._latest_frame_id
            source_frame = None if runtime._latest_frame is None else runtime._latest_frame.copy()
            inference_resume_at = runtime._inference_resume_at

        if source_frame is None:
            time.sleep(runtime._idle_seconds)
            continue

        now = time.perf_counter()
        if now < inference_resume_at:
            time.sleep(min(runtime._idle_seconds, inference_resume_at - now))
            continue

        if runtime._fast_pending_frames or frame_id == last_processed_frame_id:
            time.sleep(runtime._idle_seconds)
            continue

        if last_completed_at > 0.0:
            elapsed = now - last_completed_at
            if elapsed < runtime._ocr_interval_seconds:
                time.sleep(min(runtime._idle_seconds, runtime._ocr_interval_seconds - elapsed))
                continue

        if not camera_roi_has_foreground(source_frame):
            completed_at = time.perf_counter()
            runtime._publish_empty_camera_result(frame_id, completed_at=completed_at)
            last_processed_frame_id = frame_id
            last_completed_at = completed_at
            time.sleep(runtime._idle_seconds)
            continue

        try:
            submitted = runtime._submit_fast_frame(
                frame_id,
                source_frame,
                allow_fallback=runtime._camera_fallback_allowed(now=now),
            )
            if submitted:
                runtime._status_text = camera_parallel_status(
                    runtime._device_index,
                    runtime._backend_name,
                    trailing_period=False,
                )
        except Exception as exc:
            runtime._set_worker_error(f"Camera OCR worker submission failed: {exc}")
            time.sleep(runtime._idle_seconds)


def run_ocr(
    runtime: "CameraOCRRuntime",
    frame: np.ndarray,
    *,
    allow_fallback: bool = True,
) -> tuple[list[OCRResult], bool]:
    service = runtime._service_factory()
    task = build_ocr_task(runtime, -1, frame, allow_fallback=allow_fallback)
    ocr_results, fallback_used = service._recognize_camera_frame_internal(
        task.ocr_frame,
        source_name=f"Camera {runtime._device_index}",
        allow_fallback=task.allow_fallback,
    )
    return _scale_ocr_results(
        ocr_results,
        scale_x=task.scale_x,
        scale_y=task.scale_y,
        offset_x=task.offset_x,
        offset_y=task.offset_y,
    ), fallback_used


def set_worker_error(runtime: "CameraOCRRuntime", message: str) -> None:
    with runtime._lock:
        runtime._worker_error = message
        runtime._status_text = CAMERA_WARNING_RUNTIME_STATUS


__all__ = [
    "apply_ocr_worker_result",
    "build_fallback_task",
    "build_ocr_task",
    "drain_ocr_worker_results",
    "handle_completed_fast_frame",
    "local_inference_loop",
    "process_inference_loop",
    "publish_empty_camera_result",
    "replace_fast_task",
    "run_ocr",
    "set_worker_error",
    "submit_fast_frame",
    "submit_ocr_task",
]
