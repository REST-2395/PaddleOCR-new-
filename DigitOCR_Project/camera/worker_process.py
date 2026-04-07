"""Worker-process entrypoints for live camera OCR."""

from __future__ import annotations

import multiprocessing as mp
import os

from . import config
from .protocol import CameraOCRTask, CameraOCRWorkerConfig, CameraOCRWorkerResult
from core.ocr_engine import OCRResult


def _apply_camera_worker_env_limits(cpu_threads: int) -> None:
    """Limit OCR worker CPU thread counts before model initialization."""
    thread_value = str(max(1, int(cpu_threads)))
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = thread_value


def _build_fast_worker_engine(worker_config: CameraOCRWorkerConfig):
    from core.ocr_engine import DigitOCREngine

    return DigitOCREngine(
        dict_path=worker_config.dict_path,
        use_gpu=worker_config.use_gpu,
        det_model_dir=worker_config.det_model_dir,
        rec_model_dir=worker_config.rec_model_dir,
        cls_model_dir=worker_config.cls_model_dir,
        ocr_version=worker_config.ocr_version,
        score_threshold=worker_config.score_threshold,
        load_detection_engine=False,
        cpu_threads=worker_config.cpu_threads,
        enable_mkldnn=worker_config.enable_mkldnn,
        use_textline_orientation=worker_config.use_textline_orientation,
        language=worker_config.language,
    )


def _build_service_worker(worker_config: CameraOCRWorkerConfig):
    from core.recognition_service import DigitOCRService

    return DigitOCRService(
        dict_path=worker_config.dict_path,
        use_gpu=worker_config.use_gpu,
        det_model_dir=worker_config.det_model_dir,
        rec_model_dir=worker_config.rec_model_dir,
        cls_model_dir=worker_config.cls_model_dir,
        ocr_version=worker_config.ocr_version,
        score_threshold=worker_config.score_threshold,
        cpu_threads=worker_config.cpu_threads,
        enable_mkldnn=worker_config.enable_mkldnn,
        use_textline_orientation=worker_config.use_textline_orientation,
        language=worker_config.language,
    )


def _initialize_worker_resources(worker_config: CameraOCRWorkerConfig):
    if worker_config.worker_kind == "fast":
        return _build_fast_worker_engine(worker_config), None
    return None, _build_service_worker(worker_config)


def _publish_worker_init_error(result_queue: "mp.Queue[CameraOCRWorkerResult]", error: Exception) -> None:
    try:
        result_queue.put(
            CameraOCRWorkerResult(
                frame_id=-1,
                worker_error=f"Camera OCR worker failed to initialize: {error}",
            )
        )
    except Exception:
        pass


def _run_fast_worker_task(engine, task: CameraOCRTask) -> tuple[list[OCRResult], list[str], bool, str]:
    review_results = engine.recognize_handwriting_blocks(task.candidate_images)
    results: list[OCRResult] = []
    for review_result, candidate_box in zip(review_results, task.candidate_boxes):
        if len(review_result.text) != 1 or float(review_result.score) < config.CAMERA_FAST_MIN_REVIEW_SCORE:
            continue
        x0, y0, x1, y1 = candidate_box
        results.append(
            OCRResult(
                text=review_result.text,
                score=float(review_result.score),
                box=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            )
        )
    return results, [], False, "fast_candidate"


def _run_service_worker_task(service, worker_config: CameraOCRWorkerConfig, task: CameraOCRTask):
    assert task.ocr_frame is not None
    if worker_config.worker_kind == "board":
        results, warnings = service.recognize_board_frame(task.ocr_frame, return_warnings=True)
        return list(results), list(warnings), False, "board_frame"

    results = service._get_camera_digit_pipeline().run_fallback(task.ocr_frame)
    return list(results), [], True, "fallback_roi"


def _publish_worker_result(
    result_queue: "mp.Queue[CameraOCRWorkerResult]",
    task: CameraOCRTask,
    *,
    results: list[OCRResult],
    warnings: list[str],
    fallback_used: bool,
    task_kind: str,
) -> None:
    result_queue.put(
        CameraOCRWorkerResult(
            frame_id=task.frame_id,
            task_kind=task_kind,
            results=tuple(results),
            scale_x=task.scale_x,
            scale_y=task.scale_y,
            offset_x=task.offset_x,
            offset_y=task.offset_y,
            fallback_used=fallback_used,
            warnings=tuple(warnings),
            generation=task.generation,
            started_at=task.started_at,
            inverse_matrix=task.inverse_matrix,
            camera_mode=task.camera_mode,
        )
    )


def _publish_worker_task_error(
    result_queue: "mp.Queue[CameraOCRWorkerResult]",
    task: CameraOCRTask,
    error: Exception,
) -> None:
    result_queue.put(
        CameraOCRWorkerResult(
            frame_id=task.frame_id,
            task_kind=task.task_kind,
            worker_error=f"Camera OCR worker failed: {error}",
            scale_x=task.scale_x,
            scale_y=task.scale_y,
            offset_x=task.offset_x,
            offset_y=task.offset_y,
            generation=task.generation,
            started_at=task.started_at,
            inverse_matrix=task.inverse_matrix,
            camera_mode=task.camera_mode,
        )
    )


def camera_ocr_worker_main(
    task_queue: "mp.Queue[CameraOCRTask | None]",
    result_queue: "mp.Queue[CameraOCRWorkerResult]",
    worker_config: CameraOCRWorkerConfig,
) -> None:
    """Child-process entry point for live camera OCR."""
    _apply_camera_worker_env_limits(worker_config.cpu_threads)

    try:
        engine, service = _initialize_worker_resources(worker_config)
    except Exception as exc:
        _publish_worker_init_error(result_queue, exc)
        return

    while True:
        task = task_queue.get()
        if task is None:
            return

        try:
            if worker_config.worker_kind == "fast":
                assert engine is not None
                results, warnings, fallback_used, task_kind = _run_fast_worker_task(engine, task)
            else:
                assert service is not None
                results, warnings, fallback_used, task_kind = _run_service_worker_task(service, worker_config, task)

            _publish_worker_result(
                result_queue,
                task,
                results=results,
                warnings=warnings,
                fallback_used=fallback_used,
                task_kind=task_kind,
            )
        except Exception as exc:
            _publish_worker_task_error(result_queue, task, exc)
            return


__all__ = [
    "_apply_camera_worker_env_limits",
    "camera_ocr_worker_main",
]
