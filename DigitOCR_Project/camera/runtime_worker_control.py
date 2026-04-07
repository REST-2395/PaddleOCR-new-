"""Worker-management helpers for the camera runtime facade."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
from collections.abc import Sequence

import numpy as np

from . import config
from .protocol import (
    CameraOCRTask,
    CameraOCRWorkerConfig,
    CameraOCRWorkerResult,
    PendingFastFrame,
)
from .state import CameraDetection, CameraInferenceResult
from .runtime_support import replace_queue_item_latest_only
from .worker_process import camera_ocr_worker_main as _camera_ocr_worker_main_impl


class RuntimeWorkerControlMixin:
    """Start, stop, and coordinate OCR worker processes."""

    def _start_ocr_worker(self) -> None:
        if self._worker_config is None:
            return

        _configure_windows_worker_executable()
        context = mp.get_context("spawn")
        self._fast_processes = []
        self._fast_task_queues = []
        self._fast_result_queues = []
        self._fast_task_inflight = []

        if self._mode_profile.uses_fast_workers:
            for worker_index in range(config.CAMERA_FAST_WORKER_COUNT):
                task_queue = context.Queue(maxsize=1)
                result_queue = context.Queue(maxsize=2)
                worker = context.Process(
                    target=_camera_ocr_worker_main_impl,
                    args=(
                        task_queue,
                        result_queue,
                        CameraOCRWorkerConfig(
                            dict_path=self._worker_config.dict_path,
                            worker_kind="fast",
                            ocr_version=self._worker_config.ocr_version,
                            score_threshold=self._worker_config.score_threshold,
                            use_gpu=self._worker_config.use_gpu,
                            det_model_dir=self._worker_config.det_model_dir,
                            rec_model_dir=self._worker_config.rec_model_dir,
                            cls_model_dir=self._worker_config.cls_model_dir,
                            cpu_threads=self._mode_profile.fast_worker_threads,
                            camera_mode=self._mode_profile.mode,
                            enable_mkldnn=self._worker_config.enable_mkldnn,
                            use_textline_orientation=self._worker_config.use_textline_orientation,
                            language=self._worker_config.language,
                        ),
                    ),
                    name=f"camera-fast-worker-{worker_index + 1}",
                    daemon=True,
                )
                worker.start()
                self._fast_processes.append(worker)
                self._fast_task_queues.append(task_queue)
                self._fast_result_queues.append(result_queue)
                self._fast_task_inflight.append(False)

        self._ocr_task_queue = context.Queue(maxsize=1)
        self._ocr_result_queue = context.Queue(maxsize=2)
        self._ocr_process = context.Process(
            target=_camera_ocr_worker_main_impl,
            args=(
                self._ocr_task_queue,
                self._ocr_result_queue,
                CameraOCRWorkerConfig(
                    dict_path=self._worker_config.dict_path,
                    worker_kind=self._mode_profile.primary_worker_kind,
                    ocr_version=self._worker_config.ocr_version,
                    score_threshold=self._worker_config.score_threshold,
                    use_gpu=self._worker_config.use_gpu,
                    det_model_dir=self._worker_config.det_model_dir,
                    rec_model_dir=self._worker_config.rec_model_dir,
                    cls_model_dir=self._worker_config.cls_model_dir,
                    cpu_threads=(
                        self._worker_config.cpu_threads
                        if self._camera_mode == config.CAMERA_MODE_BOARD
                        else self._mode_profile.primary_worker_threads
                    ),
                    camera_mode=self._camera_mode,
                    enable_mkldnn=self._worker_config.enable_mkldnn,
                    use_textline_orientation=self._worker_config.use_textline_orientation,
                    language=self._worker_config.language,
                ),
            ),
            name=self._mode_profile.primary_worker_name,
            daemon=True,
        )
        self._ocr_process.start()
        self._ocr_task_inflight = False
        self._ocr_task_started_at = 0.0
        self._ocr_task_frame_id = -1

    def _drain_fast_worker_results(self) -> tuple[int | None, float]:
        processed_frame_id: int | None = None
        completed_at = 0.0
        for worker_index, result_queue in enumerate(self._fast_result_queues):
            while True:
                try:
                    worker_result = result_queue.get_nowait()
                except queue.Empty:
                    break

                self._fast_task_inflight[worker_index] = False
                completed_at = time.perf_counter()
                processed_frame_id = (
                    worker_result.frame_id
                    if processed_frame_id is None
                    else max(processed_frame_id, worker_result.frame_id)
                )
                if worker_result.worker_error is not None:
                    self._set_worker_error(worker_result.worker_error)
                    self._fast_pending_frames.pop(worker_result.frame_id, None)
                    continue

                pending = self._fast_pending_frames.get(worker_result.frame_id)
                if pending is None:
                    continue

                pending.completed_batches += 1
                pending.results.extend(worker_result.results)
                if pending.completed_batches < pending.expected_batches:
                    continue

                self._fast_pending_frames.pop(worker_result.frame_id, None)
                self._handle_completed_fast_frame(
                    pending,
                    completed_at=completed_at,
                )
        return processed_frame_id, completed_at

    def _handle_completed_fast_frame(self, pending: PendingFastFrame, *, completed_at: float) -> None:
        from .digit_loop import handle_completed_fast_frame

        handle_completed_fast_frame(self, pending, completed_at=completed_at)

    def _build_fallback_task(self, frame_id: int, frame: np.ndarray) -> CameraOCRTask:
        from .digit_loop import build_fallback_task

        return build_fallback_task(self, frame_id, frame)

    def _submit_fast_frame(
        self,
        frame_id: int,
        frame: np.ndarray,
        *,
        allow_fallback: bool,
    ) -> bool:
        from .digit_loop import submit_fast_frame

        return submit_fast_frame(
            self,
            frame_id,
            frame,
            allow_fallback=allow_fallback,
        )

    def _replace_fast_task(self, worker_index: int, task: CameraOCRTask) -> bool:
        from .digit_loop import replace_fast_task

        return replace_fast_task(self, worker_index, task)

    def _stop_ocr_worker(self) -> None:
        for task_queue in self._fast_task_queues:
            try:
                task_queue.put(None, timeout=0.1)
            except Exception:
                pass

        for process in self._fast_processes:
            process.join(timeout=max(config.CAMERA_THREAD_JOIN_TIMEOUT_MS, 1) / 1000.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.2)

        for ipc_queue in [*self._fast_task_queues, *self._fast_result_queues]:
            try:
                ipc_queue.close()
            except Exception:
                pass
            try:
                ipc_queue.join_thread()
            except Exception:
                pass

        process = self._ocr_process
        task_queue = self._ocr_task_queue
        result_queue = self._ocr_result_queue

        if task_queue is not None:
            try:
                task_queue.put(None, timeout=0.1)
            except Exception:
                pass

        if process is not None:
            process.join(timeout=max(config.CAMERA_THREAD_JOIN_TIMEOUT_MS, 1) / 1000.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.2)

        for ipc_queue in (task_queue, result_queue):
            if ipc_queue is None:
                continue
            try:
                ipc_queue.close()
            except Exception:
                pass
            try:
                ipc_queue.join_thread()
            except Exception:
                pass

        self._ocr_process = None
        self._ocr_task_queue = None
        self._ocr_result_queue = None
        self._ocr_task_inflight = False
        self._ocr_task_started_at = 0.0
        self._ocr_task_frame_id = -1
        self._fast_processes = []
        self._fast_task_queues = []
        self._fast_result_queues = []
        self._fast_task_inflight = []

    def _build_ocr_task(self, frame_id: int, frame: np.ndarray, *, allow_fallback: bool = True) -> CameraOCRTask:
        from .digit_loop import build_ocr_task

        return build_ocr_task(self, frame_id, frame, allow_fallback=allow_fallback)

    @staticmethod
    def _fallback_result_is_more_complete(
        detections: Sequence[CameraDetection],
        *,
        latest_result: CameraInferenceResult,
        average_score: float,
    ) -> bool:
        if latest_result.mode != config.CAMERA_MODE_DIGIT:
            return False

        detection_count = len(detections)
        latest_count = len(latest_result.detections)
        if detection_count > latest_count:
            return True
        if detection_count == latest_count and detection_count > 0 and average_score > float(latest_result.average_score):
            return True
        return False

    def _should_accept_stale_fallback_result(
        self,
        worker_result: CameraOCRWorkerResult,
        *,
        latest_result: CameraInferenceResult | None,
        detections: Sequence[CameraDetection],
        average_score: float,
    ) -> bool:
        if latest_result is None:
            return False
        if worker_result.task_kind != "fallback_roi" or worker_result.camera_mode != config.CAMERA_MODE_DIGIT:
            return False
        if worker_result.frame_id >= latest_result.frame_id:
            return False
        return self._fallback_result_is_more_complete(
            detections,
            latest_result=latest_result,
            average_score=average_score,
        )

    def _apply_ocr_worker_result(
        self,
        worker_result: CameraOCRWorkerResult,
        *,
        completed_at: float,
    ) -> int:
        from .digit_loop import apply_ocr_worker_result

        return apply_ocr_worker_result(self, worker_result, completed_at=completed_at)

    def _publish_empty_camera_result(
        self,
        frame_id: int,
        *,
        completed_at: float,
    ) -> None:
        from .digit_loop import publish_empty_camera_result

        publish_empty_camera_result(self, frame_id, completed_at=completed_at)

    def _drain_ocr_worker_results(self) -> tuple[int | None, float]:
        from .digit_loop import drain_ocr_worker_results

        return drain_ocr_worker_results(self)

    def _ocr_worker_exited(self) -> bool:
        return self._ocr_process is not None and not self._ocr_process.is_alive()

    def _fast_worker_exited(self) -> bool:
        return any(not process.is_alive() for process in self._fast_processes)

    def _submit_ocr_task(self, task: CameraOCRTask) -> bool:
        from .digit_loop import submit_ocr_task

        return submit_ocr_task(self, task)

    def _submit_board_task(self, task: CameraOCRTask) -> bool:
        from .board_loop import submit_board_task

        return submit_board_task(self, task)

    @staticmethod
    def _replace_queue_item_latest_only(task_queue: object, task: CameraOCRTask) -> bool:
        return replace_queue_item_latest_only(task_queue, task)


def _configure_windows_worker_executable() -> None:
    """Hide spawned OCR worker consoles on Windows by preferring pythonw.exe."""
    if sys.platform != "win32":
        return

    current_executable = Path(sys.executable).resolve()
    pythonw_executable = current_executable.with_name("pythonw.exe")
    if not pythonw_executable.exists():
        return

    try:
        mp.set_executable(str(pythonw_executable))
    except Exception:
        return
