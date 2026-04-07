from __future__ import annotations

import sys
import time
import unittest
from collections import deque
from pathlib import Path
import queue
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera import config as config
from camera.fast_path import (
    build_camera_detections_from_results,
    camera_detection_signature,
    camera_result_is_fresh,
    filter_camera_detections,
    stable_camera_sequence,
    stabilize_camera_detections,
)
from camera.protocol import CameraOCRTask, CameraOCRWorkerConfig, CameraOCRWorkerResult, CameraTrack, PendingFastFrame
from camera.roi import camera_roi_box, camera_roi_foreground_ratio, camera_roi_has_foreground
from camera.runtime import CameraOCRRuntime
from camera.state import CameraDetection, CameraInferenceResult
from core.ocr_engine import OCRResult


class _CaptureStub:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class _RejectingQueue:
    def put_nowait(self, _item: object) -> None:
        raise queue.Full

    def get_nowait(self) -> object:
        raise queue.Empty


class _ProcessStub:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.started = False


class _ContextStub:
    def Queue(self, maxsize: int = 0) -> queue.Queue:
        return queue.Queue(maxsize=maxsize)

    def Process(self, *args, **kwargs) -> _ProcessStub:
        return _ProcessStub(*args, **kwargs)


class _IdleRuntime(CameraOCRRuntime):
    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.01)

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.01)


class _TimingRuntime(CameraOCRRuntime):
    def __init__(self, *, ocr_interval_seconds: float, processing_delay: float) -> None:
        super().__init__(
            service_factory=lambda: None,
            ocr_interval_seconds=ocr_interval_seconds,
            startup_grace_seconds=0.0,
        )
        self.processing_delay = processing_delay
        self.ocr_started_at: list[float] = []
        self.ocr_finished_at: list[float] = []
        self._capture_published = False

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._capture_published:
                with self._lock:
                    frame = np.full((120, 120, 3), 255, dtype=np.uint8)
                    frame[38:82, 50:70] = 0
                    self._latest_frame = frame
                    self._latest_frame_id = 1
                self._capture_published = True
            time.sleep(0.005)

    def _run_ocr(self, frame: np.ndarray, *, allow_fallback: bool = True) -> list[OCRResult]:
        del allow_fallback
        started_at = time.perf_counter()
        self.ocr_started_at.append(started_at)
        time.sleep(self.processing_delay)
        finished_at = time.perf_counter()
        self.ocr_finished_at.append(finished_at)

        if len(self.ocr_started_at) == 1:
            with self._lock:
                self._latest_frame = frame.copy()
                self._latest_frame_id = 2
        else:
            self._stop_event.set()

        return [OCRResult(text="7", score=0.99, box=[[2, 2], [12, 2], [12, 20], [2, 20]])]


class _BlankSkipRuntime(CameraOCRRuntime):
    def __init__(self) -> None:
        super().__init__(
            service_factory=lambda: None,
            ocr_interval_seconds=0.0,
            startup_grace_seconds=0.0,
        )
        self.ocr_calls = 0
        self._capture_published = False

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._capture_published:
                with self._lock:
                    self._latest_frame = np.full((90, 140, 3), 255, dtype=np.uint8)
                    self._latest_frame_id = 1
                self._capture_published = True
            time.sleep(0.005)

    def _run_ocr(self, frame: np.ndarray, *, allow_fallback: bool = True) -> list[OCRResult]:
        del frame, allow_fallback
        self.ocr_calls += 1
        return []


class CameraRuntimeTests(unittest.TestCase):
    def test_camera_roi_box_stays_centered_with_fixed_ratios(self) -> None:
        self.assertEqual(camera_roi_box((100, 200, 3)), (45, 26, 155, 74))
        self.assertEqual(camera_roi_box((120, 120, 3)), (27, 36, 93, 84))

    def test_camera_roi_foreground_helpers_detect_center_content(self) -> None:
        blank = np.full((120, 200, 3), 255, dtype=np.uint8)
        self.assertFalse(camera_roi_has_foreground(blank))

        roi_box = camera_roi_box(blank.shape)
        active = blank.copy()
        x0, y0, x1, y1 = roi_box
        active[y0 + 10 : y1 - 10, x0 + 20 : x0 + 36] = 0

        self.assertGreaterEqual(
            camera_roi_foreground_ratio(active, roi_box=roi_box),
            config.CAMERA_ROI_MIN_FOREGROUND_RATIO,
        )
        self.assertTrue(camera_roi_has_foreground(active, roi_box=roi_box))

    def test_build_camera_detections_from_results_sorts_boxes(self) -> None:
        results = [
            OCRResult(text="3", score=0.98, box=[[90, 40], [120, 40], [120, 90], [90, 90]]),
            OCRResult(text="1", score=0.95, box=[[10, 10], [35, 10], [35, 80], [10, 80]]),
            OCRResult(text="2", score=0.97, box=[[50, 12], [80, 12], [80, 82], [50, 82]]),
        ]

        detections = build_camera_detections_from_results(results, frame_shape=(120, 160, 3))

        self.assertEqual([item.text for item in detections], ["1", "2", "3"])

    def test_filter_camera_detections_hides_low_average_results(self) -> None:
        detections = (
            CameraDetection(text="1", score=0.30, box=(10, 10, 40, 90)),
            CameraDetection(text="2", score=0.40, box=(50, 10, 90, 90)),
        )

        filtered, state = filter_camera_detections(detections)

        self.assertEqual(filtered, ())
        self.assertEqual(state["reason"], "average_low")

    def test_filter_camera_detections_keeps_only_visible_boxes(self) -> None:
        detections = (
            CameraDetection(text="1", score=0.88, box=(10, 10, 40, 90)),
            CameraDetection(text="2", score=0.48, box=(50, 10, 90, 90)),
            CameraDetection(text="3", score=0.92, box=(100, 10, 140, 90)),
        )

        filtered, state = filter_camera_detections(detections)

        self.assertEqual([item.text for item in filtered], ["1", "3"])
        self.assertEqual(state["hidden_count"], 1)

    def test_filter_camera_detections_keeps_medium_confidence_boxes_with_new_threshold(self) -> None:
        detections = (
            CameraDetection(text="1", score=0.52, box=(10, 10, 40, 90)),
            CameraDetection(text="2", score=0.58, box=(50, 10, 90, 90)),
        )

        filtered, state = filter_camera_detections(detections)

        self.assertEqual([item.text for item in filtered], ["1", "2"])
        self.assertEqual(state["reason"], "ok")
        self.assertEqual(state["hidden_count"], 0)

    def test_stabilize_camera_detections_smooths_matching_boxes(self) -> None:
        previous_track = CameraTrack(
            track_id=1,
            detection=CameraDetection(text="5", score=0.90, box=(10, 10, 50, 90)),
            misses=0,
        )
        detections = (
            CameraDetection(text="5", score=0.97, box=(20, 20, 60, 100)),
        )

        stabilized, tracks, next_track_id = stabilize_camera_detections(
            detections,
            [previous_track],
            next_track_id=2,
            allow_missed_tracks=False,
        )

        self.assertEqual(next_track_id, 2)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(stabilized[0].text, "5")
        self.assertEqual(stabilized[0].box, (16, 16, 56, 96))

    def test_stable_camera_sequence_requires_repetition(self) -> None:
        history = deque(["12", "13", "12"], maxlen=6)

        unstable = stable_camera_sequence(deque(["12", "13"], maxlen=6), "13")
        stable = stable_camera_sequence(history, "12")

        self.assertEqual(unstable, "13")
        self.assertEqual(stable, "12")

    def test_camera_detection_signature_quantizes_boxes(self) -> None:
        detections = (
            CameraDetection(text="8", score=0.95, box=(11, 19, 49, 88)),
        )

        signature = camera_detection_signature("8", detections)

        self.assertEqual(signature[0], "8")
        self.assertEqual(signature[1][0][0], "8")

    def test_start_with_capture_uses_preopened_device(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=2, backend_name="TEST")
        try:
            self.assertTrue(runtime.is_running)
            self.assertEqual(runtime.device_index, 2)
            self.assertEqual(runtime.backend_name, "TEST")
            self.assertIs(runtime._capture, capture)
        finally:
            runtime.stop()

        self.assertTrue(capture.released)

    def test_get_snapshot_omits_frame_when_gui_already_consumed_it(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()
        frame = np.full((24, 32, 3), 7, dtype=np.uint8)
        result = CameraInferenceResult(
            frame_id=3,
            detections=(CameraDetection(text="5", score=0.91, box=(1, 2, 10, 20)),),
            average_score=0.91,
        )

        runtime.start_with_capture(capture, device_index=1, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = frame
                runtime._latest_frame_id = 3
                runtime._latest_result = result
                runtime._capture_fps = 14.5
                runtime._ocr_fps = 6.2
                runtime._status_text = "Camera 1 (TEST) running"

            snapshot = runtime.get_snapshot(last_frame_id=3)
        finally:
            runtime.stop()

        self.assertFalse(snapshot.has_new_frame)
        self.assertIsNone(snapshot.frame_bgr)
        self.assertEqual(snapshot.frame_id, 3)
        self.assertEqual(snapshot.latest_result, result)
        self.assertEqual(snapshot.capture_fps, 14.5)
        self.assertEqual(snapshot.ocr_fps, 6.2)
        self.assertEqual(snapshot.status_text, "Camera 1 (TEST) running")

    def test_get_snapshot_returns_shared_frame_for_new_publish(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()
        frame = np.full((18, 18, 3), 5, dtype=np.uint8)

        runtime.start_with_capture(capture, device_index=4, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = frame
                runtime._latest_frame_id = 9
                runtime._status_text = "Camera 4 (TEST) running"

            snapshot = runtime.get_snapshot(last_frame_id=8)
        finally:
            runtime.stop()

        self.assertTrue(snapshot.has_new_frame)
        self.assertIs(snapshot.frame_bgr, frame)
        self.assertEqual(snapshot.frame_id, 9)

    def test_camera_result_is_fresh_uses_age_window(self) -> None:
        self.assertTrue(camera_result_is_fresh(10.0, now=10.6, max_age_seconds=config.CAMERA_RESULT_HOLD_SECONDS))
        self.assertFalse(camera_result_is_fresh(10.0, now=10.9, max_age_seconds=config.CAMERA_RESULT_HOLD_SECONDS))

    def test_build_ocr_task_crops_center_roi_and_tracks_offset(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)

        task = runtime._build_ocr_task(11, frame, allow_fallback=False)
        roi_box = camera_roi_box(frame.shape)
        x0, y0, x1, y1 = roi_box

        self.assertEqual(task.frame_id, 11)
        self.assertEqual((task.offset_x, task.offset_y), (x0, y0))
        self.assertFalse(task.allow_fallback)
        self.assertEqual(task.ocr_frame.shape, (y1 - y0, x1 - x0, 3))
        self.assertTrue(np.array_equal(task.ocr_frame, frame[y0:y1, x0:x1]))

    def test_camera_fallback_window_blocks_rapid_reentry(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)

        self.assertTrue(runtime._camera_fallback_allowed(now=10.0))
        runtime._record_camera_fallback(completed_at=10.0)
        self.assertFalse(runtime._camera_fallback_allowed(now=11.0))
        self.assertTrue(runtime._camera_fallback_allowed(now=12.0))

    def test_submit_fast_frame_distributes_candidates_across_fast_workers(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._fast_task_queues = [queue.Queue(), queue.Queue()]
        runtime._fast_result_queues = []
        runtime._fast_task_inflight = [False, False]
        runtime._fast_pending_frames = {}

        frame = np.full((240, 360, 3), 255, dtype=np.uint8)
        frame[80:160, 90:120] = 0
        frame[80:160, 220:250] = 0

        submitted = runtime._submit_fast_frame(3, frame, allow_fallback=False)

        self.assertTrue(submitted)
        self.assertEqual(runtime._fast_task_queues[0].qsize(), 1)
        self.assertEqual(runtime._fast_task_queues[1].qsize(), 1)
        self.assertEqual(runtime._fast_pending_frames[3].expected_batches, 2)
        self.assertEqual(runtime._fast_pending_frames[3].total_candidates, 2)

    def test_submit_fast_frame_accepts_six_candidates(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._fast_task_queues = [queue.Queue(), queue.Queue()]
        runtime._fast_result_queues = []
        runtime._fast_task_inflight = [False, False]
        runtime._fast_pending_frames = {}

        dummy_crop = np.full((24, 24, 3), 255, dtype=np.uint8)
        candidate_boxes = tuple((index * 14, 0, index * 14 + 10, 20) for index in range(6))

        with patch("camera.digit_loop.extract_camera_fast_candidates", return_value=((dummy_crop,) * 6, candidate_boxes)):
            submitted = runtime._submit_fast_frame(4, np.full((120, 220, 3), 255, dtype=np.uint8), allow_fallback=False)

        self.assertTrue(submitted)
        self.assertEqual(runtime._fast_task_queues[0].qsize(), 1)
        self.assertEqual(runtime._fast_task_queues[1].qsize(), 1)
        self.assertEqual(runtime._fast_pending_frames[4].expected_batches, 2)
        self.assertEqual(runtime._fast_pending_frames[4].total_candidates, 6)

    def test_submit_fast_frame_uses_fallback_for_overcrowded_candidates(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._fast_task_queues = [queue.Queue(), queue.Queue()]
        runtime._fast_result_queues = []
        runtime._fast_task_inflight = [False, False]
        runtime._fast_pending_frames = {}
        runtime._ocr_task_inflight = False

        dummy_crop = np.full((24, 24, 3), 255, dtype=np.uint8)
        crowded_boxes = (
            (0, 0, 10, 20),
            (12, 0, 22, 20),
            (24, 0, 34, 20),
            (36, 0, 46, 20),
            (48, 0, 58, 20),
            (60, 0, 70, 20),
            (72, 0, 82, 20),
        )
        fallback_submissions: list[CameraOCRTask] = []

        with patch("camera.digit_loop.extract_camera_fast_candidates", return_value=((dummy_crop,) * 7, crowded_boxes)):
            runtime._submit_ocr_task = lambda task: fallback_submissions.append(task)
            submitted = runtime._submit_fast_frame(5, np.full((120, 200, 3), 255, dtype=np.uint8), allow_fallback=True)

        self.assertFalse(submitted)
        self.assertEqual(len(fallback_submissions), 1)
        self.assertEqual(fallback_submissions[0].task_kind, "fallback_roi")

    def test_submit_ocr_task_replaces_pending_queue_item_without_blocking(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._ocr_task_queue = queue.Queue(maxsize=1)
        runtime._roi_generation = 3
        old_task = CameraOCRTask(frame_id=1, generation=3)
        new_task = CameraOCRTask(frame_id=2)
        runtime._ocr_task_queue.put_nowait(old_task)

        submitted = runtime._submit_ocr_task(new_task)

        self.assertTrue(submitted)
        queued_task = runtime._ocr_task_queue.get_nowait()
        self.assertEqual(queued_task.frame_id, 2)
        self.assertEqual(queued_task.generation, 3)
        self.assertGreater(queued_task.started_at, 0.0)

    def test_replace_fast_task_replaces_pending_queue_item_without_blocking(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._fast_task_queues = [queue.Queue(maxsize=1)]
        runtime._fast_task_inflight = [False]
        runtime._fast_task_queues[0].put_nowait(CameraOCRTask(frame_id=1))

        submitted = runtime._replace_fast_task(0, CameraOCRTask(frame_id=2))

        self.assertTrue(submitted)
        queued_task = runtime._fast_task_queues[0].get_nowait()
        self.assertEqual(queued_task.frame_id, 2)
        self.assertTrue(runtime._fast_task_inflight[0])

    def test_update_roi_size_bumps_generation_and_drains_pending_task_queues(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._roi_generation = 4
        runtime._ocr_task_queue = queue.Queue(maxsize=1)
        runtime._ocr_task_queue.put_nowait(CameraOCRTask(frame_id=10))
        runtime._fast_task_queues = [queue.Queue(maxsize=1)]
        runtime._fast_task_queues[0].put_nowait(CameraOCRTask(frame_id=11))
        runtime._fast_task_inflight = [True]
        runtime._fast_pending_frames = {9: PendingFastFrame(frame_id=9, expected_batches=1, total_candidates=1)}
        runtime._ocr_task_inflight = True

        runtime.update_roi_size(0.5, 0.5)

        self.assertEqual(runtime._roi_generation, 5)
        self.assertTrue(runtime._ocr_task_queue.empty())
        self.assertTrue(runtime._fast_task_queues[0].empty())
        self.assertEqual(runtime._fast_pending_frames, {})
        self.assertEqual(runtime._fast_task_inflight, [False])
        self.assertFalse(runtime._ocr_task_inflight)

    def test_submit_fast_frame_partial_batch_failure_tracks_only_successful_batches(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._fast_task_queues = [queue.Queue(maxsize=1), _RejectingQueue()]
        runtime._fast_result_queues = [queue.Queue(), queue.Queue()]
        runtime._fast_task_inflight = [False, False]
        runtime._fast_pending_frames = {}
        runtime._roi_generation = 2
        runtime._submit_ocr_task = lambda task: True

        dummy_crop = np.full((24, 24, 3), 255, dtype=np.uint8)
        candidate_boxes = ((0, 0, 12, 20), (20, 0, 32, 20))
        with patch("camera.digit_loop.extract_camera_fast_candidates", return_value=((dummy_crop, dummy_crop), candidate_boxes)):
            submitted = runtime._submit_fast_frame(6, np.full((120, 200, 3), 255, dtype=np.uint8), allow_fallback=True)

        self.assertTrue(submitted)
        self.assertEqual(runtime._fast_pending_frames[6].expected_batches, 1)
        runtime._fast_result_queues[0].put_nowait(
            CameraOCRWorkerResult(
                frame_id=6,
                task_kind="fast_candidate",
                results=(OCRResult(text="3", score=0.95, box=[[0, 0], [12, 0], [12, 20], [0, 20]]),),
                generation=2,
                started_at=time.perf_counter() - 0.2,
            )
        )

        runtime._drain_fast_worker_results()

        self.assertNotIn(6, runtime._fast_pending_frames)

    def test_submit_fast_frame_all_failed_batches_clears_pending_frame(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._fast_task_queues = [_RejectingQueue(), _RejectingQueue()]
        runtime._fast_result_queues = []
        runtime._fast_task_inflight = [False, False]
        runtime._fast_pending_frames = {}
        runtime._ocr_task_inflight = False
        runtime._submit_ocr_task = lambda task: False

        dummy_crop = np.full((24, 24, 3), 255, dtype=np.uint8)
        candidate_boxes = ((0, 0, 12, 20), (20, 0, 32, 20))
        with patch("camera.digit_loop.extract_camera_fast_candidates", return_value=((dummy_crop, dummy_crop), candidate_boxes)):
            submitted = runtime._submit_fast_frame(7, np.full((120, 200, 3), 255, dtype=np.uint8), allow_fallback=True)

        self.assertFalse(submitted)
        self.assertNotIn(7, runtime._fast_pending_frames)

    def test_local_inference_loop_skips_blank_roi_without_calling_ocr(self) -> None:
        runtime = _BlankSkipRuntime()
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            deadline = time.perf_counter() + 1.0
            latest_result: CameraInferenceResult | None = None
            while time.perf_counter() < deadline:
                with runtime._lock:
                    latest_result = runtime._latest_result
                if latest_result is not None and latest_result.frame_id == 1:
                    break
                time.sleep(0.01)
            else:
                self.fail("Timed out waiting for blank ROI skip result.")
        finally:
            runtime.stop()

        assert latest_result is not None
        self.assertEqual(runtime.ocr_calls, 0)
        self.assertEqual(latest_result.detections, ())

    def test_inference_loop_limits_from_previous_completion(self) -> None:
        runtime = _TimingRuntime(ocr_interval_seconds=0.08, processing_delay=0.04)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            deadline = time.perf_counter() + 2.0
            latest_result: CameraInferenceResult | None = None
            while time.perf_counter() < deadline:
                with runtime._lock:
                    latest_result = runtime._latest_result
                if len(runtime.ocr_started_at) >= 2 and latest_result is not None and latest_result.frame_id == 2:
                    break
                time.sleep(0.01)
            else:
                self.fail("Timed out waiting for timed camera OCR runs.")
        finally:
            runtime.stop()

        assert latest_result is not None
        self.assertEqual(len(runtime.ocr_started_at), 2)
        self.assertGreater(latest_result.completed_at, 0.0)
        self.assertGreaterEqual(runtime.ocr_started_at[1] - runtime.ocr_finished_at[0], 0.06)

    def test_apply_ocr_worker_result_scales_worker_boxes_back_to_preview(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        latest_result: CameraInferenceResult | None = None
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((100, 200, 3), dtype=np.uint8)
                runtime._ocr_task_started_at = time.perf_counter() - 0.2

            frame_id = runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=5,
                    results=(OCRResult(text="3", score=0.88, box=[[10, 5], [30, 5], [30, 25], [10, 25]]),),
                    scale_x=2.0,
                    scale_y=4.0,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        self.assertEqual(frame_id, 5)
        assert latest_result is not None
        self.assertEqual(latest_result.frame_id, 5)
        self.assertEqual(latest_result.detections[0].box, (20, 20, 60, 99))

    def test_apply_ocr_worker_result_offsets_boxes_from_roi_back_to_preview(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        latest_result: CameraInferenceResult | None = None
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((120, 200, 3), dtype=np.uint8)
                runtime._ocr_task_started_at = time.perf_counter() - 0.2

            frame_id = runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=6,
                    results=(OCRResult(text="8", score=0.91, box=[[5, 5], [15, 5], [15, 25], [5, 25]]),),
                    scale_x=2.0,
                    scale_y=2.0,
                    offset_x=40,
                    offset_y=10,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        self.assertEqual(frame_id, 6)
        assert latest_result is not None
        self.assertEqual(latest_result.detections[0].box, (50, 20, 70, 60))

    def test_apply_ocr_worker_result_rejects_older_frame_id(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((100, 200, 3), dtype=np.uint8)
                runtime._latest_result = CameraInferenceResult(frame_id=10, detections=(), average_score=0.0, completed_at=1.0)

            frame_id = runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=8,
                    task_kind="fast_candidate",
                    results=(OCRResult(text="2", score=0.9, box=[[5, 5], [15, 5], [15, 25], [5, 25]]),),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.2,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        self.assertEqual(frame_id, 8)
        assert latest_result is not None
        self.assertEqual(latest_result.frame_id, 10)

    def test_apply_ocr_worker_result_accepts_more_complete_stale_fallback_result(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((120, 240, 3), dtype=np.uint8)
                runtime._latest_result = CameraInferenceResult(
                    frame_id=12,
                    detections=(
                        CameraDetection(text="1", score=0.81, box=(10, 10, 30, 50)),
                        CameraDetection(text="2", score=0.80, box=(40, 10, 60, 50)),
                        CameraDetection(text="3", score=0.79, box=(70, 10, 90, 50)),
                    ),
                    average_score=0.80,
                    completed_at=1.0,
                    mode=config.CAMERA_MODE_DIGIT,
                )

            frame_id = runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=11,
                    task_kind="fallback_roi",
                    results=(
                        OCRResult(text="1", score=0.90, box=[[10, 10], [30, 10], [30, 50], [10, 50]]),
                        OCRResult(text="2", score=0.91, box=[[40, 10], [60, 10], [60, 50], [40, 50]]),
                        OCRResult(text="3", score=0.92, box=[[70, 10], [90, 10], [90, 50], [70, 50]]),
                        OCRResult(text="4", score=0.93, box=[[100, 10], [120, 10], [120, 50], [100, 50]]),
                    ),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.2,
                    fallback_used=True,
                    camera_mode=config.CAMERA_MODE_DIGIT,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        self.assertEqual(frame_id, 12)
        assert latest_result is not None
        self.assertEqual(latest_result.frame_id, 12)
        self.assertEqual([item.text for item in latest_result.detections], ["1", "2", "3", "4"])

    def test_apply_ocr_worker_result_accepts_stale_fallback_with_same_count_higher_confidence(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((120, 240, 3), dtype=np.uint8)
                runtime._latest_result = CameraInferenceResult(
                    frame_id=12,
                    detections=(
                        CameraDetection(text="4", score=0.60, box=(10, 10, 30, 50)),
                        CameraDetection(text="5", score=0.62, box=(40, 10, 60, 50)),
                    ),
                    average_score=0.61,
                    completed_at=1.0,
                    mode=config.CAMERA_MODE_DIGIT,
                )

            frame_id = runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=11,
                    task_kind="fallback_roi",
                    results=(
                        OCRResult(text="8", score=0.94, box=[[10, 10], [30, 10], [30, 50], [10, 50]]),
                        OCRResult(text="9", score=0.96, box=[[40, 10], [60, 10], [60, 50], [40, 50]]),
                    ),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.2,
                    fallback_used=True,
                    camera_mode=config.CAMERA_MODE_DIGIT,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        self.assertEqual(frame_id, 12)
        assert latest_result is not None
        self.assertEqual(latest_result.frame_id, 12)
        self.assertEqual([item.text for item in latest_result.detections], ["8", "9"])

    def test_apply_ocr_worker_result_rejects_stale_fallback_when_not_better(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((120, 240, 3), dtype=np.uint8)
                runtime._latest_result = CameraInferenceResult(
                    frame_id=12,
                    detections=(
                        CameraDetection(text="5", score=0.94, box=(10, 10, 30, 50)),
                        CameraDetection(text="6", score=0.92, box=(40, 10, 60, 50)),
                    ),
                    average_score=0.93,
                    completed_at=1.0,
                    mode=config.CAMERA_MODE_DIGIT,
                )

            frame_id = runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=11,
                    task_kind="fallback_roi",
                    results=(
                        OCRResult(text="1", score=0.70, box=[[10, 10], [30, 10], [30, 50], [10, 50]]),
                        OCRResult(text="2", score=0.72, box=[[40, 10], [60, 10], [60, 50], [40, 50]]),
                    ),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.2,
                    fallback_used=True,
                    camera_mode=config.CAMERA_MODE_DIGIT,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        self.assertEqual(frame_id, 11)
        assert latest_result is not None
        self.assertEqual(latest_result.frame_id, 12)
        self.assertEqual([item.text for item in latest_result.detections], ["5", "6"])

    def test_apply_ocr_worker_result_allows_replacing_same_frame_id(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((100, 200, 3), dtype=np.uint8)
            runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=12,
                    results=(OCRResult(text="1", score=0.8, box=[[5, 5], [15, 5], [15, 25], [5, 25]]),),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.4,
                ),
                completed_at=time.perf_counter() - 0.2,
            )
            runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=12,
                    results=(OCRResult(text="9", score=0.95, box=[[8, 5], [18, 5], [18, 25], [8, 25]]),),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.2,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        assert latest_result is not None
        self.assertEqual(latest_result.frame_id, 12)
        self.assertEqual([item.text for item in latest_result.detections], ["9"])

    def test_apply_ocr_worker_result_drops_old_generation_result(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((100, 200, 3), dtype=np.uint8)
            stale_generation = runtime._roi_generation
            runtime.update_roi_size(0.45, 0.45)

            runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=14,
                    results=(OCRResult(text="4", score=0.9, box=[[5, 5], [15, 5], [15, 25], [5, 25]]),),
                    generation=stale_generation,
                    started_at=time.perf_counter() - 0.2,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        self.assertIsNone(latest_result)

    def test_drain_fast_worker_results_ignores_old_generation_batches(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        runtime._fast_task_queues = [queue.Queue(maxsize=1)]
        runtime._fast_result_queues = [queue.Queue()]
        runtime._fast_task_inflight = [True]
        runtime._fast_pending_frames = {
            20: PendingFastFrame(
                frame_id=20,
                expected_batches=1,
                total_candidates=1,
                generation=1,
                started_at=time.perf_counter() - 0.2,
            )
        }
        runtime._roi_generation = 1
        with runtime._lock:
            runtime._latest_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        runtime._fast_result_queues[0].put_nowait(
            CameraOCRWorkerResult(
                frame_id=20,
                task_kind="fast_candidate",
                results=(OCRResult(text="7", score=0.95, box=[[0, 0], [12, 0], [12, 20], [0, 20]]),),
                generation=1,
                started_at=time.perf_counter() - 0.2,
            )
        )

        runtime.update_roi_size(0.5, 0.5)
        processed_frame_id, _ = runtime._drain_fast_worker_results()

        self.assertEqual(processed_frame_id, 20)
        self.assertIsNone(runtime._latest_result)

    def test_apply_ocr_worker_result_uses_worker_started_at_for_fps(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((100, 200, 3), dtype=np.uint8)
                runtime._ocr_task_started_at = time.perf_counter() - 999.0

            runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=16,
                    results=(OCRResult(text="6", score=0.9, box=[[5, 5], [15, 5], [15, 25], [5, 25]]),),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.2,
                ),
                completed_at=time.perf_counter(),
            )
            ocr_fps = runtime._ocr_fps
        finally:
            runtime.stop()

        self.assertGreater(ocr_fps, 3.0)
        self.assertLess(ocr_fps, 8.0)

    def test_apply_ocr_worker_result_records_fallback_usage(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((120, 200, 3), dtype=np.uint8)
                runtime._ocr_task_started_at = time.perf_counter() - 0.2

            before = runtime._last_fallback_completed_at
            runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=8,
                    results=(),
                    fallback_used=True,
                ),
                completed_at=12.5,
            )
            self.assertEqual(before, 0.0)
            self.assertEqual(runtime._last_fallback_completed_at, 12.5)
        finally:
            runtime.stop()

    def test_apply_ocr_worker_result_preserves_worker_warnings_on_latest_result(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None, camera_mode=config.CAMERA_MODE_BOARD)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            with runtime._lock:
                runtime._latest_frame = np.zeros((120, 200, 3), dtype=np.uint8)
                runtime._ocr_task_started_at = time.perf_counter() - 0.2

            runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(
                    frame_id=18,
                    task_kind="board_frame",
                    results=(),
                    warnings=("检测到黑板模式下的复杂数字区域，请尽量保持数字清晰并分开。",),
                    generation=runtime._roi_generation,
                    started_at=time.perf_counter() - 0.2,
                    camera_mode=config.CAMERA_MODE_BOARD,
                ),
                completed_at=time.perf_counter(),
            )
            latest_result = runtime._latest_result
        finally:
            runtime.stop()

        assert latest_result is not None
        self.assertEqual(latest_result.warnings, ("检测到黑板模式下的复杂数字区域，请尽量保持数字清晰并分开。",))

    def test_apply_ocr_worker_result_surfaces_worker_error(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None)
        capture = _CaptureStub()

        runtime.start_with_capture(capture, device_index=0, backend_name="TEST")
        try:
            frame_id = runtime._apply_ocr_worker_result(
                CameraOCRWorkerResult(frame_id=7, worker_error="worker exploded"),
                completed_at=time.perf_counter(),
            )
            snapshot = runtime.get_snapshot()
        finally:
            runtime.stop()

        self.assertEqual(frame_id, -1)
        self.assertEqual(snapshot.error_message, "worker exploded")

    def test_build_camera_detections_from_results_keeps_multi_char_boxes_for_board_mode(self) -> None:
        detections = build_camera_detections_from_results(
            [OCRResult(text="2026", score=0.93, box=[[10, 10], [80, 10], [80, 40], [10, 40]])],
            frame_shape=(120, 160, 3),
            allow_multi_char=True,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].text, "2026")

    def test_prepare_board_frame_plan_uses_fixed_roi_crop_for_board_mode(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None, camera_mode=config.CAMERA_MODE_BOARD)
        frame = np.arange(120 * 200 * 3, dtype=np.uint8).reshape(120, 200, 3)
        roi_box = camera_roi_box(frame.shape, width_ratio=runtime.roi_width_ratio, height_ratio=runtime.roi_height_ratio)

        plan = runtime._prepare_board_frame_plan(4, frame)

        assert plan.task is not None
        self.assertIsNone(plan.task.inverse_matrix)
        self.assertEqual(plan.task.camera_mode, config.CAMERA_MODE_BOARD)
        self.assertEqual(plan.task.task_kind, "board_frame")
        self.assertEqual((plan.task.offset_x, plan.task.offset_y), roi_box[:2])
        self.assertTrue(np.array_equal(plan.task.ocr_frame, frame[roi_box[1] : roi_box[3], roi_box[0] : roi_box[2]]))

    def test_prepare_board_frame_plan_respects_manual_roi_size(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None, camera_mode=config.CAMERA_MODE_BOARD)
        runtime.update_roi_size(0.70, 0.55)
        frame = np.full((180, 320, 3), 255, dtype=np.uint8)
        roi_box = camera_roi_box(frame.shape, width_ratio=0.70, height_ratio=0.55)

        plan = runtime._prepare_board_frame_plan(6, frame)

        assert plan.task is not None
        self.assertEqual((plan.task.offset_x, plan.task.offset_y), roi_box[:2])
        self.assertEqual(plan.task.ocr_frame.shape, (roi_box[3] - roi_box[1], roi_box[2] - roi_box[0], 3))

    def test_start_ocr_worker_skips_fast_workers_for_board_mode(self) -> None:
        runtime = CameraOCRRuntime(
            worker_config=CameraOCRWorkerConfig(
                dict_path="digits_dict.txt",
                camera_mode=config.CAMERA_MODE_BOARD,
                enable_mkldnn=True,
                use_textline_orientation=False,
            ),
            camera_mode=config.CAMERA_MODE_BOARD,
        )

        with patch("camera.runtime.mp.get_context", return_value=_ContextStub()):
            runtime._start_ocr_worker()

        self.assertEqual(runtime.camera_mode, config.CAMERA_MODE_BOARD)
        self.assertEqual(runtime._fast_processes, [])
        self.assertEqual(runtime._fast_task_queues, [])
        self.assertIsNotNone(runtime._ocr_process)
        runtime._stop_ocr_worker()

    def test_start_ocr_worker_prefers_pythonw_executable_on_windows(self) -> None:
        runtime = CameraOCRRuntime(
            worker_config=CameraOCRWorkerConfig(
                dict_path="digits_dict.txt",
                camera_mode=config.CAMERA_MODE_BOARD,
                enable_mkldnn=True,
                use_textline_orientation=False,
            ),
            camera_mode=config.CAMERA_MODE_BOARD,
        )

        with patch("camera.runtime_worker_control.sys.platform", "win32"), patch(
            "camera.runtime_worker_control.sys.executable",
            str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        ), patch("camera.runtime_worker_control.Path.exists", return_value=True), patch(
            "camera.runtime_worker_control.mp.set_executable"
        ) as mocked_set_executable, patch("camera.runtime.mp.get_context", return_value=_ContextStub()):
            runtime._start_ocr_worker()

        mocked_set_executable.assert_called_once()
        runtime._stop_ocr_worker()


if __name__ == "__main__":
    unittest.main()
