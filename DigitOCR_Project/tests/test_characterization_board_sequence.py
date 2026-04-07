from __future__ import annotations

import queue
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera import config as config
from camera.fast_path import build_camera_detections_from_results
from camera.protocol import CameraOCRWorkerConfig
from camera.roi import camera_roi_box
from camera.runtime import CameraOCRRuntime
from camera.state import CameraDetection, CameraInferenceResult
from core.ocr_engine import OCRResult


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


class BoardSequenceCharacterizationTests(unittest.TestCase):
    def test_prepare_board_frame_plan_uses_fixed_manual_roi(self) -> None:
        runtime = _IdleRuntime(service_factory=lambda: None, camera_mode=config.CAMERA_MODE_BOARD)
        frame = np.arange(120 * 200 * 3, dtype=np.uint8).reshape(120, 200, 3)
        roi_box = camera_roi_box(frame.shape, width_ratio=runtime.roi_width_ratio, height_ratio=runtime.roi_height_ratio)

        plan = runtime._prepare_board_frame_plan(4, frame)

        assert plan.task is not None
        self.assertEqual(plan.task.task_kind, "board_frame")
        self.assertEqual(plan.task.camera_mode, config.CAMERA_MODE_BOARD)
        self.assertEqual((plan.task.offset_x, plan.task.offset_y), roi_box[:2])
        self.assertTrue(np.array_equal(plan.task.ocr_frame, frame[roi_box[1] : roi_box[3], roi_box[0] : roi_box[2]]))

    def test_build_camera_detections_keeps_multi_char_results_for_board_mode(self) -> None:
        detections = build_camera_detections_from_results(
            [
                OCRResult(text="2026", score=0.93, box=[[10, 10], [80, 10], [80, 40], [10, 40]]),
                OCRResult(text="7", score=0.88, box=[[90, 10], [110, 10], [110, 40], [90, 40]]),
            ],
            frame_shape=(120, 160, 3),
            allow_multi_char=True,
        )

        self.assertEqual([item.text for item in detections], ["2026", "7"])

    def test_board_inference_result_formats_digits_as_rows(self) -> None:
        result = CameraInferenceResult(
            frame_id=1,
            detections=(
                CameraDetection(text="2026", score=0.93, box=(10, 10, 80, 40)),
                CameraDetection(text="7", score=0.88, box=(90, 10, 110, 40)),
                CameraDetection(text="8", score=0.91, box=(10, 60, 30, 90)),
            ),
            average_score=0.90,
            completed_at=1.0,
            mode=config.CAMERA_MODE_BOARD,
        )

        self.assertEqual(result.combined_text, "2026 7\n8")

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

        self.assertEqual(runtime._fast_processes, [])
        self.assertEqual(runtime._fast_task_queues, [])
        self.assertEqual(runtime._fast_result_queues, [])
        self.assertIsNotNone(runtime._ocr_process)
        runtime._stop_ocr_worker()


if __name__ == "__main__":
    unittest.main()
