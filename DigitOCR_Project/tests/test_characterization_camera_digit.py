from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

import numpy as np


import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera import config as config
from camera.protocol import CameraOCRWorkerResult
from camera.roi import camera_roi_box
from camera.runtime import CameraOCRRuntime
from camera.state import CameraDetection, CameraInferenceResult
from core.image_processor import ImageProcessor
from core.ocr_engine import OCRResult
from core.recognition_service import DigitOCRService, ImageCandidateBlock


class _CaptureStub:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _IdleRuntime(CameraOCRRuntime):
    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.01)

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.01)


class CameraDigitCharacterizationTests(unittest.TestCase):
    @staticmethod
    def make_service() -> DigitOCRService:
        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = ImageProcessor()
        service.camera_processor = ImageProcessor(
            min_short_side=256,
            bilateral_diameter=5,
            bilateral_sigma_color=45,
            bilateral_sigma_space=45,
            clahe_clip_limit=1.6,
        )
        service.engine = None
        return service

    def setUp(self) -> None:
        self.service = self.make_service()
        self.camera_pipeline = self.service._get_camera_digit_pipeline()

    def test_build_ocr_task_keeps_fixed_centered_roi_crop(self) -> None:
        runtime = CameraOCRRuntime(service_factory=lambda: None)
        frame = np.arange(120 * 200 * 3, dtype=np.uint8).reshape(120, 200, 3)
        roi_box = camera_roi_box(
            frame.shape,
            width_ratio=runtime.roi_width_ratio,
            height_ratio=runtime.roi_height_ratio,
        )

        task = runtime._build_ocr_task(5, frame, allow_fallback=False)

        self.assertEqual((task.offset_x, task.offset_y), roi_box[:2])
        self.assertEqual(task.ocr_frame.shape, (roi_box[3] - roi_box[1], roi_box[2] - roi_box[0], 3))
        self.assertFalse(task.allow_fallback)

    def test_fast_path_accepts_four_to_six_digits(self) -> None:
        image = np.full((180, 420, 3), 255, dtype=np.uint8)

        for digit_count in (4, 6):
            candidate_boxes = [(index * 20, 10, index * 20 + 12, 70) for index in range(digit_count)]
            self.camera_pipeline._collect_camera_fast_candidate_boxes = lambda _image, boxes=candidate_boxes: list(boxes)
            self.camera_pipeline._build_image_candidate_block = lambda _image, box: ImageCandidateBlock(
                display_box=box,
                region_image=np.full((60, 12, 3), 255, dtype=np.uint8),
                foreground_mask=np.full((60, 12), 255, dtype=np.uint8),
            )
            self.camera_pipeline._resolve_camera_fast_candidate = lambda block: (
                [
                    OCRResult(
                        text=str((block.display_box[0] // 20) + 1),
                        score=0.95,
                        box=self.camera_pipeline._region_box_to_polygon(block.display_box),
                    )
                ],
                False,
            )

            with self.subTest(digit_count=digit_count):
                results, fallback_needed = self.camera_pipeline._resolve_camera_fast_path(image)
                self.assertFalse(fallback_needed)
                self.assertEqual(len(results), digit_count)
                self.assertEqual([item.text for item in results], [str(index) for index in range(1, digit_count + 1)])

    def test_overcrowded_scene_requests_fallback_before_fast_review(self) -> None:
        image = np.full((180, 480, 3), 255, dtype=np.uint8)
        candidate_boxes = [(index * 20, 10, index * 20 + 12, 70) for index in range(7)]

        self.camera_pipeline._collect_camera_fast_candidate_boxes = lambda _image: list(candidate_boxes)
        self.camera_pipeline._resolve_camera_fast_candidate = lambda _block: (_ for _ in ()).throw(
            AssertionError("crowded digit scenes should fallback before fast review")
        )

        results, fallback_needed = self.camera_pipeline._resolve_camera_fast_path(image)

        self.assertEqual(results, [])
        self.assertTrue(fallback_needed)

    def test_more_complete_stale_fallback_result_replaces_fast_path_result(self) -> None:
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

    def test_roi_update_drops_old_generation_worker_results(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
