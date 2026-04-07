from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ocr_engine import OCRResult
from core.recognition_service import DigitOCRService, RecognitionOutput


def _clone_results(results: list[OCRResult]) -> list[OCRResult]:
    return [
        OCRResult(
            text=result.text,
            score=result.score,
            box=[point[:] for point in result.box],
        )
        for result in results
    ]


class _ProcessorStub:
    def __init__(self, output_image: np.ndarray) -> None:
        self.output_image = output_image
        self.calls: list[np.ndarray] = []

    def enhance(self, image: np.ndarray) -> np.ndarray:
        self.calls.append(image.copy())
        return self.output_image.copy()


class _SentinelProcessor:
    def enhance(self, _image: np.ndarray) -> np.ndarray:
        raise AssertionError("unexpected processor usage")


class _EngineStub:
    def __init__(
        self,
        *,
        recognize_results: list[OCRResult] | None = None,
        annotated_image: np.ndarray | None = None,
    ) -> None:
        self.recognize_results = [] if recognize_results is None else recognize_results
        self.annotated_image = (
            np.zeros((16, 16, 3), dtype=np.uint8)
            if annotated_image is None
            else annotated_image
        )
        self.recognize_calls: list[np.ndarray] = []
        self.draw_calls: list[tuple[np.ndarray, list[OCRResult]]] = []

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        self.recognize_calls.append(image.copy())
        return _clone_results(self.recognize_results)

    def draw_results(self, image: np.ndarray, results: list[OCRResult]) -> np.ndarray:
        self.draw_calls.append((image.copy(), _clone_results(results)))
        return self.annotated_image.copy()


class ServiceContractCharacterizationTests(unittest.TestCase):
    def test_recognize_image_returns_recognition_output_with_sorted_mapped_results(self) -> None:
        input_image = np.full((100, 200, 3), 255, dtype=np.uint8)
        processed_image = np.full((50, 100, 3), 180, dtype=np.uint8)
        annotated_image = np.full((100, 200, 3), 77, dtype=np.uint8)
        raw_results = [
            OCRResult(text="2", score=0.81, box=[[40, 5], [50, 5], [50, 25], [40, 25]]),
            OCRResult(text="1", score=0.95, box=[[10, 5], [20, 5], [20, 25], [10, 25]]),
        ]

        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _ProcessorStub(processed_image)
        service.camera_processor = _SentinelProcessor()
        service.engine = _EngineStub(recognize_results=raw_results, annotated_image=annotated_image)
        service._image_pipeline = type(
            "_ImagePipelineStub",
            (),
            {
                "run": lambda _self, _image, results: (_clone_results(results), ["kept"]),
            },
        )()

        output = service.recognize_image(input_image, source_name="sample.png", annotate_on_original=True)

        self.assertIsInstance(output, RecognitionOutput)
        self.assertEqual(output.source_name, "sample.png")
        self.assertTrue(np.array_equal(output.input_image, input_image))
        self.assertTrue(np.array_equal(output.processed_image, processed_image))
        self.assertTrue(np.array_equal(output.annotated_image, annotated_image))
        self.assertEqual(output.warnings, ["kept"])
        self.assertEqual([item.text for item in output.results], ["1", "2"])
        self.assertEqual(output.results[0].box, [[20, 10], [40, 10], [40, 50], [20, 50]])
        self.assertEqual(output.combined_text, "1 2")
        self.assertEqual(output.summary_text, "1 (0.95), 2 (0.81)")
        self.assertEqual(len(service.engine.draw_calls), 1)
        self.assertTrue(np.array_equal(service.engine.draw_calls[0][0], input_image))

    def test_recognize_camera_frame_returns_plain_ocr_results(self) -> None:
        input_image = np.full((80, 160, 3), 255, dtype=np.uint8)
        processed_image = np.full((40, 80, 3), 210, dtype=np.uint8)

        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _SentinelProcessor()
        service.camera_processor = _ProcessorStub(processed_image)
        service.engine = _EngineStub()
        service._camera_digit_pipeline = type(
            "_CameraPipelineStub",
            (),
            {
                "run": lambda _self, _image, allow_fallback=True: (
                    [OCRResult(text="9", score=0.96, box=[[10, 5], [20, 5], [20, 25], [10, 25]])],
                    False,
                ),
            },
        )()

        results = service.recognize_camera_frame(input_image, source_name="camera")

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], OCRResult)
        self.assertEqual(results[0].text, "9")
        self.assertEqual(results[0].box, [[20, 10], [40, 10], [40, 50], [20, 50]])
        self.assertEqual(len(service.camera_processor.calls), 1)

    def test_recognize_board_frame_keeps_multi_char_results(self) -> None:
        grayscale_image = np.full((60, 120), 255, dtype=np.uint8)
        processed_image = np.full((30, 60, 3), 140, dtype=np.uint8)
        recognized_results = [
            OCRResult(text="7", score=0.83, box=[[35, 4], [45, 4], [45, 20], [35, 20]]),
            OCRResult(text="2026", score=0.91, box=[[5, 5], [25, 5], [25, 20], [5, 20]]),
        ]

        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _SentinelProcessor()
        service.camera_processor = _ProcessorStub(processed_image)
        service.engine = _EngineStub(recognize_results=recognized_results)

        results = service.recognize_board_frame(grayscale_image, source_name="board")

        self.assertEqual([item.text for item in results], ["2026", "7"])
        self.assertEqual(results[0].box, [[10, 10], [50, 10], [50, 40], [10, 40]])
        self.assertTrue(any(len(item.text) > 1 for item in results))
        self.assertEqual(len(service.camera_processor.calls), 1)

    def test_recognize_board_frame_can_return_board_warning_when_no_digits_found(self) -> None:
        grayscale_image = np.full((60, 120), 255, dtype=np.uint8)
        processed_image = np.full((30, 60, 3), 140, dtype=np.uint8)

        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _SentinelProcessor()
        service.camera_processor = _ProcessorStub(processed_image)
        service.engine = _EngineStub(recognize_results=[])

        results, warnings = service.recognize_board_frame(
            grayscale_image,
            source_name="board-empty",
            return_warnings=True,
        )

        self.assertEqual(results, [])
        self.assertEqual(warnings, [service.board_warning_text])

    def test_recognize_handwriting_returns_recognition_output(self) -> None:
        grayscale_image = np.full((40, 80), 255, dtype=np.uint8)
        annotated_image = np.full((40, 80, 3), 33, dtype=np.uint8)
        handwriting_results = [
            OCRResult(text="3", score=0.94, box=[[5, 5], [20, 5], [20, 30], [5, 30]]),
            OCRResult(text="4", score=0.92, box=[[30, 5], [45, 5], [45, 30], [30, 30]]),
        ]
        validated_images: list[np.ndarray] = []

        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _SentinelProcessor()
        service.camera_processor = _SentinelProcessor()
        service.engine = _EngineStub(annotated_image=annotated_image)
        service._handwriting_pipeline = type(
            "_HandwritingPipelineStub",
            (),
            {
                "run": lambda _self, image, progress_callback=None: (
                    validated_images.append(image.copy()) or _clone_results(handwriting_results),
                    ["warned"],
                ),
            },
        )()

        output = service.recognize_handwriting(grayscale_image, source_name="handwriting")

        self.assertIsInstance(output, RecognitionOutput)
        self.assertEqual([item.text for item in output.results], ["3", "4"])
        self.assertEqual(output.warnings, ["warned"])
        self.assertEqual(output.combined_text, "3 4")
        self.assertTrue(np.array_equal(output.annotated_image, annotated_image))
        self.assertEqual(len(validated_images), 1)
        self.assertEqual(validated_images[0].shape, (40, 80, 3))


if __name__ == "__main__":
    unittest.main()
