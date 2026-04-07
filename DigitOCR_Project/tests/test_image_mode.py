from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.image_processor import ImageProcessor
from core.ocr_engine import OCRResult, TextOnlyResult
from core.recognition_service import DigitOCRService, ImageCandidate, ImageCandidateBlock, ImageReviewResult


class ImageModeAlgorithmTests(unittest.TestCase):
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
        self.image_pipeline = self.service._get_image_pipeline()
        self.camera_pipeline = self.service._get_camera_digit_pipeline()

    def test_recognize_camera_frame_uses_camera_fast_path(self) -> None:
        class _ProcessorStub:
            def __init__(self) -> None:
                self.enhance_calls = 0

            def enhance(self, image: np.ndarray) -> np.ndarray:
                self.enhance_calls += 1
                return np.full((240, 320, 3), 255, dtype=np.uint8)

        class _ProcessorSentinel:
            def enhance(self, _image: np.ndarray) -> np.ndarray:
                raise AssertionError("camera OCR should use the dedicated camera processor")

        class _EngineStub:
            def __init__(self) -> None:
                self.recognize_calls = 0

            def recognize(self, _image: np.ndarray) -> list[OCRResult]:
                self.recognize_calls += 1
                return [
                    OCRResult(text="12", score=0.98, box=[[20, 30], [220, 30], [220, 120], [20, 120]]),
                ]

            def draw_results(self, _image: np.ndarray, _results: list[OCRResult]) -> np.ndarray:
                raise AssertionError("camera OCR should not draw annotated previews")

        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _ProcessorSentinel()
        service.camera_processor = _ProcessorStub()
        service.engine = _EngineStub()
        fast_results = [
            OCRResult(text="1", score=0.95, box=[[20, 30], [110, 30], [110, 120], [20, 120]]),
            OCRResult(text="2", score=0.94, box=[[130, 30], [220, 30], [220, 120], [130, 120]]),
        ]
        service._camera_digit_pipeline = type(
            "_CameraPipelineStub",
            (),
            {
                "run": lambda _self, _image, allow_fallback=True: (fast_results, False),
            },
        )()

        results = service.recognize_camera_frame(image)

        self.assertEqual([result.text for result in results], ["1", "2"])
        self.assertEqual(service.camera_processor.enhance_calls, 1)
        self.assertEqual(service.engine.recognize_calls, 0)
        self.assertEqual(results[0].box, [[10, 15], [55, 15], [55, 60], [10, 60]])

    def test_recognize_camera_frame_uses_fallback_when_fast_path_requests_it(self) -> None:
        class _ProcessorStub:
            def enhance(self, image: np.ndarray) -> np.ndarray:
                return np.full((240, 320, 3), 255, dtype=np.uint8)

        class _EngineStub:
            def recognize(self, _image: np.ndarray) -> list[OCRResult]:
                return []

        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _ProcessorStub()
        service.camera_processor = _ProcessorStub()
        service.engine = _EngineStub()
        service._camera_digit_pipeline = type(
            "_CameraPipelineStub",
            (),
            {
                "run": lambda _self, _image, allow_fallback=True: (
                    [OCRResult(text="7", score=0.91, box=[[20, 30], [110, 30], [110, 120], [20, 120]])],
                    True,
                ),
            },
        )()

        results, fallback_used = service._recognize_camera_frame_internal(image)

        self.assertTrue(fallback_used)
        self.assertEqual([result.text for result in results], ["7"])

    def test_collect_camera_fast_candidate_boxes_prefers_simple_roi_digits(self) -> None:
        image = np.full((240, 360, 3), 255, dtype=np.uint8)
        cv2.putText(image, "12", (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 0), 12, cv2.LINE_AA)

        boxes = self.camera_pipeline._collect_camera_fast_candidate_boxes(image)

        self.assertGreaterEqual(len(boxes), 1)
        self.assertLessEqual(len(boxes), 2)

    def test_resolve_camera_fast_path_accepts_up_to_six_candidates(self) -> None:
        image = np.full((180, 420, 3), 255, dtype=np.uint8)
        candidate_boxes = [(index * 20, 10, index * 20 + 12, 70) for index in range(6)]

        self.camera_pipeline._collect_camera_fast_candidate_boxes = lambda _image: list(candidate_boxes)
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

        results, fallback_needed = self.camera_pipeline._resolve_camera_fast_path(image)

        self.assertFalse(fallback_needed)
        self.assertEqual([result.text for result in results], ["1", "2", "3", "4", "5", "6"])

    def test_resolve_camera_fast_path_requests_fallback_for_more_than_six_candidates(self) -> None:
        image = np.full((180, 480, 3), 255, dtype=np.uint8)
        candidate_boxes = [(index * 20, 10, index * 20 + 12, 70) for index in range(7)]

        self.camera_pipeline._collect_camera_fast_candidate_boxes = lambda _image: list(candidate_boxes)
        self.camera_pipeline._resolve_camera_fast_candidate = lambda _block: (_ for _ in ()).throw(
            AssertionError("overcrowded camera scenes should request fallback before fast review runs")
        )

        results, fallback_needed = self.camera_pipeline._resolve_camera_fast_path(image)

        self.assertEqual(results, [])
        self.assertTrue(fallback_needed)

    def test_resolve_camera_fast_candidate_supports_six_way_split(self) -> None:
        block = ImageCandidateBlock(
            display_box=(0, 0, 240, 40),
            region_image=np.full((40, 240, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((40, 240), 255, dtype=np.uint8),
        )
        child_blocks = [
            ImageCandidateBlock(
                display_box=(index * 40, 0, (index + 1) * 40, 40),
                region_image=np.full((40, 40, 3), 255, dtype=np.uint8),
                foreground_mask=np.full((40, 40), 255, dtype=np.uint8),
            )
            for index in range(6)
        ]
        observed_max_segments: list[int | None] = []

        def estimate_segment_count(_binary: np.ndarray, *, preferred_count: int, aspect_ratio: float, max_segments: int | None = None) -> int:
            self.assertEqual(preferred_count, 0)
            self.assertGreater(aspect_ratio, 1.0)
            observed_max_segments.append(max_segments)
            return 6

        self.camera_pipeline._estimate_projection_segment_count = estimate_segment_count
        self.camera_pipeline._split_image_candidate_block = lambda _block, *, char_hint: child_blocks if char_hint == 6 else []
        self.camera_pipeline._review_image_candidate_block = lambda child_block: ImageReviewResult(
            text=str((child_block.display_box[0] // 40) + 1),
            score=0.95,
            support=1,
            attempts=1,
        )
        self.camera_pipeline._build_image_result_from_review = lambda child_block, _candidate, review: OCRResult(
            text=review.text,
            score=review.score,
            box=self.camera_pipeline._region_box_to_polygon(child_block.display_box),
        )

        results, fallback_needed = self.camera_pipeline._resolve_camera_fast_candidate(block)

        self.assertFalse(fallback_needed)
        self.assertEqual(observed_max_segments, [6])
        self.assertEqual([result.text for result in results], ["1", "2", "3", "4", "5", "6"])

    def test_extract_image_candidate_boxes_supports_dark_text(self) -> None:
        image = np.full((240, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "12", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0, 0, 0), 12, cv2.LINE_AA)

        boxes = self.image_pipeline._extract_image_candidate_boxes(image)

        self.assertEqual(len(boxes), 2)
        self.assertLess(boxes[0][0], boxes[1][0])

    def test_extract_image_candidate_boxes_supports_light_text(self) -> None:
        image = np.full((240, 420, 3), 25, dtype=np.uint8)
        cv2.putText(image, "34", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (255, 255, 255), 12, cv2.LINE_AA)

        boxes = self.image_pipeline._extract_image_candidate_boxes(image)

        self.assertEqual(len(boxes), 2)
        self.assertLess(boxes[0][0], boxes[1][0])

    def test_wide_candidate_prefers_projection_split(self) -> None:
        image = np.full((220, 360, 3), 255, dtype=np.uint8)
        cv2.putText(image, "12", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 0), 10, cv2.LINE_AA)
        block = self.image_pipeline._build_image_candidate_block(image, (0, 0, image.shape[1], image.shape[0]))
        self.assertIsNotNone(block)

        candidate = ImageCandidate(
            display_box=block.display_box,
            text_hint="12",
            score_hint=0.99,
            sources={"ocr"},
        )
        review = ImageReviewResult(text="1", score=0.40, support=1, attempts=3)

        split_count = self.image_pipeline._determine_image_split_count(block, candidate, review)

        self.assertGreaterEqual(split_count, 2)

    def test_dedupe_image_candidates_merges_equivalent_boxes(self) -> None:
        candidates = [
            ImageCandidate(display_box=(10, 10, 50, 80), sources={"segment"}),
            ImageCandidate(display_box=(12, 12, 49, 79), text_hint="7", score_hint=0.88, sources={"ocr"}),
            ImageCandidate(display_box=(90, 10, 130, 80), sources={"segment"}),
        ]

        deduped = self.image_pipeline._dedupe_image_candidates(candidates)

        self.assertEqual(len(deduped), 2)
        merged = next(candidate for candidate in deduped if "ocr" in candidate.sources)
        self.assertIn("segment", merged.sources)
        self.assertEqual(merged.text_hint, "7")

    def test_should_use_structured_photo_mode_prefers_sequences_or_long_rows(self) -> None:
        ocr_results = [
            OCRResult(
                text="123",
                score=0.99,
                box=[[0, 0], [180, 0], [180, 60], [0, 60]],
            )
        ]
        structured_rows = [[(10, 10, 60, 110), (80, 10, 130, 110), (150, 10, 200, 110)]]
        scattered_rows = [[(10, 10, 50, 90), (110, 120, 150, 200)]]

        self.assertTrue(self.image_pipeline._should_use_structured_photo_mode(ocr_results, structured_rows))
        self.assertFalse(self.image_pipeline._should_use_structured_photo_mode([], scattered_rows))

    def test_filter_structured_panel_boxes_removes_singleton_noise_rows(self) -> None:
        boxes = [
            (0, 0, 200, 30),
            (40, 80, 90, 180),
            (110, 80, 165, 182),
            (185, 82, 240, 181),
            (260, 81, 315, 184),
            (30, 220, 210, 250),
        ]

        filtered = self.image_pipeline._filter_structured_panel_boxes(boxes)

        self.assertEqual(
            filtered,
            [(40, 80, 90, 180), (110, 80, 165, 182), (185, 82, 240, 181), (260, 81, 315, 184)],
        )

    def test_strong_structured_hint_skips_retry_review_for_stable_middle_child(self) -> None:
        block = ImageCandidateBlock(
            display_box=(20, 0, 44, 40),
            region_image=np.full((40, 24, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((40, 24), 255, dtype=np.uint8),
        )
        review = ImageReviewResult(text="3", score=0.60, support=1, attempts=3)

        should_retry = self.image_pipeline._should_retry_structured_child_review(
            block,
            review=review,
            hint_char="3",
            hint_score=0.97,
            position_index=1,
            total_positions=3,
            split_is_stable=True,
        )

        self.assertFalse(should_retry)

    def test_structured_panel_only_runs_when_sequence_results_leave_gaps(self) -> None:
        existing_results = [OCRResult(text="1", score=0.98, box=[[0, 0], [10, 0], [10, 20], [0, 20]])]

        self.assertTrue(
            self.image_pipeline._should_resolve_structured_panel(
                existing_results=existing_results,
                minimum_expected_count=3,
            )
        )
        self.assertFalse(
            self.image_pipeline._should_resolve_structured_panel(
                existing_results=existing_results * 3,
                minimum_expected_count=3,
            )
        )

    def test_resolve_structured_missing_positions_returns_only_uncovered_boxes(self) -> None:
        panel_boxes = [(0, 0, 20, 40), (25, 0, 45, 40), (50, 0, 70, 40)]
        occupied_boxes = [(0, 0, 20, 40)]

        missing = self.image_pipeline._resolve_structured_missing_positions(panel_boxes, occupied_boxes, limit=2)

        self.assertEqual(missing, [(25, 0, 45, 40), (50, 0, 70, 40)])

    def test_split_structured_sequence_boxes_falls_back_to_even_split_when_projection_is_unstable(self) -> None:
        binary = np.full((40, 90), 255, dtype=np.uint8)
        self.image_pipeline._extract_projection_split_boxes = lambda _binary, segment_count: [(0, 0, 2, 40), (2, 0, 88, 40), (88, 0, 90, 40)]

        split_boxes = self.image_pipeline._split_structured_sequence_boxes(binary, 3)

        self.assertEqual(len(split_boxes), 3)
        self.assertTrue(all(box[2] - box[0] >= 3 for box in split_boxes))

    def test_resolve_structured_panel_candidates_stops_after_filling_expected_gaps(self) -> None:
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        panel_boxes = [(0, 0, 20, 40), (24, 0, 44, 40), (48, 0, 68, 40)]
        existing_results = [OCRResult(text="1", score=0.98, box=[[0, 0], [20, 0], [20, 40], [0, 40]])]
        self.image_pipeline._build_image_candidate_block = lambda _image, box: ImageCandidateBlock(
            display_box=box,
            region_image=np.full((40, 20, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((40, 20), 255, dtype=np.uint8),
        )
        self.image_pipeline._review_image_candidate_block_results = lambda _block: [TextOnlyResult(text="2", score=0.93)]
        self.image_pipeline._should_retry_structured_panel_review = lambda _block, review: False
        self.image_pipeline._build_image_result_from_review = lambda block, candidate, review: OCRResult(
            text="2",
            score=0.95,
            box=[[block.display_box[0], block.display_box[1]], [block.display_box[2], block.display_box[1]], [block.display_box[2], block.display_box[3]], [block.display_box[0], block.display_box[3]]],
        )

        resolved = self.image_pipeline._resolve_structured_panel_candidates(
            image,
            panel_boxes,
            existing_results,
            minimum_expected_count=2,
        )

        self.assertEqual(len(resolved), 1)

    def test_warp_polygon_crop_returns_rectified_crop(self) -> None:
        image = np.full((220, 220, 3), 255, dtype=np.uint8)
        polygon = [[40, 50], [170, 30], [180, 160], [55, 180]]
        cv2.fillConvexPoly(image, np.array(polygon, dtype=np.int32), (30, 30, 30))

        warped, inverse = self.image_pipeline._warp_polygon_crop(image, polygon)

        self.assertIsNotNone(warped)
        self.assertIsNotNone(inverse)
        assert warped is not None
        self.assertGreater(warped.shape[0], 0)
        self.assertGreater(warped.shape[1], 0)

    def test_review_image_candidate_block_with_retry_reuses_base_results(self) -> None:
        service = self.make_service()
        dummy_image = np.full((24, 24, 3), 255, dtype=np.uint8)
        image_pipeline = service._get_image_pipeline()
        dummy_block = image_pipeline._create_image_candidate_block(dummy_image, (0, 0, 24, 24))
        assert dummy_block is not None

        first_variant = np.full((8, 8, 3), 10, dtype=np.uint8)
        second_variant = np.full((8, 8, 3), 20, dtype=np.uint8)
        image_pipeline._build_image_review_variants = lambda _block: [first_variant, second_variant]
        image_pipeline._rotate_image = lambda image, angle: (image.copy(), None, None)
        image_pipeline._create_image_candidate_block = lambda region_image, display_box: ImageCandidateBlock(
            display_box=display_box,
            region_image=region_image,
            foreground_mask=np.full(region_image.shape[:2], 255, dtype=np.uint8),
        )

        class _EngineStub:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def recognize_handwriting_blocks(self, images: list[np.ndarray]) -> list[TextOnlyResult]:
                self.calls.append(len(images))
                if len(self.calls) == 1:
                    return [
                        TextOnlyResult(text="1", score=0.62),
                        TextOnlyResult(text="7", score=0.61),
                    ]
                return [TextOnlyResult(text="1", score=0.97) for _ in images]

        service.engine = _EngineStub()

        review = image_pipeline._review_image_candidate_block_with_retry(dummy_block)

        self.assertEqual(service.engine.calls, [2, 4])
        self.assertEqual(review.text, "1")
        self.assertGreaterEqual(review.score, 0.97)

    def test_resolve_image_candidate_uses_retry_review_for_uncertain_ocr_candidate(self) -> None:
        block = ImageCandidateBlock(
            display_box=(0, 0, 40, 30),
            region_image=np.full((30, 40, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((30, 40), 255, dtype=np.uint8),
        )
        candidate = ImageCandidate(display_box=block.display_box, text_hint="12", score_hint=0.91, sources={"ocr"})
        retry_calls: list[str] = []

        self.image_pipeline._review_image_candidate_block_results = lambda _block: [TextOnlyResult(text="1", score=0.60)]
        self.image_pipeline._review_image_candidate_block_with_retry_from_results = lambda _block, _results: (
            retry_calls.append("retry") or ImageReviewResult(text="2", score=0.98, support=2, attempts=3)
        )
        self.image_pipeline._determine_image_split_count = lambda _block, _candidate, _review: 1
        self.image_pipeline._build_image_result_from_review = lambda _block, _candidate, review: OCRResult(
            text=review.text,
            score=review.score,
            box=[[0, 0], [10, 0], [10, 20], [0, 20]],
        )

        results = self.image_pipeline._resolve_image_candidate(block, candidate=candidate, warnings=[], depth=0)

        self.assertEqual(retry_calls, ["retry"])
        self.assertEqual([result.text for result in results], ["2"])

    def test_resolve_image_digit_results_merges_generic_and_fallback_results(self) -> None:
        generic_result = OCRResult(text="1", score=0.94, box=[[0, 0], [10, 0], [10, 20], [0, 20]])
        fallback_result = OCRResult(text="2", score=0.91, box=[[20, 0], [30, 0], [30, 20], [20, 20]])
        image = np.full((40, 60, 3), 255, dtype=np.uint8)

        self.image_pipeline._resolve_structured_photo_results = lambda _image, _results: ([], [])
        self.image_pipeline._collect_image_candidates = lambda _image, _results: [ImageCandidate(display_box=(0, 0, 10, 20), sources={"segment"})]
        self.image_pipeline._resolve_image_candidates = lambda _image, _candidates, _warnings: [generic_result]
        self.image_pipeline._resolve_image_results_with_ocr_fallback = lambda _image, _results, _warnings: [fallback_result]

        results, warnings = self.image_pipeline._resolve_image_digit_results(image, [])

        self.assertEqual([result.text for result in results], ["1", "2"])
        self.assertEqual(warnings, [])

    def test_resolve_image_candidates_does_not_skip_multichar_ocr_hint_when_covered(self) -> None:
        image = np.full((40, 60, 3), 255, dtype=np.uint8)
        candidate = ImageCandidate(display_box=(0, 0, 30, 20), text_hint="12", score_hint=0.95, sources={"ocr"})
        self.image_pipeline._is_box_covered_by_results = lambda _box, _results: True
        self.image_pipeline._build_image_candidate_block = lambda _image, _box: ImageCandidateBlock(
            display_box=(0, 0, 30, 20),
            region_image=np.full((20, 30, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((20, 30), 255, dtype=np.uint8),
        )
        self.image_pipeline._resolve_image_candidate = lambda _block, candidate, warnings, depth: [
            OCRResult(text="1", score=0.9, box=[[0, 0], [10, 0], [10, 20], [0, 20]])
        ]

        results = self.image_pipeline._resolve_image_candidates(image, [candidate], warnings=[])

        self.assertEqual([result.text for result in results], ["1"])

    def test_split_image_multi_digit_result_keeps_high_confidence_hint_when_child_review_fails(self) -> None:
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        result = OCRResult(text="12", score=0.98, box=[[0, 0], [40, 0], [40, 20], [0, 20]])
        parent_block = ImageCandidateBlock(
            display_box=(0, 0, 40, 20),
            region_image=np.full((20, 40, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((20, 40), 255, dtype=np.uint8),
        )
        child_one = ImageCandidateBlock(
            display_box=(0, 0, 18, 20),
            region_image=np.full((20, 18, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((20, 18), 255, dtype=np.uint8),
        )
        child_two = ImageCandidateBlock(
            display_box=(20, 0, 38, 20),
            region_image=np.full((20, 18, 3), 255, dtype=np.uint8),
            foreground_mask=np.full((20, 18), 255, dtype=np.uint8),
        )

        self.image_pipeline._build_image_candidate_block = lambda _image, _box: parent_block
        self.image_pipeline._split_image_candidate_block = lambda _block, char_hint: [child_one, child_two]

        def resolve_child(block, *, candidate, warnings, depth):
            if block.display_box == child_one.display_box:
                return [OCRResult(text="1", score=0.93, box=[[0, 0], [18, 0], [18, 20], [0, 20]])]
            return []

        self.image_pipeline._resolve_image_candidate = resolve_child

        split_results = self.image_pipeline._split_image_multi_digit_result(image, result, warnings=[])

        self.assertEqual([item.text for item in split_results], ["1", "2"])

    def test_recognize_image_skips_full_rotation_retry_when_segmentation_already_has_multiple_candidates(self) -> None:
        class _ProcessorStub:
            def enhance(self, image: np.ndarray) -> np.ndarray:
                return image.copy()

        class _EngineStub:
            def __init__(self) -> None:
                self.recognize_calls = 0

            def recognize(self, _image: np.ndarray) -> list[OCRResult]:
                self.recognize_calls += 1
                return [OCRResult(text="6", score=0.45, box=[[0, 0], [40, 0], [40, 40], [0, 40]])]

            def draw_results(self, image: np.ndarray, _results: list[OCRResult]) -> np.ndarray:
                return image.copy()

        service = DigitOCRService.__new__(DigitOCRService)
        service.processor = _ProcessorStub()
        service.camera_processor = _ProcessorStub()
        service.engine = _EngineStub()
        image_pipeline = service._get_image_pipeline()
        image_pipeline._extract_image_candidate_boxes = lambda _image: [(0, 0, 20, 20), (30, 0, 50, 20)]
        image_pipeline._retry_rotated_image_results = lambda _image: (_ for _ in ()).throw(
            AssertionError("full-image rotation retry should be skipped when segmentation already has multiple candidates")
        )
        image_pipeline._resolve_image_digit_results = lambda _image, results: (results, [])

        output = service.recognize_image(np.full((60, 60, 3), 255, dtype=np.uint8), source_name="segmented-image")

        self.assertEqual([result.text for result in output.results], ["6"])
        self.assertEqual(service.engine.recognize_calls, 1)


class ImageModeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.service = DigitOCRService(dict_path=PROJECT_ROOT / "config" / "digits_dict.txt")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"PaddleOCR runtime unavailable: {exc}") from exc

    @staticmethod
    def _load_image(file_name: str) -> np.ndarray:
        image_path = PROJECT_ROOT / "data" / "input" / file_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise AssertionError(f"Failed to load sample image: {image_path}")
        return image

    @staticmethod
    def _make_centered_text_image(
        text: str,
        *,
        image_size: tuple[int, int],
        font_scale: float,
        thickness: int,
    ) -> np.ndarray:
        height, width = image_size
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        origin_x = max(8, (width - text_size[0]) // 2)
        origin_y = max(text_size[1] + 8, (height + text_size[1]) // 2)
        cv2.putText(
            image,
            text,
            (origin_x, min(height - baseline - 8, origin_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        return image

    def test_printed_sequence_image_returns_single_digits(self) -> None:
        image = self._load_image("5b42782bead04a498a8ce1fa43b60f56.jpeg~tplv-a9rns2rl98-downsize_watermark_1_5_b.png")

        output = self.service.recognize_image(image, source_name="printed-sequence")

        self.assertEqual([result.text for result in output.results], list("12345"))
        self.assertEqual(output.warnings, [])

    def test_camera_path_returns_single_digits_for_centered_sequence_image(self) -> None:
        image = self._make_centered_text_image(
            "12345",
            image_size=(220, 520),
            font_scale=4.0,
            thickness=10,
        )

        results = self.service.recognize_camera_frame(image, source_name="camera-centered-sequence")

        self.assertEqual([result.text for result in results], list("12345"))

    def test_scattered_handwriting_image_returns_four_single_digit_boxes(self) -> None:
        image = self._load_image("3132.png")

        output = self.service.recognize_image(image, source_name="scattered-handwriting")

        self.assertEqual(len(output.results), 4)
        self.assertTrue(all(len(result.text) == 1 for result in output.results))

    def test_camera_path_returns_single_digits_for_centered_pair_image(self) -> None:
        image = self._make_centered_text_image(
            "12",
            image_size=(240, 360),
            font_scale=5.0,
            thickness=12,
        )

        results = self.service.recognize_camera_frame(image, source_name="camera-centered-pair")

        self.assertEqual([result.text for result in results], ["1", "2"])

    def test_structured_photo_twenty_digit_grid_returns_full_sequence(self) -> None:
        image = self._load_image("115494f2894e4b0b8eed550ab24b37ca.jpeg~tplv-a9rns2rl98-downsize_watermark_1_5_b.png")

        output = self.service.recognize_image(image, source_name="structured-grid-20")

        self.assertEqual([result.text for result in output.results], list("0123456789" * 2))
        self.assertEqual(output.warnings, [])

    def test_structured_photo_blackboard_returns_clean_three_digits(self) -> None:
        image = self._load_image("03b6d396f4db43af9f15839228bbb3f3.jpeg~tplv-a9rns2rl98-downsize_watermark_1_5_b.png")

        output = self.service.recognize_image(image, source_name="structured-blackboard")

        self.assertEqual([result.text for result in output.results], list("123"))
        self.assertEqual(output.warnings, [])

    def test_structured_photo_nine_digit_grid_returns_complete_sequence(self) -> None:
        image = self._load_image("2cdf2f02d7f645ac95051ae0eaca9675.jpeg~tplv-a9rns2rl98-downsize_watermark_1_5_b.png")

        output = self.service.recognize_image(image, source_name="structured-grid-9")

        self.assertEqual([result.text for result in output.results], list("012345789"))
        self.assertEqual(output.warnings, [])


if __name__ == "__main__":
    unittest.main()
