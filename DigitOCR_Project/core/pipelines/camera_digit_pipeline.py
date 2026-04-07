"""Camera digit OCR pipeline extracted from the service facade."""

from __future__ import annotations

from camera import config as camera_settings
import numpy as np

from ..ocr_engine import OCRResult
from .handwriting_blocks import HandwritingBlockBuilderMixin
from .handwriting_scoring import HandwritingScoringMixin
from .handwriting_segmentation import HandwritingSegmentationMixin
from .image_candidate_support import ImageCandidateSupportMixin
from .image_structured import StructuredImageMixin


class CameraDigitPipeline(
    StructuredImageMixin,
    ImageCandidateSupportMixin,
    HandwritingScoringMixin,
    HandwritingSegmentationMixin,
    HandwritingBlockBuilderMixin,
):
    """Digit-mode camera OCR pipeline that delegates shared utilities to the service."""

    def __init__(self, service) -> None:
        self.service = service

    def __getattr__(self, name: str):
        return getattr(self.service, name)

    @property
    def engine(self):
        return self.service.engine

    def run(
        self,
        image: np.ndarray,
        *,
        allow_fallback: bool = True,
    ) -> tuple[list[OCRResult], bool]:
        fast_results, should_fallback = self._resolve_camera_fast_path(image)
        selected_results = fast_results
        fallback_used = False
        if allow_fallback and should_fallback:
            fallback_results = self.run_fallback(image)
            fallback_used = True
            if fallback_results:
                selected_results = fallback_results
        return selected_results, fallback_used

    def run_fallback(self, image: np.ndarray) -> list[OCRResult]:
        return self._resolve_camera_fallback_path(image)

    def _resolve_camera_fast_path(
        self,
        image: np.ndarray,
    ) -> tuple[list[OCRResult], bool]:
        """Resolve camera OCR with segmentation-first single-digit review."""
        candidate_boxes = self._collect_camera_fast_candidate_boxes(image)
        if not candidate_boxes:
            return [], True

        if len(candidate_boxes) > camera_settings.CAMERA_FAST_MAX_CANDIDATES:
            return [], True

        resolved_results: list[OCRResult] = []
        fallback_needed = False
        for candidate_box in candidate_boxes:
            block = self._build_image_candidate_block(image, candidate_box)
            if block is None:
                fallback_needed = True
                continue

            block_results, block_fallback = self._resolve_camera_fast_candidate(block)
            fallback_needed = fallback_needed or block_fallback
            for result in block_results:
                if not self._is_duplicate_image_result(result, resolved_results):
                    resolved_results.append(result)

        resolved_results = self._dedupe_final_image_results(resolved_results)
        if not resolved_results:
            return [], True

        average_score = sum(float(item.score) for item in resolved_results) / float(len(resolved_results))
        if average_score < camera_settings.CAMERA_FAST_MIN_REVIEW_SCORE:
            fallback_needed = True
        return resolved_results, fallback_needed

    def _collect_camera_fast_candidate_boxes(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Collect the small set of plausible camera ROI digit boxes."""
        image_height, image_width = image.shape[:2]
        min_area = max(24, int(round(image_height * image_width * camera_settings.CAMERA_FAST_MIN_BOX_AREA_RATIO)))
        min_height = max(16, int(round(image_height * camera_settings.CAMERA_FAST_MIN_BOX_HEIGHT_RATIO)))
        center_x = image_width / 2.0
        center_y = image_height / 2.0

        filtered_boxes = [
            box
            for box in self._extract_image_candidate_boxes(image)
            if self._box_area(box) >= min_area and self._box_height(box) >= min_height
        ]
        return sorted(
            filtered_boxes,
            key=lambda box: (
                -self._box_area(box),
                abs(self._box_center(box)[0] - center_x) + abs(self._box_center(box)[1] - center_y),
                box[1],
                box[0],
            ),
        )

    def _resolve_camera_fast_candidate(
        self,
        block,
    ) -> tuple[list[OCRResult], bool]:
        """Resolve one fast-path camera candidate with optional local projection splitting."""
        service_types = self._service_types()
        candidate = service_types.ImageCandidate(display_box=block.display_box, sources={"camera_fast"})
        width = self._box_width(block.display_box)
        height = self._box_height(block.display_box)
        aspect_ratio = width / float(max(1, height))
        split_count = self._estimate_projection_segment_count(
            block.foreground_mask,
            preferred_count=0,
            aspect_ratio=aspect_ratio,
            max_segments=camera_settings.CAMERA_FAST_MAX_CANDIDATES,
        )

        if split_count > camera_settings.CAMERA_FAST_MAX_CANDIDATES:
            return [], True

        if split_count > 1:
            child_blocks = self._split_image_candidate_block(block, char_hint=split_count)
            if len(child_blocks) != split_count:
                return [], True

            resolved_results: list[OCRResult] = []
            fallback_needed = False
            for child_block in child_blocks:
                child_candidate = service_types.ImageCandidate(display_box=child_block.display_box, sources={"camera_fast_split"})
                review = self._review_image_candidate_block(child_block)
                result = self._build_image_result_from_review(child_block, child_candidate, review)
                if result is None or float(result.score) < camera_settings.CAMERA_FAST_MIN_REVIEW_SCORE:
                    fallback_needed = True
                    continue
                resolved_results.append(result)

            if not resolved_results:
                return [], True
            return resolved_results, fallback_needed

        review = self._review_image_candidate_block(block)
        result = self._build_image_result_from_review(block, candidate, review)
        if result is None or float(result.score) < camera_settings.CAMERA_FAST_MIN_REVIEW_SCORE:
            return [], True
        return [result], False

    def _resolve_camera_fallback_path(
        self,
        image: np.ndarray,
    ) -> list[OCRResult]:
        """Run the limited heavy OCR fallback for complex camera ROI frames."""
        recognized_results = self.engine.recognize(image)
        recognized_results = self._sort_results(recognized_results)

        structured_results, _ = self._resolve_structured_photo_results(image, recognized_results)
        if structured_results:
            return self._dedupe_final_image_results(structured_results)

        warnings: list[str] = []
        resolved_results = self._resolve_image_results_with_ocr_fallback(
            image,
            recognized_results,
            warnings,
        )
        return self._dedupe_final_image_results(resolved_results)

    @staticmethod
    def _service_types():
        from .. import service_types

        return service_types


__all__ = ["CameraDigitPipeline"]
