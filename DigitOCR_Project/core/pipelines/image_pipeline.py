"""Image-mode OCR pipeline extracted from the service facade."""

from __future__ import annotations

import numpy as np

from ..ocr_engine import OCRResult
from .handwriting_blocks import HandwritingBlockBuilderMixin
from .handwriting_scoring import HandwritingScoringMixin
from .handwriting_segmentation import HandwritingSegmentationMixin
from .image_candidate_support import ImageCandidateSupportMixin
from .image_structured import StructuredImageMixin


class ImageRecognitionPipeline(
    StructuredImageMixin,
    ImageCandidateSupportMixin,
    HandwritingScoringMixin,
    HandwritingSegmentationMixin,
    HandwritingBlockBuilderMixin,
):
    """Image-mode OCR pipeline backed by service-shared helpers and models."""

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
        results: list[OCRResult],
    ) -> tuple[list[OCRResult], list[str]]:
        best_results = results
        if self._should_retry_image_results(results) and self._should_retry_full_image_rotation(image, results):
            rotated_results = self._retry_rotated_image_results(image)
            if self._is_better_image_result_set(rotated_results, results):
                best_results = rotated_results
        return self._resolve_image_digit_results(image, best_results)

    def _should_retry_full_image_rotation(
        self,
        image: np.ndarray,
        results: list[OCRResult],
    ) -> bool:
        del results
        candidate_boxes = self._extract_image_candidate_boxes(image)
        return len(candidate_boxes) < 2

    def _retry_rotated_image_results(self, image: np.ndarray) -> list[OCRResult]:
        candidate_infos: list[tuple[np.ndarray, np.ndarray]] = []
        preview_images: list[np.ndarray] = []
        for angle in self._build_rotation_retry_angles(self._estimate_foreground_angle(image)):
            rotated_image, _, inverse_matrix = self._rotate_image(image, angle)
            candidate_infos.append((rotated_image, inverse_matrix))
            preview_images.append(self._build_rotation_preview_image(rotated_image))
        if not candidate_infos:
            return []
        preview_results = self.engine.recognize_handwriting_blocks(preview_images)
        best_index = 0
        best_quality = self._text_result_quality(preview_results[0]) if preview_results else (False, 0.0, 0)
        for index, preview_result in enumerate(preview_results[1:], start=1):
            quality = self._text_result_quality(preview_result)
            if quality > best_quality:
                best_index = index
                best_quality = quality
        rotated_image, inverse_matrix = candidate_infos[best_index]
        rotated_results = self.engine.recognize(rotated_image)
        return self._map_results_with_affine(rotated_results, inverse_matrix, to_shape=image.shape)

    def _resolve_image_digit_results(
        self,
        image: np.ndarray,
        results: list[OCRResult],
    ) -> tuple[list[OCRResult], list[str]]:
        structured_results, structured_warnings = self._resolve_structured_photo_results(image, results)
        if structured_results:
            return self._sort_results(structured_results), structured_warnings
        warnings: list[str] = []
        candidates = self._collect_image_candidates(image, results)
        generic_warnings: list[str] = []
        resolved_results = self._resolve_image_candidates(image, candidates, generic_warnings) if candidates else []
        fallback_warnings: list[str] = []
        fallback_results = self._resolve_image_results_with_ocr_fallback(image, results, fallback_warnings)
        merged_results = self._merge_image_result_sets(resolved_results, fallback_results)
        warnings.extend(generic_warnings)
        if fallback_results or not resolved_results:
            warnings.extend(fallback_warnings)
        return merged_results, warnings


__all__ = ["ImageRecognitionPipeline"]
