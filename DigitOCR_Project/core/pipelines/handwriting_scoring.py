"""Scoring and normalization helpers for the handwriting pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from ..ocr_engine import TextOnlyResult


class HandwritingScoringMixin:
    """Score handwriting OCR results and prepare fallback images."""

    def _should_retry_handwriting_result(self, result: TextOnlyResult) -> bool:
        if not self._is_usable_handwriting_result(result):
            return True
        return result.score < self.handwriting_fallback_score

    def _should_retry_rotated_handwriting_result(self, result: TextOnlyResult) -> bool:
        if not self._is_single_digit_result(result):
            return True
        return result.score < self.handwriting_fallback_score

    @staticmethod
    def _is_better_handwriting_result(candidate: TextOnlyResult, current: TextOnlyResult) -> bool:
        return HandwritingScoringMixin._handwriting_result_quality(candidate) > HandwritingScoringMixin._handwriting_result_quality(
            current
        )

    @staticmethod
    def _handwriting_result_quality(result: TextOnlyResult) -> tuple[bool, bool, float]:
        return HandwritingScoringMixin._is_single_digit_result(result), bool(result.text), float(result.score)

    @staticmethod
    def _is_usable_handwriting_result(result: TextOnlyResult) -> bool:
        return bool(result.text)

    @staticmethod
    def _is_single_digit_result(result: TextOnlyResult) -> bool:
        return len(result.text) == 1

    def _normalize_handwriting_region(self, image: np.ndarray, *, min_side: int) -> np.ndarray:
        region = self._ensure_bgr(image)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mask = gray < self.handwriting_threshold

        if not np.any(mask):
            return self._resize_to_min_side(region, min_short_side=min_side)

        points = cv2.findNonZero(mask.astype(np.uint8))
        x, y, width, height = cv2.boundingRect(points)
        cropped = region[y : y + height, x : x + width]

        padding = max(18, int(0.28 * max(width, height)))
        side = max(width, height) + padding * 2
        side = max(side, min_side)

        normalized = np.full((side, side, 3), 255, dtype=np.uint8)
        offset_x = (side - width) // 2
        offset_y = (side - height) // 2
        normalized[offset_y : offset_y + height, offset_x : offset_x + width] = cropped
        return normalized

    def _build_rotation_preview_image(self, image: np.ndarray) -> np.ndarray:
        region = self._ensure_bgr(image)
        mask = self._build_foreground_mask(region)
        points = cv2.findNonZero(mask)
        if points is not None:
            x, y, width, height = cv2.boundingRect(points)
            region = region[y : y + height, x : x + width]

        padding = max(12, int(0.18 * max(region.shape[:2])))
        padded_region = self._pad_image_border(region, padding)
        return self._normalize_handwriting_region(padded_region, min_side=self.handwriting_candidate_min_side)
