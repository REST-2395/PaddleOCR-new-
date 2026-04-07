"""Candidate review helpers for image OCR."""

from __future__ import annotations

import cv2
import numpy as np

from ..ocr_engine import OCRResult, TextOnlyResult
from ..service_types import ImageCandidate, ImageCandidateBlock, ImageReviewResult, RegionBox


class ImageCandidateReviewMixin:
    """Build candidate blocks and review them with OCR variants."""

    def _review_image_candidate_block_with_retry(self, block: ImageCandidateBlock) -> ImageReviewResult:
        base_results = self._review_image_candidate_block_results(block)
        base_review = self._aggregate_image_review_results(base_results)
        if self._is_reliable_image_review(base_review):
            return base_review
        retry_images: list[np.ndarray] = []
        for angle in (-12.0, 12.0):
            rotated_image, _, _ = self._rotate_image(block.region_image, angle)
            rotated_block = self._create_image_candidate_block(
                rotated_image,
                (0, 0, rotated_image.shape[1], rotated_image.shape[0]),
            )
            if rotated_block is not None:
                retry_images.extend(self._build_image_review_variants(rotated_block))
        if not retry_images:
            return base_review
        retry_results = self.engine.recognize_handwriting_blocks(retry_images)
        return self._aggregate_image_review_results([*base_results, *retry_results])

    def _extract_image_candidate_boxes(self, image: np.ndarray) -> list[RegionBox]:
        gray = self._build_image_segmentation_gray(image)
        candidate_boxes: list[RegionBox] = []
        for invert in (False, True):
            mask = self._build_image_polarity_mask(gray, invert=invert)
            mask = self._filter_image_mask_components(mask)
            candidate_boxes.extend(self._extract_image_component_boxes(mask, image_shape=image.shape))
        return self._dedupe_region_boxes(self._sort_region_boxes(candidate_boxes))

    def _build_image_candidate_block(
        self,
        image: np.ndarray,
        region_box: RegionBox,
    ) -> ImageCandidateBlock | None:
        padding = max(3, int(round(min(self._box_width(region_box), self._box_height(region_box)) * 0.06)))
        expanded_box = self._expand_region_box(region_box, image.shape, padding=padding)
        x0, y0, x1, y1 = expanded_box
        region = image[y0:y1, x0:x1]
        return self._create_image_candidate_block(region, expanded_box)

    def _create_image_candidate_block(
        self,
        region_image: np.ndarray,
        display_box: RegionBox,
    ) -> ImageCandidateBlock | None:
        if region_image is None or region_image.size == 0:
            return None
        region = self._ensure_bgr(region_image)
        foreground_mask = self._build_image_candidate_mask(region)
        points = cv2.findNonZero(foreground_mask)
        if points is None:
            return None
        x, y, width, height = cv2.boundingRect(points)
        tight_region = region[y : y + height, x : x + width]
        tight_mask = foreground_mask[y : y + height, x : x + width]
        tight_box = (
            display_box[0] + x,
            display_box[1] + y,
            display_box[0] + x + width,
            display_box[1] + y + height,
        )
        return ImageCandidateBlock(
            display_box=tight_box,
            region_image=tight_region,
            foreground_mask=tight_mask,
        )

    def _build_image_candidate_mask(self, region_image: np.ndarray) -> np.ndarray:
        gray = self._build_image_segmentation_gray(region_image)
        candidate_masks = [
            self._filter_image_mask_components(self._build_image_polarity_mask(gray, invert=False)),
            self._filter_image_mask_components(self._build_image_polarity_mask(gray, invert=True)),
        ]
        return max(candidate_masks, key=self._image_mask_quality)

    def _review_image_candidate_block(self, block: ImageCandidateBlock) -> ImageReviewResult:
        return self._aggregate_image_review_results(self._review_image_candidate_block_results(block))

    def _review_image_candidate_block_results(self, block: ImageCandidateBlock) -> list[TextOnlyResult]:
        variant_images = self._build_image_review_variants(block)
        if not variant_images:
            return []
        return self.engine.recognize_handwriting_blocks(variant_images)

    def _review_image_candidate_block_with_retry_from_results(
        self,
        block: ImageCandidateBlock,
        base_results: list[TextOnlyResult],
    ) -> ImageReviewResult:
        base_review = self._aggregate_image_review_results(base_results)
        if self._is_reliable_image_review(base_review):
            return base_review
        retry_images: list[np.ndarray] = []
        for angle in (-12.0, 12.0):
            rotated_image, _, _ = self._rotate_image(block.region_image, angle)
            rotated_block = self._create_image_candidate_block(
                rotated_image,
                (0, 0, rotated_image.shape[1], rotated_image.shape[0]),
            )
            if rotated_block is not None:
                retry_images.extend(self._build_image_review_variants(rotated_block))
        if not retry_images:
            return base_review
        retry_results = self.engine.recognize_handwriting_blocks(retry_images)
        return self._aggregate_image_review_results([*base_results, *retry_results])

    def _build_image_review_variants(self, block: ImageCandidateBlock) -> list[np.ndarray]:
        variants = [
            self._render_image_candidate_variant(block.region_image, block.foreground_mask, mode="gray"),
            self._render_image_candidate_variant(block.region_image, block.foreground_mask, mode="binary"),
            self._render_image_candidate_variant(block.region_image, block.foreground_mask, mode="enhanced"),
        ]
        unique_variants: list[np.ndarray] = []
        for variant in variants:
            if not any(np.array_equal(variant, existing) for existing in unique_variants):
                unique_variants.append(variant)
        return unique_variants

    def _render_image_candidate_variant(
        self,
        region_image: np.ndarray,
        foreground_mask: np.ndarray,
        *,
        mode: str,
    ) -> np.ndarray:
        points = cv2.findNonZero(foreground_mask)
        if points is None:
            return self._resize_to_min_side(self._ensure_bgr(region_image), min_short_side=self.handwriting_candidate_min_side)
        gray = self._select_image_foreground_gray(region_image, foreground_mask)
        x, y, width, height = cv2.boundingRect(points)
        crop_gray = gray[y : y + height, x : x + width]
        crop_mask = foreground_mask[y : y + height, x : x + width] > 0
        rendered = np.full((height, width), 255, dtype=np.uint8)
        if mode == "binary":
            rendered[crop_mask] = 0
        else:
            rendered[crop_mask] = crop_gray[crop_mask]
            if mode == "enhanced":
                rendered = self._enhance_review_variant(rendered, crop_mask)
        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)
        return self._normalize_handwriting_region(rendered_bgr, min_side=self.handwriting_candidate_min_side)

    def _select_image_foreground_gray(self, region_image: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(self._ensure_bgr(region_image), cv2.COLOR_BGR2GRAY)
        mask = foreground_mask > 0
        if not np.any(mask):
            return gray
        foreground_mean = float(np.mean(gray[mask]))
        background_pixels = gray[~mask]
        background_mean = float(np.mean(background_pixels)) if background_pixels.size else 255.0
        return 255 - gray if foreground_mean > background_mean else gray

    def _enhance_review_variant(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        enhanced = cv2.filter2D(enhanced, ddepth=-1, kernel=kernel)
        enhanced[~mask] = 255
        return enhanced

    def _aggregate_image_review_results(self, results: list[TextOnlyResult]) -> ImageReviewResult:
        score_by_text: dict[str, list[float]] = {}
        for result in results:
            if self._is_single_digit_result(result):
                score_by_text.setdefault(result.text, []).append(float(result.score))
        if not score_by_text:
            return ImageReviewResult(text="", score=0.0, support=0, attempts=len(results))
        ranked = sorted(
            score_by_text.items(),
            key=lambda item: (
                len(item[1]),
                max(item[1]),
                sum(item[1]) / float(max(1, len(item[1]))),
                item[0],
            ),
            reverse=True,
        )
        best_text, best_scores = ranked[0]
        best_support = len(best_scores)
        best_score = max(max(best_scores), sum(best_scores) / float(max(1, best_support)))
        runner_up_support = 0
        runner_up_score = 0.0
        if len(ranked) > 1:
            _, runner_up_scores = ranked[1]
            runner_up_support = len(runner_up_scores)
            runner_up_score = max(runner_up_scores)
        return ImageReviewResult(
            text=best_text,
            score=float(best_score),
            support=best_support,
            attempts=len(results),
            runner_up_support=runner_up_support,
            runner_up_score=float(runner_up_score),
        )

    def _build_image_result_from_review(
        self,
        block: ImageCandidateBlock,
        candidate: ImageCandidate,
        review: ImageReviewResult,
    ) -> OCRResult | None:
        width = self._box_width(block.display_box)
        height = self._box_height(block.display_box)
        if width < 12 or height < 18:
            return None
        if self._is_reliable_image_review(review):
            if width < 16 and review.score < self.image_candidate_strong_review_score:
                return None
            return OCRResult(
                text=review.text,
                score=review.score,
                box=self._region_box_to_polygon(block.display_box),
            )
        if len(candidate.text_hint) == 1 and candidate.score_hint >= self.image_candidate_strong_review_score:
            return OCRResult(
                text=candidate.text_hint,
                score=float(candidate.score_hint),
                box=self._region_box_to_polygon(block.display_box),
            )
        return None

    def _is_reliable_image_review(self, review: ImageReviewResult) -> bool:
        if not review.text:
            return False
        if review.support >= 2:
            if review.runner_up_support == review.support and review.score - review.runner_up_score < 0.08:
                return False
            return review.score >= self.image_candidate_review_score
        if review.runner_up_support == 0:
            return review.score >= self.image_candidate_strong_review_score
        return (
            review.score >= self.image_candidate_strong_review_score
            and review.score - review.runner_up_score >= 0.10
        )

    def _determine_image_split_count(
        self,
        block: ImageCandidateBlock,
        candidate: ImageCandidate,
        review: ImageReviewResult,
    ) -> int:
        width = self._box_width(block.display_box)
        height = self._box_height(block.display_box)
        aspect_ratio = width / float(max(1, height))
        hint_count = len(candidate.text_hint) if len(candidate.text_hint) > 1 else 0
        should_probe = (
            hint_count > 1
            or aspect_ratio >= 1.15
            or not self._is_reliable_image_review(review)
            or review.score < self.image_candidate_split_score
        )
        if not should_probe:
            return 1
        return self._estimate_projection_segment_count(
            block.foreground_mask,
            preferred_count=hint_count,
            aspect_ratio=aspect_ratio,
        )

    def _should_retry_image_candidate_review(
        self,
        block: ImageCandidateBlock,
        candidate: ImageCandidate,
        review: ImageReviewResult,
    ) -> bool:
        aspect_ratio = self._box_width(block.display_box) / float(max(1, self._box_height(block.display_box)))
        has_multi_char_hint = len(candidate.text_hint) > 1
        is_fallback_candidate = "fallback" in candidate.sources
        is_uncertain = (not self._is_reliable_image_review(review)) or review.score < self.image_candidate_split_score
        return has_multi_char_hint or (is_fallback_candidate and is_uncertain) or (aspect_ratio >= 1.35 and is_uncertain)
