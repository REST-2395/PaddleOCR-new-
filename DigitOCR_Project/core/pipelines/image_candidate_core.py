"""Core candidate collection helpers for image OCR."""

from __future__ import annotations

import numpy as np

from ..ocr_engine import OCRResult
from ..service_types import ImageCandidate, ImageCandidateBlock


class ImageCandidateCollectionMixin:
    """Collect candidates and resolve them into OCR results."""

    def _collect_image_candidates(
        self,
        image: np.ndarray,
        ocr_results: list[OCRResult],
    ) -> list[ImageCandidate]:
        candidates = [ImageCandidate(display_box=box, sources={"segment"}) for box in self._extract_image_candidate_boxes(image)]
        for result in ocr_results:
            region_box = self._polygon_to_region_box(result.box)
            padding = max(3, int(round(min(self._box_width(region_box), self._box_height(region_box)) * 0.08)))
            candidates.append(
                ImageCandidate(
                    display_box=self._expand_region_box(region_box, image.shape, padding=padding),
                    text_hint=result.text,
                    score_hint=float(result.score),
                    sources={"ocr"},
                )
            )
        return self._dedupe_image_candidates(candidates)

    def _resolve_image_candidates(
        self,
        image: np.ndarray,
        candidates: list[ImageCandidate],
        warnings: list[str],
    ) -> list[OCRResult]:
        resolved_results: list[OCRResult] = []
        for candidate in sorted(candidates, key=self._image_candidate_sort_key):
            if (len(candidate.text_hint) <= 1 or "ocr" not in candidate.sources) and self._is_box_covered_by_results(
                candidate.display_box,
                resolved_results,
            ):
                continue
            block = self._build_image_candidate_block(image, candidate.display_box)
            if block is None:
                continue
            block_results = self._resolve_image_candidate(
                block,
                candidate=candidate,
                warnings=warnings,
                depth=0,
            )
            for result in block_results:
                if not self._is_duplicate_image_result(result, resolved_results):
                    resolved_results.append(result)
        return self._dedupe_final_image_results(resolved_results)

    def _resolve_image_candidate(
        self,
        block: ImageCandidateBlock,
        *,
        candidate: ImageCandidate,
        warnings: list[str],
        depth: int,
    ) -> list[OCRResult]:
        if depth > self.handwriting_split_depth_limit:
            self._add_warning(warnings, self.image_warning_text)
            return []
        base_results = self._review_image_candidate_block_results(block)
        review = self._aggregate_image_review_results(base_results)
        if self._should_retry_image_candidate_review(block, candidate, review):
            review = self._review_image_candidate_block_with_retry_from_results(block, base_results)
        split_count = self._determine_image_split_count(block, candidate, review)
        if split_count > 1:
            child_blocks = self._split_image_candidate_block(block, char_hint=split_count)
            if child_blocks:
                resolved_results: list[OCRResult] = []
                for child_block in child_blocks:
                    resolved_results.extend(
                        self._resolve_image_candidate(
                            child_block,
                            candidate=ImageCandidate(display_box=child_block.display_box, sources={"split"}),
                            warnings=warnings,
                            depth=depth + 1,
                        )
                    )
                if resolved_results:
                    return resolved_results
            self._add_warning(warnings, self.image_warning_text)
            return []
        result = self._build_image_result_from_review(block, candidate, review)
        return [result] if result is not None else []

    def _resolve_image_results_with_ocr_fallback(
        self,
        image: np.ndarray,
        results: list[OCRResult],
        warnings: list[str],
    ) -> list[OCRResult]:
        if not results:
            return []
        resolved_results: list[OCRResult] = []
        for result in self._sort_results(results):
            if len(result.text) == 1:
                resolved_results.append(result)
                continue
            if len(result.text) < 2:
                continue
            split_results = self._split_image_multi_digit_result(image, result, warnings)
            if split_results:
                resolved_results.extend(split_results)
            else:
                self._add_warning(warnings, self.image_warning_text)
        return resolved_results

    def _merge_image_result_sets(
        self,
        primary_results: list[OCRResult],
        fallback_results: list[OCRResult],
    ) -> list[OCRResult]:
        merged_results = list(self._dedupe_final_image_results(primary_results))
        for fallback_result in self._sort_results(fallback_results):
            result_box = self._polygon_to_region_box(fallback_result.box)
            if merged_results:
                if float(fallback_result.score) < self.image_candidate_review_score:
                    continue
                if self._is_duplicate_image_result(fallback_result, merged_results):
                    continue
                if self._is_box_covered_by_results(result_box, merged_results):
                    continue
            merged_results.append(fallback_result)
        return self._dedupe_final_image_results(merged_results)
