"""Structured sequence helpers for the image OCR pipeline."""

from __future__ import annotations

import numpy as np

from ..ocr_engine import OCRResult
from ..service_types import ImageCandidateBlock, ImageReviewResult, StructuredSequence


class StructuredSequenceMixin:
    """Resolve structured OCR sequences from wide panel crops."""

    def _resolve_structured_photo_results(
        self,
        image: np.ndarray,
        ocr_results: list[OCRResult],
    ) -> tuple[list[OCRResult], list[str]]:
        warnings: list[str] = []
        panel_boxes = self._filter_structured_panel_boxes(self._extract_image_candidate_boxes(image))
        row_groups = self._group_region_boxes_into_rows(panel_boxes)
        if not self._should_use_structured_photo_mode(ocr_results, row_groups):
            return [], warnings
        sequences = self._collect_structured_sequences(ocr_results)
        sequence_results = self._resolve_structured_sequences(image, sequences, warnings)
        minimum_expected_count = self._minimum_structured_result_count(sequences, row_groups)
        panel_results: list[OCRResult] = []
        if self._should_resolve_structured_panel(
            existing_results=sequence_results,
            minimum_expected_count=minimum_expected_count,
        ):
            panel_results = self._resolve_structured_panel_candidates(
                image,
                panel_boxes,
                sequence_results,
                minimum_expected_count=minimum_expected_count,
            )
        merged_results = self._dedupe_structured_results([*sequence_results, *panel_results])
        if minimum_expected_count > 0 and len(merged_results) < minimum_expected_count:
            return [], warnings
        return merged_results, warnings

    def _should_use_structured_photo_mode(
        self,
        ocr_results: list[OCRResult],
        row_groups: list[list[tuple[int, int, int, int]]],
    ) -> bool:
        strong_sequence_chars = sum(
            len(result.text)
            for result in ocr_results
            if len(result.text) >= 2
            and float(result.score) >= self.structured_sequence_min_score
            and self._box_width(self._polygon_to_region_box(result.box))
            >= max(40, int(self._box_height(self._polygon_to_region_box(result.box)) * 1.45))
        )
        if strong_sequence_chars >= 2:
            return True
        structured_rows = [row for row in row_groups if len(row) >= self.structured_route_min_row_items]
        if not structured_rows:
            return False
        if len(structured_rows) >= 2:
            return True
        return any(self._structured_row_is_consistent(row) for row in structured_rows)

    def _minimum_structured_result_count(
        self,
        sequences: list[StructuredSequence],
        row_groups: list[list[tuple[int, int, int, int]]],
    ) -> int:
        stable_rows = [row for row in row_groups if len(row) >= 2]
        expected_counts: list[int] = []
        if sequences:
            expected_counts.append(sum(len(sequence.text_hint) for sequence in sequences))
        if stable_rows:
            expected_counts.append(sum(len(row) for row in stable_rows))
        elif row_groups:
            expected_counts.append(max(len(row) for row in row_groups))
        return max(expected_counts, default=0)

    def _collect_structured_sequences(self, ocr_results: list[OCRResult]) -> list[StructuredSequence]:
        sequences: list[StructuredSequence] = []
        for result in self._sort_results(ocr_results):
            if len(result.text) < 2 or float(result.score) < self.structured_sequence_min_score:
                continue
            region_box = self._polygon_to_region_box(result.box)
            width = self._box_width(region_box)
            height = self._box_height(region_box)
            if width < max(40, int(height * 1.45)):
                continue
            if any(self._are_boxes_layout_equivalent(region_box, existing.display_box) for existing in sequences):
                continue
            sequences.append(
                StructuredSequence(
                    polygon=[point[:] for point in result.box],
                    display_box=region_box,
                    text_hint=result.text,
                    score_hint=float(result.score),
                )
            )
        return sequences

    def _resolve_structured_sequences(
        self,
        image: np.ndarray,
        sequences: list[StructuredSequence],
        warnings: list[str],
    ) -> list[OCRResult]:
        resolved_results: list[OCRResult] = []
        for sequence in sequences:
            resolved_results.extend(self._resolve_structured_sequence(image, sequence, warnings))
        return resolved_results

    def _resolve_structured_sequence(
        self,
        image: np.ndarray,
        sequence: StructuredSequence,
        warnings: list[str],
    ) -> list[OCRResult]:
        warped_image, inverse_matrix = self._warp_polygon_crop(image, sequence.polygon)
        if warped_image is None or inverse_matrix is None:
            return []
        sequence_block = self._create_image_candidate_block(
            warped_image,
            (0, 0, warped_image.shape[1], warped_image.shape[0]),
        )
        if sequence_block is None:
            return []
        split_boxes = self._split_structured_sequence_boxes(sequence_block.foreground_mask, len(sequence.text_hint))
        if len(split_boxes) != len(sequence.text_hint):
            self._add_warning(warnings, self.image_warning_text)
            return []
        split_is_stable = self._structured_split_boxes_are_stable(
            split_boxes,
            total_width=max(1, sequence_block.foreground_mask.shape[1]),
        )
        child_results: list[OCRResult] = []
        for index, (split_box, hint_char) in enumerate(zip(split_boxes, sequence.text_hint)):
            child_block = self._create_image_candidate_block(
                sequence_block.region_image[split_box[1] : split_box[3], split_box[0] : split_box[2]],
                split_box,
            )
            if child_block is None:
                self._add_warning(warnings, self.image_warning_text)
                return []
            base_results = self._review_image_candidate_block_results(child_block)
            review = self._aggregate_image_review_results(base_results)
            if self._should_retry_structured_child_review(
                child_block,
                review=review,
                hint_char=hint_char,
                hint_score=sequence.score_hint,
                position_index=index,
                total_positions=len(sequence.text_hint),
                split_is_stable=split_is_stable,
            ):
                review = self._review_image_candidate_block_with_retry_from_results(child_block, base_results)
            result = self._build_structured_sequence_result(
                child_block,
                review=review,
                hint_char=hint_char,
                hint_score=sequence.score_hint,
                inverse_matrix=inverse_matrix,
                image_shape=image.shape,
            )
            if result is None:
                self._add_warning(warnings, self.image_warning_text)
                return []
            child_results.append(result)
        return child_results

    def _split_structured_sequence_boxes(self, binary: np.ndarray, segment_count: int) -> list[tuple[int, int, int, int]]:
        split_boxes = self._extract_projection_split_boxes(binary, segment_count=segment_count)
        if split_boxes and len(split_boxes) == segment_count and all(box[2] - box[0] >= 3 for box in split_boxes):
            return split_boxes
        return self._build_even_split_boxes(binary, segment_count)

    def _build_even_split_boxes(self, binary: np.ndarray, segment_count: int) -> list[tuple[int, int, int, int]]:
        if segment_count < 2:
            return []
        height, width = binary.shape[:2]
        if width < segment_count:
            return []
        split_boxes: list[tuple[int, int, int, int]] = []
        boundaries = np.linspace(0, width, num=segment_count + 1, dtype=int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                return []
            split_boxes.append((start, 0, end, height))
        return split_boxes

    def _should_retry_structured_child_review(
        self,
        block: ImageCandidateBlock,
        *,
        review: ImageReviewResult,
        hint_char: str,
        hint_score: float,
        position_index: int,
        total_positions: int,
        split_is_stable: bool,
    ) -> bool:
        aspect_ratio = self._box_width(block.display_box) / float(max(1, self._box_height(block.display_box)))
        edge_position = position_index in {0, max(0, total_positions - 1)}
        strong_hint_score = max(self.structured_sequence_min_score + 0.12, 0.90)
        if hint_char and hint_score >= strong_hint_score and split_is_stable and not edge_position and 0.22 <= aspect_ratio <= 1.25:
            return False
        if not hint_char or hint_score < self.structured_sequence_min_score:
            return True
        if self._is_reliable_image_review(review) and review.text == hint_char:
            return False
        if not split_is_stable or aspect_ratio < 0.18 or aspect_ratio > 1.35:
            return True
        return edge_position and not self._is_reliable_image_review(review)
