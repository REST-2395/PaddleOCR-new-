"""Structured panel and row-layout helpers for the image OCR pipeline."""

from __future__ import annotations

import numpy as np

from ..ocr_engine import OCRResult
from ..service_types import ImageCandidate, ImageCandidateBlock, ImageReviewResult


class StructuredPanelMixin:
    """Resolve missing structured panel positions and row layout."""

    @staticmethod
    def _should_resolve_structured_panel(
        *,
        existing_results: list[OCRResult],
        minimum_expected_count: int,
    ) -> bool:
        if minimum_expected_count <= 0:
            return False
        return len(existing_results) < minimum_expected_count

    def _resolve_structured_missing_positions(
        self,
        panel_boxes: list[tuple[int, int, int, int]],
        occupied_boxes: list[tuple[int, int, int, int]],
        *,
        limit: int,
    ) -> list[tuple[int, int, int, int]]:
        if limit <= 0:
            return []
        available_boxes = [
            box
            for box in panel_boxes
            if not any(self._are_boxes_layout_equivalent(box, occupied_box) for occupied_box in occupied_boxes)
        ]
        return self._sort_region_boxes(available_boxes)[:limit]

    def _should_retry_structured_panel_review(
        self,
        block: ImageCandidateBlock,
        *,
        review: ImageReviewResult,
    ) -> bool:
        aspect_ratio = self._box_width(block.display_box) / float(max(1, self._box_height(block.display_box)))
        if self._is_reliable_image_review(review):
            return False
        return 0.20 <= aspect_ratio <= 1.20 and self._box_height(block.display_box) >= 18

    def _dedupe_structured_results(self, results: list[OCRResult]) -> list[OCRResult]:
        unique_results: list[OCRResult] = []
        ranked_results = sorted(
            results,
            key=lambda item: (
                -float(item.score),
                self._box_area(self._polygon_to_region_box(item.box)),
                min(point[1] for point in item.box),
                min(point[0] for point in item.box),
            ),
        )
        for result in ranked_results:
            result_box = self._polygon_to_region_box(result.box)
            if any(
                existing_result.text == result.text
                and self._are_boxes_layout_equivalent(result_box, self._polygon_to_region_box(existing_result.box))
                for existing_result in unique_results
            ):
                continue
            unique_results.append(result)
        return self._sort_results(unique_results)

    def _build_structured_sequence_result(
        self,
        block: ImageCandidateBlock,
        *,
        review: ImageReviewResult,
        hint_char: str,
        hint_score: float,
        inverse_matrix: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> OCRResult | None:
        selected_text = ""
        selected_score = 0.0
        strong_hint_score = max(self.structured_sequence_min_score + 0.12, 0.90)
        if hint_char and hint_score >= strong_hint_score:
            selected_text = hint_char
            selected_score = min(0.99, hint_score)
            if review.text == hint_char:
                selected_score = max(selected_score, review.score)
        elif hint_char and hint_score >= self.structured_sequence_min_score:
            selected_text = hint_char
            selected_score = min(0.99, hint_score)
            if review.text == hint_char and self._is_reliable_image_review(review):
                selected_score = max(selected_score, review.score)
            elif self._is_reliable_image_review(review) and review.text and review.text != hint_char:
                selected_text = review.text
                selected_score = review.score
        elif self._is_reliable_image_review(review):
            selected_text = review.text
            selected_score = review.score
        elif review.text:
            selected_text = review.text
            selected_score = review.score
        if not selected_text:
            return None
        mapped_polygon = self._map_polygon_with_perspective(
            self._region_box_to_polygon(block.display_box),
            inverse_matrix,
            to_shape=image_shape,
        )
        return OCRResult(text=selected_text, score=selected_score, box=mapped_polygon)

    def _resolve_structured_panel_candidates(
        self,
        image: np.ndarray,
        panel_boxes: list[tuple[int, int, int, int]],
        existing_results: list[OCRResult],
        *,
        minimum_expected_count: int,
    ) -> list[OCRResult]:
        resolved_results: list[OCRResult] = []
        occupied_boxes = [self._polygon_to_region_box(result.box) for result in existing_results]
        missing_boxes = self._resolve_structured_missing_positions(
            panel_boxes,
            occupied_boxes,
            limit=max(0, minimum_expected_count - len(existing_results)),
        )
        for box in missing_boxes:
            if any(self._are_boxes_layout_equivalent(box, occupied_box) for occupied_box in occupied_boxes):
                continue
            block = self._build_image_candidate_block(image, box)
            if block is None:
                continue
            base_results = self._review_image_candidate_block_results(block)
            review = self._aggregate_image_review_results(base_results)
            if self._should_retry_structured_panel_review(block, review=review):
                review = self._review_image_candidate_block_with_retry_from_results(block, base_results)
            result = self._build_image_result_from_review(
                block,
                ImageCandidate(display_box=box, sources={"structured_panel"}),
                review,
            )
            if result is None:
                continue
            result_box = self._polygon_to_region_box(result.box)
            if any(
                existing_result.text == result.text
                and self._are_boxes_layout_equivalent(result_box, self._polygon_to_region_box(existing_result.box))
                for existing_result in [*existing_results, *resolved_results]
            ):
                continue
            occupied_boxes.append(result_box)
            resolved_results.append(result)
            if len(existing_results) + len(resolved_results) >= minimum_expected_count:
                break
        return resolved_results

    def _filter_structured_panel_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        row_groups = self._group_region_boxes_into_rows(boxes)
        candidate_rows = [row for row in row_groups if len(row) >= 2] or row_groups
        filtered_boxes: list[tuple[int, int, int, int]] = []
        for row in candidate_rows:
            if not row:
                continue
            row_areas = [self._box_area(box) for box in row]
            row_heights = [self._box_height(box) for box in row]
            row_aspects = [self._box_width(box) / float(max(1, self._box_height(box))) for box in row]
            median_area = float(np.median(row_areas))
            median_height = float(np.median(row_heights))
            median_aspect = float(np.median(row_aspects))
            stable_row = [
                box
                for box in row
                if self._box_area(box) >= median_area * self.structured_panel_min_area_ratio
                and self._box_height(box) >= median_height * self.structured_panel_min_height_ratio
                and (self._box_width(box) / float(max(1, self._box_height(box)))) >= max(0.08, median_aspect * 0.55)
            ]
            if len(stable_row) >= max(1, min(2, len(row))):
                filtered_boxes.extend(stable_row)
        if filtered_boxes:
            return self._dedupe_region_boxes(self._sort_region_boxes(filtered_boxes))
        all_areas = [self._box_area(box) for box in boxes]
        all_heights = [self._box_height(box) for box in boxes]
        median_area = float(np.median(all_areas))
        median_height = float(np.median(all_heights))
        fallback_boxes = [
            box
            for box in boxes
            if self._box_area(box) >= median_area * self.structured_panel_min_area_ratio
            and self._box_height(box) >= median_height * self.structured_panel_min_height_ratio
        ]
        return self._dedupe_region_boxes(self._sort_region_boxes(fallback_boxes))

    def _group_region_boxes_into_rows(self, boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
        if not boxes:
            return []
        ordered_boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
        median_height = float(np.median([max(1, self._box_height(box)) for box in ordered_boxes]))
        row_threshold = max(12.0, median_height * 0.55)
        rows: list[dict[str, object]] = []
        for box in ordered_boxes:
            center_y = (box[1] + box[3]) / 2.0
            target_row: dict[str, object] | None = None
            for row in rows:
                if abs(center_y - float(row["center_y"])) <= row_threshold:
                    target_row = row
                    break
            if target_row is None:
                rows.append({"center_y": center_y, "boxes": [box]})
                continue
            row_boxes = target_row["boxes"]
            assert isinstance(row_boxes, list)
            row_boxes.append(box)
            target_row["center_y"] = float(np.mean([(item[1] + item[3]) / 2.0 for item in row_boxes]))
        grouped_rows: list[list[tuple[int, int, int, int]]] = []
        for row in sorted(rows, key=lambda item: float(item["center_y"])):
            row_boxes = sorted(row["boxes"], key=lambda box: box[0])
            grouped_rows.append(row_boxes)
        return grouped_rows

    def _structured_row_is_consistent(self, row: list[tuple[int, int, int, int]]) -> bool:
        if len(row) < self.structured_route_min_row_items:
            return False
        widths = [max(1, self._box_width(box)) for box in row]
        heights = [max(1, self._box_height(box)) for box in row]
        median_width = float(np.median(widths))
        median_height = float(np.median(heights))
        stable_items = [
            box
            for box in row
            if self._box_width(box) >= median_width * 0.45
            and self._box_width(box) <= median_width * 2.1
            and self._box_height(box) >= median_height * 0.55
            and self._box_height(box) <= median_height * 1.8
        ]
        return len(stable_items) >= self.structured_route_min_row_items

    @staticmethod
    def _structured_split_boxes_are_stable(split_boxes: list[tuple[int, int, int, int]], *, total_width: int) -> bool:
        if not split_boxes:
            return False
        widths = [max(1, box[2] - box[0]) for box in split_boxes]
        median_width = float(np.median(widths))
        if median_width <= 0:
            return False
        if any(width < max(3, int(total_width * 0.04)) for width in widths):
            return False
        return all(width <= median_width * 2.4 for width in widths)
