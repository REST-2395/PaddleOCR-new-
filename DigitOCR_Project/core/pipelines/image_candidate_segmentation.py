"""Segmentation and dedupe helpers for image OCR candidates."""

from __future__ import annotations

import cv2
import numpy as np

from ..ocr_engine import OCRResult
from ..service_types import ImageCandidate, ImageCandidateBlock, RegionBox


class ImageCandidateSegmentationMixin:
    """Handle segmentation, dedupe, and split helpers for image candidates."""

    def _estimate_projection_segment_count(
        self,
        binary: np.ndarray,
        *,
        preferred_count: int,
        aspect_ratio: float,
        max_segments: int | None = None,
    ) -> int:
        effective_max_segments = self.image_candidate_max_segments if max_segments is None else max(2, int(max_segments))
        candidate_counts: list[int] = []
        if preferred_count > 1:
            candidate_counts.append(min(effective_max_segments, preferred_count))
        max_count = min(effective_max_segments, max(2, int(round(aspect_ratio + 0.35))))
        candidate_counts.extend(range(max_count, 1, -1))
        seen: set[int] = set()
        for segment_count in candidate_counts:
            if segment_count in seen or segment_count < 2:
                continue
            seen.add(segment_count)
            if self._extract_projection_split_boxes(binary, segment_count=segment_count):
                return segment_count
        return 1

    def _split_image_candidate_block(
        self,
        block: ImageCandidateBlock,
        *,
        char_hint: int,
    ) -> list[ImageCandidateBlock]:
        if char_hint < 2:
            return []
        local_boxes = self._extract_projection_split_boxes(block.foreground_mask, segment_count=char_hint)
        if not local_boxes and char_hint != 2:
            local_boxes = self._extract_projection_split_boxes(block.foreground_mask, segment_count=2)
        if not local_boxes:
            return []
        child_blocks: list[ImageCandidateBlock] = []
        origin_x, origin_y = block.display_box[0], block.display_box[1]
        for local_x0, local_y0, local_x1, local_y1 in local_boxes:
            child_region = block.region_image[local_y0:local_y1, local_x0:local_x1]
            child_box = (
                origin_x + local_x0,
                origin_y + local_y0,
                origin_x + local_x1,
                origin_y + local_y1,
            )
            child_block = self._create_image_candidate_block(child_region, child_box)
            if child_block is not None:
                child_blocks.append(child_block)
        return self._sort_image_blocks(child_blocks)

    def _build_image_segmentation_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(self._ensure_bgr(image), cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _build_image_polarity_mask(self, gray: np.ndarray, *, invert: bool) -> np.ndarray:
        source = 255 - gray if invert else gray
        block_size = self._image_adaptive_block_size(source.shape)
        adaptive = cv2.adaptiveThreshold(
            source,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            7,
        )
        _, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_or(adaptive, otsu)
        kernel_size = self._image_mask_kernel_size(source.shape)
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _filter_image_mask_components(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        component_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
        if not component_areas:
            return mask
        min_area = max(6, int(max(component_areas) * 0.04), mask.size // 4000)
        filtered = np.zeros_like(mask)
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                filtered[labels == label] = 255
        return filtered if np.any(filtered) else mask

    def _extract_image_component_boxes(
        self,
        mask: np.ndarray,
        *,
        image_shape: tuple[int, ...],
    ) -> list[RegionBox]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = max(20, (mask.shape[0] * mask.shape[1]) // 18000)
        max_area = max(min_area + 1, int(mask.shape[0] * mask.shape[1] * 0.55))
        component_boxes: list[RegionBox] = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area or width < 4 or height < 4:
                continue
            aspect_ratio = width / float(max(1, height))
            inverse_aspect_ratio = height / float(max(1, width))
            fill_ratio = area / float(max(1, width * height))
            if aspect_ratio > 9.0 or inverse_aspect_ratio > 18.0:
                continue
            if fill_ratio < 0.06 and area < min_area * 2:
                continue
            padding = max(2, int(round(min(width, height) * 0.08)))
            component_boxes.append(
                self._expand_region_box((x, y, x + width, y + height), image_shape, padding=padding)
            )
        return component_boxes

    def _image_mask_quality(self, mask: np.ndarray) -> tuple[bool, bool, float, float, int]:
        non_zero = int(np.count_nonzero(mask))
        if non_zero == 0:
            return False, False, float("-inf"), float("-inf"), 0
        points = cv2.findNonZero(mask)
        if points is None:
            return False, False, float("-inf"), float("-inf"), 0
        _, _, width, height = cv2.boundingRect(points)
        bbox_area = max(1, width * height)
        image_area = max(1, mask.shape[0] * mask.shape[1])
        area_ratio = non_zero / float(image_area)
        fill_ratio = non_zero / float(bbox_area)
        return (
            0.01 <= area_ratio <= 0.70,
            0.05 <= fill_ratio <= 0.90,
            -abs(area_ratio - 0.18),
            -abs(fill_ratio - 0.35),
            non_zero,
        )

    def _dedupe_region_boxes(self, boxes: list[RegionBox]) -> list[RegionBox]:
        deduped_boxes: list[RegionBox] = []
        for box in boxes:
            matched = False
            for index, existing_box in enumerate(deduped_boxes):
                if self._are_boxes_equivalent(box, existing_box):
                    if self._box_area(box) < self._box_area(existing_box):
                        deduped_boxes[index] = box
                    matched = True
                    break
            if not matched:
                deduped_boxes.append(box)
        return deduped_boxes

    def _dedupe_image_candidates(self, candidates: list[ImageCandidate]) -> list[ImageCandidate]:
        deduped_candidates: list[ImageCandidate] = []
        for candidate in sorted(candidates, key=self._image_candidate_sort_key):
            matched_candidate: ImageCandidate | None = None
            for existing_candidate in deduped_candidates:
                if self._are_boxes_equivalent(candidate.display_box, existing_candidate.display_box):
                    matched_candidate = existing_candidate
                    break
            if matched_candidate is None:
                deduped_candidates.append(
                    ImageCandidate(
                        display_box=candidate.display_box,
                        text_hint=candidate.text_hint,
                        score_hint=candidate.score_hint,
                        sources=set(candidate.sources),
                    )
                )
                continue
            self._merge_image_candidate(matched_candidate, candidate)
        return deduped_candidates

    def _merge_image_candidate(self, target: ImageCandidate, incoming: ImageCandidate) -> None:
        target.sources.update(incoming.sources)
        if self._box_area(incoming.display_box) < self._box_area(target.display_box):
            target.display_box = incoming.display_box
        if incoming.text_hint and (
            not target.text_hint
            or len(incoming.text_hint) > len(target.text_hint)
            or incoming.score_hint > target.score_hint
        ):
            target.text_hint = incoming.text_hint
            target.score_hint = incoming.score_hint
        else:
            target.score_hint = max(target.score_hint, incoming.score_hint)

    def _image_candidate_sort_key(self, candidate: ImageCandidate) -> tuple[int, int, int, int]:
        source_rank = 0 if "segment" in candidate.sources else 1
        x0, y0, _, _ = candidate.display_box
        return self._box_area(candidate.display_box), source_rank, y0, x0

    def _sort_image_blocks(self, blocks: list[ImageCandidateBlock]) -> list[ImageCandidateBlock]:
        if not blocks:
            return []
        block_by_box = {block.display_box: block for block in blocks}
        ordered_boxes = self._sort_region_boxes(list(block_by_box))
        return [block_by_box[box] for box in ordered_boxes if box in block_by_box]

    def _is_box_covered_by_results(self, box: RegionBox, results: list[OCRResult]) -> bool:
        if not results:
            return False
        overlapping_areas = [
            self._intersection_area(box, self._polygon_to_region_box(result.box))
            for result in results
            if self._intersection_area(box, self._polygon_to_region_box(result.box)) > 0
        ]
        if not overlapping_areas:
            return False
        box_area = max(1, self._box_area(box))
        coverage = min(box_area, sum(sorted(overlapping_areas, reverse=True)[:4])) / float(box_area)
        return coverage >= 0.88 or (len(overlapping_areas) >= 2 and coverage >= 0.65)

    def _is_duplicate_image_result(self, result: OCRResult, existing_results: list[OCRResult]) -> bool:
        result_box = self._polygon_to_region_box(result.box)
        for existing_result in existing_results:
            existing_box = self._polygon_to_region_box(existing_result.box)
            if result.text == existing_result.text and self._are_boxes_equivalent(result_box, existing_box):
                return True
        return False

    def _dedupe_final_image_results(self, results: list[OCRResult]) -> list[OCRResult]:
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
            if not self._is_duplicate_image_result(result, unique_results):
                unique_results.append(result)
        return self._sort_results(unique_results)

    def _split_image_multi_digit_result(
        self,
        image: np.ndarray,
        result: OCRResult,
        warnings: list[str],
    ) -> list[OCRResult]:
        region_box = self._expand_region_box(
            self._polygon_to_region_box(result.box),
            image.shape,
            padding=max(4, int(round(max(1, len(result.text)) * 2))),
        )
        block = self._build_image_candidate_block(image, region_box)
        if block is None:
            return []
        child_blocks = self._split_image_candidate_block(block, char_hint=len(result.text))
        if not child_blocks or len(child_blocks) != len(result.text):
            return []
        resolved_results: list[OCRResult] = []
        per_digit_hint_score = min(0.99, max(0.0, float(result.score) - 0.08))
        for child_block, hint_char in zip(child_blocks, result.text):
            child_results = self._resolve_image_candidate(
                child_block,
                candidate=ImageCandidate(
                    display_box=child_block.display_box,
                    text_hint=hint_char,
                    score_hint=per_digit_hint_score,
                    sources={"fallback"},
                ),
                warnings=warnings,
                depth=1,
            )
            if child_results:
                resolved_results.extend(child_results)
            elif len(hint_char) == 1 and float(result.score) >= self.image_candidate_strong_review_score:
                resolved_results.append(
                    OCRResult(
                        text=hint_char,
                        score=per_digit_hint_score,
                        box=self._region_box_to_polygon(child_block.display_box),
                    )
                )
        return resolved_results
