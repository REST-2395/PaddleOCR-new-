"""Block-building helpers for the handwriting pipeline."""

from __future__ import annotations

import numpy as np


class HandwritingBlockBuilderMixin:
    """Build and split handwriting candidate blocks."""

    def _build_handwriting_blocks(self, image: np.ndarray) -> list["HandwritingBlock"]:
        blocks: list["HandwritingBlock"] = []
        for region_box in self._extract_handwriting_regions(image):
            block = self._build_handwriting_block(image, region_box)
            if block is not None:
                blocks.append(block)
        return blocks

    def _build_handwriting_block(
        self,
        image: np.ndarray,
        region_box: tuple[int, int, int, int],
    ) -> "HandwritingBlock | None":
        x0, y0, x1, y1 = region_box
        region = image[y0:y1, x0:x1]
        return self._create_handwriting_block(region, region_box)

    def _create_handwriting_block(
        self,
        region_image: np.ndarray,
        display_box: tuple[int, int, int, int],
    ) -> "HandwritingBlock | None":
        if region_image is None or region_image.size == 0:
            return None

        block_region = self._ensure_bgr(region_image)
        padding = max(12, int(0.24 * max(block_region.shape[:2])))
        ocr_region = self._pad_image_border(block_region, padding)
        primary_image = self._resize_to_min_side(ocr_region, min_short_side=self.handwriting_candidate_min_side)
        fallback_image = self._normalize_handwriting_region(ocr_region, min_side=self.handwriting_candidate_min_side)

        return self._service_types().HandwritingBlock(
            display_box=display_box,
            region_image=block_region,
            primary_image=primary_image,
            fallback_image=fallback_image,
        )

    def _split_handwriting_block(self, block: "HandwritingBlock", *, char_hint: int) -> list["HandwritingBlock"]:
        if char_hint < 2:
            return []

        binary = self._build_handwriting_binary_mask(block.region_image)
        if not np.any(binary):
            return []

        local_boxes = self._extract_projection_split_boxes(binary, segment_count=char_hint)
        if not local_boxes:
            local_boxes = self._extract_projection_split_boxes(binary, segment_count=2)
        if not local_boxes:
            return []

        child_blocks: list["HandwritingBlock"] = []
        origin_x, origin_y = block.display_box[0], block.display_box[1]
        for local_x0, local_y0, local_x1, local_y1 in local_boxes:
            child_region = block.region_image[local_y0:local_y1, local_x0:local_x1]
            child_box = (
                origin_x + local_x0,
                origin_y + local_y0,
                origin_x + local_x1,
                origin_y + local_y1,
            )
            child_block = self._create_handwriting_block(child_region, child_box)
            if child_block is not None:
                child_blocks.append(child_block)

        return self._sort_handwriting_blocks(child_blocks)

    def _extract_projection_split_boxes(
        self,
        binary: np.ndarray,
        *,
        segment_count: int,
    ) -> list[tuple[int, int, int, int]]:
        if segment_count < 2:
            return []
        mask = (binary > 0).astype(np.uint8)
        if not np.any(mask):
            return []
        row_counts = np.count_nonzero(mask, axis=1)
        col_counts = np.count_nonzero(mask, axis=0)
        active_rows = np.flatnonzero(row_counts)
        active_cols = np.flatnonzero(col_counts)
        if active_rows.size == 0 or active_cols.size == 0:
            return []
        top = int(active_rows[0])
        bottom = int(active_rows[-1]) + 1
        left = int(active_cols[0])
        right = int(active_cols[-1]) + 1
        window_counts = col_counts[left:right]
        min_segment_width = self._min_split_segment_width(
            width=max(1, right - left),
            height=max(1, bottom - top),
            segment_count=segment_count,
        )
        cut_positions = self._choose_projection_cuts(
            window_counts,
            segment_count=segment_count,
            min_segment_width=min_segment_width,
        )
        if len(cut_positions) != segment_count - 1:
            return []
        boundaries = [0, *cut_positions, len(window_counts)]
        split_boxes: list[tuple[int, int, int, int]] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start < min_segment_width:
                return []
            split_boxes.append((left + start, top, left + end, bottom))
        return split_boxes

    def _choose_projection_cuts(
        self,
        column_counts: np.ndarray,
        *,
        segment_count: int,
        min_segment_width: int,
    ) -> list[int]:
        positions = self._candidate_split_positions(column_counts, min_segment_width)
        if len(positions) < segment_count - 1:
            return []
        ranked_positions = sorted(
            positions,
            key=lambda position: (
                self._projection_cut_score(column_counts, position),
                abs(position - (len(column_counts) / max(1, segment_count))),
            ),
        )
        selected: list[int] = []
        for position in ranked_positions:
            if position < min_segment_width or len(column_counts) - position < min_segment_width:
                continue
            if any(abs(position - existing) < min_segment_width for existing in selected):
                continue
            selected.append(position)
            if len(selected) == segment_count - 1:
                break
        if len(selected) != segment_count - 1:
            return []
        ordered = sorted(selected)
        boundaries = [0, *ordered, len(column_counts)]
        if any(end - start < min_segment_width for start, end in zip(boundaries[:-1], boundaries[1:])):
            return []
        return ordered

    @staticmethod
    def _projection_cut_score(column_counts: np.ndarray, position: int) -> float:
        window_start = max(0, position - 1)
        window_end = min(len(column_counts), position + 2)
        return float(np.mean(column_counts[window_start:window_end]))

    @staticmethod
    def _candidate_split_positions(column_counts: np.ndarray, min_segment_width: int) -> list[int]:
        positions: list[int] = []
        for index in range(min_segment_width, len(column_counts) - min_segment_width):
            current = column_counts[index]
            left = column_counts[max(0, index - 1)]
            right = column_counts[min(len(column_counts) - 1, index + 1)]
            if current <= left and current <= right:
                positions.append(index)
        return positions

    @staticmethod
    def _min_split_segment_width(*, width: int, height: int, segment_count: int) -> int:
        return max(3, min(width // max(1, segment_count), max(3, int(round(height * 0.18)))))
