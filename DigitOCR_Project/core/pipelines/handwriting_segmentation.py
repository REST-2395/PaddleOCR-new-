"""Segmentation helpers for the handwriting pipeline."""

from __future__ import annotations

import cv2
import numpy as np


class HandwritingSegmentationMixin:
    """Extract and merge handwriting regions conservatively."""

    def _extract_handwriting_regions(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.handwriting_threshold, 255, cv2.THRESH_BINARY_INV)

        kernel_size = self._segmentation_kernel_size(image.shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        component_boxes = self._extract_component_boxes(binary)
        if not component_boxes:
            return []

        merged_regions = self._merge_component_boxes(component_boxes)
        return self._sort_region_boxes(merged_regions)

    def _extract_component_boxes(self, binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = max(24, (binary.shape[0] * binary.shape[1]) // 12000)
        component_boxes: list[tuple[int, int, int, int]] = []

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < min_area or width < 3 or height < 3:
                continue

            component_boxes.append((x, y, x + width, y + height))

        if component_boxes:
            return component_boxes

        points = cv2.findNonZero(binary)
        if points is None:
            return []

        x, y, width, height = cv2.boundingRect(points)
        return [(x, y, x + width, y + height)]

    def _merge_component_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if len(boxes) <= 1:
            return boxes

        median_width = int(np.median([self._box_width(box) for box in boxes]))
        median_height = int(np.median([self._box_height(box) for box in boxes]))
        parents = list(range(len(boxes)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(first_index: int, second_index: int) -> None:
            first_root = find(first_index)
            second_root = find(second_index)
            if first_root != second_root:
                parents[second_root] = first_root

        for first_index in range(len(boxes)):
            for second_index in range(first_index + 1, len(boxes)):
                if self._should_merge_boxes(
                    boxes[first_index],
                    boxes[second_index],
                    median_width=median_width,
                    median_height=median_height,
                ):
                    union(first_index, second_index)

        grouped_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
        for index, box in enumerate(boxes):
            grouped_boxes.setdefault(find(index), []).append(box)

        merged_boxes = [self._combine_boxes(group) for group in grouped_boxes.values()]
        return self._merge_boxes_until_stable(merged_boxes, median_width=median_width, median_height=median_height)

    def _merge_boxes_until_stable(
        self,
        boxes: list[tuple[int, int, int, int]],
        *,
        median_width: int,
        median_height: int,
    ) -> list[tuple[int, int, int, int]]:
        pending_boxes = boxes[:]

        while len(pending_boxes) > 1:
            merged_any = False
            next_round: list[tuple[int, int, int, int]] = []
            consumed = [False] * len(pending_boxes)

            for index, current_box in enumerate(pending_boxes):
                if consumed[index]:
                    continue

                consumed[index] = True
                merged_box = current_box

                for other_index in range(index + 1, len(pending_boxes)):
                    if consumed[other_index]:
                        continue

                    if self._should_merge_boxes(
                        merged_box,
                        pending_boxes[other_index],
                        median_width=median_width,
                        median_height=median_height,
                    ):
                        merged_box = self._combine_boxes((merged_box, pending_boxes[other_index]))
                        consumed[other_index] = True
                        merged_any = True

                next_round.append(merged_box)

            pending_boxes = next_round
            if not merged_any:
                break

        return pending_boxes

    def _should_merge_boxes(
        self,
        first_box: tuple[int, int, int, int],
        second_box: tuple[int, int, int, int],
        *,
        median_width: int,
        median_height: int,
    ) -> bool:
        gap_x = self._axis_gap((first_box[0], first_box[2]), (second_box[0], second_box[2]))
        gap_y = self._axis_gap((first_box[1], first_box[3]), (second_box[1], second_box[3]))
        vertical_overlap = self._axis_overlap_ratio((first_box[1], first_box[3]), (second_box[1], second_box[3]))
        horizontal_overlap = self._axis_overlap_ratio((first_box[0], first_box[2]), (second_box[0], second_box[2]))

        reference_dim = max(1, max(median_width, median_height))
        max_dim = max(
            self._box_width(first_box),
            self._box_height(first_box),
            self._box_width(second_box),
            self._box_height(second_box),
        )
        tiny_gap = max(4, int(0.06 * reference_dim), int(0.04 * max_dim))

        if gap_x == 0 and gap_y == 0:
            return True

        if gap_x <= tiny_gap and vertical_overlap >= 0.82:
            return True

        if gap_y <= tiny_gap and horizontal_overlap >= 0.82:
            return True

        return False

    def _sort_handwriting_blocks(self, blocks: list["HandwritingBlock"]) -> list["HandwritingBlock"]:
        if not blocks:
            return []

        box_to_block = {block.display_box: block for block in blocks}
        ordered_boxes = self._sort_region_boxes(list(box_to_block))
        return [box_to_block[box] for box in ordered_boxes if box in box_to_block]

    def _build_handwriting_binary_mask(self, image: np.ndarray) -> np.ndarray:
        return self._build_foreground_mask(image)

    def _build_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(self._ensure_bgr(image), cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.handwriting_threshold, 255, cv2.THRESH_BINARY_INV)
        if min(binary.shape[:2]) >= 5:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return binary

    def _estimate_foreground_angle(self, image: np.ndarray) -> float:
        binary = self._build_foreground_mask(image)
        points = cv2.findNonZero(binary)
        if points is None or len(points) < 5:
            return 0.0

        (_, _), (width, height), angle = cv2.minAreaRect(points)
        if width < height:
            angle += 90.0

        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return float(angle)

    def _build_rotation_retry_angles(self, estimated_angle: float) -> list[float]:
        candidate_angles: list[float] = []
        base_angle = -float(estimated_angle)
        for extra_angle in self.rotation_retry_angles:
            candidate_angle = base_angle + extra_angle
            if any(abs(candidate_angle - existing) < 1e-3 for existing in candidate_angles):
                continue
            candidate_angles.append(candidate_angle)
        return candidate_angles

    def _validate_handwriting_content(self, image: np.ndarray) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not np.any(gray < self.handwriting_threshold):
            raise ValueError("Please write digits on the canvas before recognition.")
