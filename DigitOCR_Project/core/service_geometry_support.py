"""Geometry and result-mapping helpers for the service facade."""

from __future__ import annotations

import numpy as np

from . import geometry as geometry_utils
from . import result_mapping as result_mapping_utils
from .ocr_engine import OCRResult
from .service_types import RegionBox


class ServiceGeometryMixin:
    """Keep geometry wrappers and layout comparison helpers outside the facade."""

    @staticmethod
    def _combine_boxes(boxes: list[RegionBox] | tuple[RegionBox, ...]) -> RegionBox:
        return geometry_utils._combine_boxes(boxes)

    @staticmethod
    def _expand_region_box(box: RegionBox, image_shape: tuple[int, ...], *, padding: int) -> RegionBox:
        return geometry_utils._expand_region_box(box, image_shape, padding=padding)

    @staticmethod
    def _region_box_to_polygon(box: RegionBox) -> list[list[int]]:
        return geometry_utils._region_box_to_polygon(box)

    @staticmethod
    def _polygon_to_region_box(points: list[list[int]]) -> RegionBox:
        return geometry_utils._polygon_to_region_box(points)

    @staticmethod
    def _box_width(box: RegionBox) -> int:
        return geometry_utils._box_width(box)

    @staticmethod
    def _box_height(box: RegionBox) -> int:
        return geometry_utils._box_height(box)

    @staticmethod
    def _box_area(box: RegionBox) -> int:
        return geometry_utils._box_area(box)

    @staticmethod
    def _box_center(box: RegionBox) -> tuple[float, float]:
        return geometry_utils._box_center(box)

    @staticmethod
    def _intersection_area(first_box: RegionBox, second_box: RegionBox) -> int:
        return geometry_utils._intersection_area(first_box, second_box)

    @staticmethod
    def _box_iou(first_box: RegionBox, second_box: RegionBox) -> float:
        return geometry_utils._box_iou(first_box, second_box)

    def _are_boxes_equivalent(self, first_box: RegionBox, second_box: RegionBox) -> bool:
        if self._box_iou(first_box, second_box) >= self.image_candidate_duplicate_iou:
            return True
        smaller_area = min(self._box_area(first_box), self._box_area(second_box))
        if smaller_area <= 0:
            return False
        overlap_ratio = self._intersection_area(first_box, second_box) / float(smaller_area)
        return overlap_ratio >= 0.88

    def _are_boxes_layout_equivalent(self, first_box: RegionBox, second_box: RegionBox) -> bool:
        if self._are_boxes_equivalent(first_box, second_box):
            return True
        smaller_area = min(self._box_area(first_box), self._box_area(second_box))
        if smaller_area <= 0:
            return False
        overlap_ratio = self._intersection_area(first_box, second_box) / float(smaller_area)
        if overlap_ratio >= 0.55:
            return True
        first_center_x, first_center_y = self._box_center(first_box)
        second_center_x, second_center_y = self._box_center(second_box)
        center_distance = float(np.hypot(first_center_x - second_center_x, first_center_y - second_center_y))
        reference_dim = max(
            self._box_width(first_box),
            self._box_height(first_box),
            self._box_width(second_box),
            self._box_height(second_box),
        )
        area_ratio = smaller_area / float(max(self._box_area(first_box), self._box_area(second_box)))
        return center_distance <= max(10.0, reference_dim * 0.35) and area_ratio >= 0.42

    @staticmethod
    def _axis_gap(first_range: tuple[int, int], second_range: tuple[int, int]) -> int:
        if first_range[1] < second_range[0]:
            return second_range[0] - first_range[1]
        if second_range[1] < first_range[0]:
            return first_range[0] - second_range[1]
        return 0

    @staticmethod
    def _axis_overlap_ratio(first_range: tuple[int, int], second_range: tuple[int, int]) -> float:
        top = max(first_range[0], second_range[0])
        bottom = min(first_range[1], second_range[1])
        overlap = max(0, bottom - top)
        shortest = max(1, min(first_range[1] - first_range[0], second_range[1] - second_range[0]))
        return overlap / float(shortest)

    @staticmethod
    def _remap_results(
        results: list[OCRResult],
        *,
        from_shape: tuple[int, ...],
        to_shape: tuple[int, ...],
        offset: tuple[int, int] = (0, 0),
    ) -> list[OCRResult]:
        return result_mapping_utils._remap_results(
            results,
            from_shape=from_shape,
            to_shape=to_shape,
            offset=offset,
        )

    @staticmethod
    def _sort_results(results: list[OCRResult]) -> list[OCRResult]:
        return result_mapping_utils._sort_results(results)
