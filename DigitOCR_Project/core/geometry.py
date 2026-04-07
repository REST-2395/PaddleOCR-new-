"""Shared OCR geometry helpers extracted from the service facade."""

from __future__ import annotations


RegionBox = tuple[int, int, int, int]


def _combine_boxes(boxes: list[RegionBox] | tuple[RegionBox, ...]) -> RegionBox:
    """Return the bounding box covering every box in the collection."""
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return x0, y0, x1, y1


def _expand_region_box(box: RegionBox, image_shape: tuple[int, ...], *, padding: int) -> RegionBox:
    """Expand a region box while keeping it inside the image bounds."""
    image_height, image_width = image_shape[:2]
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(image_width, x1 + padding),
        min(image_height, y1 + padding),
    )


def _region_box_to_polygon(box: RegionBox) -> list[list[int]]:
    """Convert a rectangle into the polygon format expected by the renderer."""
    x0, y0, x1, y1 = box
    return [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ]


def _polygon_to_region_box(points: list[list[int]]) -> RegionBox:
    """Convert a polygon back into a rectangular region box."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _box_width(box: RegionBox) -> int:
    return max(0, box[2] - box[0])


def _box_height(box: RegionBox) -> int:
    return max(0, box[3] - box[1])


def _box_area(box: RegionBox) -> int:
    return _box_width(box) * _box_height(box)


def _box_center(box: RegionBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _intersection_area(first_box: RegionBox, second_box: RegionBox) -> int:
    """Return the overlap area between two boxes."""
    x0 = max(first_box[0], second_box[0])
    y0 = max(first_box[1], second_box[1])
    x1 = min(first_box[2], second_box[2])
    y1 = min(first_box[3], second_box[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def _box_iou(first_box: RegionBox, second_box: RegionBox) -> float:
    """Return IoU between two boxes."""
    intersection = _intersection_area(first_box, second_box)
    if intersection <= 0:
        return 0.0

    union = _box_area(first_box) + _box_area(second_box) - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


__all__ = [
    "RegionBox",
    "_box_area",
    "_box_center",
    "_box_height",
    "_box_iou",
    "_box_width",
    "_combine_boxes",
    "_expand_region_box",
    "_intersection_area",
    "_polygon_to_region_box",
    "_region_box_to_polygon",
]
