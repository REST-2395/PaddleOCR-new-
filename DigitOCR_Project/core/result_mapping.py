"""Shared OCR result mapping helpers extracted from the service facade."""

from __future__ import annotations

import numpy as np

from .ocr_engine import OCRResult


def _sort_results(results: list[OCRResult]) -> list[OCRResult]:
    """Sort OCR results by reading order with row grouping."""
    if not results:
        return []

    def top(item: OCRResult) -> int:
        return min(point[1] for point in item.box)

    def left(item: OCRResult) -> int:
        return min(point[0] for point in item.box)

    def height(item: OCRResult) -> int:
        ys = [point[1] for point in item.box]
        return max(ys) - min(ys)

    ordered_candidates = sorted(results, key=lambda item: (top(item), left(item)))
    median_height = int(np.median([max(1, height(item)) for item in ordered_candidates]))
    row_threshold = max(10, int(median_height * 0.45))

    rows: list[dict[str, object]] = []
    for item in ordered_candidates:
        center_y = float(np.mean([point[1] for point in item.box]))
        target_row: dict[str, object] | None = None

        for row in rows:
            if abs(center_y - float(row["center_y"])) <= row_threshold:
                target_row = row
                break

        if target_row is None:
            rows.append({"center_y": center_y, "items": [item]})
        else:
            row_items = row["items"]
            assert isinstance(row_items, list)
            row_items.append(item)
            target_row["center_y"] = float(
                np.mean([np.mean([point[1] for point in entry.box]) for entry in row_items])
            )

    sorted_results: list[OCRResult] = []
    for row in sorted(rows, key=lambda item: float(item["center_y"])):
        row_items = sorted(row["items"], key=left)
        sorted_results.extend(row_items)

    return sorted_results


def _remap_results(
    results: list[OCRResult],
    *,
    from_shape: tuple[int, ...],
    to_shape: tuple[int, ...],
    offset: tuple[int, int] = (0, 0),
) -> list[OCRResult]:
    """Map OCR result boxes between differently sized images."""
    if not results:
        return []

    from_height, from_width = from_shape[:2]
    to_height, to_width = to_shape[:2]
    scale_x = to_width / float(max(1, from_width))
    scale_y = to_height / float(max(1, from_height))
    offset_x, offset_y = offset

    mapped_results: list[OCRResult] = []
    for result in results:
        mapped_box = [
            [
                int(round(point[0] * scale_x + offset_x)),
                int(round(point[1] * scale_y + offset_y)),
            ]
            for point in result.box
        ]
        mapped_results.append(
            OCRResult(
                text=result.text,
                score=result.score,
                box=mapped_box,
            )
        )

    return _sort_results(mapped_results)


__all__ = ["_remap_results", "_sort_results"]
