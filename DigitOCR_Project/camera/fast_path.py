"""Digit-mode fast-path helpers for live camera OCR."""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Sequence

import cv2
import numpy as np

from . import config
from .protocol import CameraTrack
from .roi import crop_camera_roi
from .state import CameraBox, CameraDetection, sort_camera_detections
from core.ocr_engine import OCRResult


def extract_camera_fast_candidates(
    frame_bgr: np.ndarray,
    *,
    width_ratio: float = config.CAMERA_ROI_WIDTH_RATIO,
    height_ratio: float = config.CAMERA_ROI_HEIGHT_RATIO,
) -> tuple[tuple[np.ndarray, ...], tuple[CameraBox, ...]]:
    """Extract a small set of likely digit crops from the current ROI."""
    roi_frame, roi_box = crop_camera_roi(frame_bgr, width_ratio=width_ratio, height_ratio=height_ratio)
    x_offset, y_offset, _, _ = roi_box
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    candidate_boxes: list[CameraBox] = []
    for invert in (False, True):
        source = 255 - gray if invert else gray
        adaptive = cv2.adaptiveThreshold(
            source,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            7,
        )
        _, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_or(adaptive, otsu)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidate_boxes.extend(_extract_fast_component_boxes(mask, roi_box=roi_box))

    deduped_boxes = _dedupe_camera_boxes(candidate_boxes)
    expanded_boxes: list[CameraBox] = []
    for box in deduped_boxes:
        expanded_boxes.extend(_split_camera_fast_box(roi_frame, box, roi_offset=(x_offset, y_offset)))

    ordered_boxes = _order_camera_boxes(_dedupe_camera_boxes(expanded_boxes))
    limited_boxes = ordered_boxes[: config.CAMERA_FAST_MAX_CANDIDATES + 1]
    crops = tuple(frame_bgr[y0:y1, x0:x1].copy() for x0, y0, x1, y1 in limited_boxes)
    return crops, tuple(limited_boxes)


def _extract_fast_component_boxes(mask: np.ndarray, *, roi_box: CameraBox) -> list[CameraBox]:
    roi_height, roi_width = mask.shape[:2]
    min_area = max(16, int(round(roi_height * roi_width * config.CAMERA_FAST_MIN_BOX_AREA_RATIO)))
    min_height = max(14, int(round(roi_height * config.CAMERA_FAST_MIN_BOX_HEIGHT_RATIO)))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes: list[CameraBox] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or height < min_height or width < 6:
            continue
        if width / float(max(1, height)) > 5.0:
            continue
        x0 = roi_box[0] + x
        y0 = roi_box[1] + y
        boxes.append((x0, y0, x0 + width, y0 + height))
    return boxes


def _split_camera_fast_box(
    roi_frame: np.ndarray,
    box: CameraBox,
    *,
    roi_offset: tuple[int, int],
) -> list[CameraBox]:
    x_offset, y_offset = roi_offset
    local_box = (
        max(0, box[0] - x_offset),
        max(0, box[1] - y_offset),
        max(0, box[2] - x_offset),
        max(0, box[3] - y_offset),
    )
    x0, y0, x1, y1 = local_box
    crop = roi_frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if width / float(height) < 1.15:
        return [box]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    column_projection = np.sum(binary > 0, axis=0)
    valley_threshold = max(1, int(round(np.max(column_projection) * 0.16)))
    zero_columns = np.where(column_projection <= valley_threshold)[0]
    if zero_columns.size == 0:
        return [box]

    split_points: list[int] = []
    start = zero_columns[0]
    previous = zero_columns[0]
    for value in zero_columns[1:]:
        if value != previous + 1:
            split_points.append((start + previous) // 2)
            start = value
        previous = value
    split_points.append((start + previous) // 2)

    boundaries = [0, *[point for point in split_points if 8 <= point <= width - 8], width]
    boundaries = sorted(set(boundaries))
    if len(boundaries) <= 2:
        return [box]

    sub_boxes: list[CameraBox] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < 8:
            continue
        sub_boxes.append((box[0] + left, box[1], box[0] + right, box[3]))
    if 1 < len(sub_boxes) <= config.CAMERA_FAST_MAX_CANDIDATES:
        compacted_boxes: list[CameraBox] = []
        for candidate_box in sorted(sub_boxes, key=_camera_box_area, reverse=True):
            if any(_camera_boxes_equivalent(candidate_box, existing_box) for existing_box in compacted_boxes):
                continue
            compacted_boxes.append(candidate_box)
        if len(compacted_boxes) > 1:
            return sorted(compacted_boxes, key=lambda current_box: current_box[0])
    return [box]


def build_camera_detections_from_results(
    results: Sequence[OCRResult],
    *,
    frame_shape: tuple[int, ...],
    allow_multi_char: bool = False,
) -> tuple[CameraDetection, ...]:
    """Convert OCR polygons into sorted axis-aligned live-camera detections."""
    detections: list[CameraDetection] = []
    for result in results:
        if not allow_multi_char and len(result.text) != 1:
            continue

        box = _result_to_camera_box(result, frame_shape)
        if box is None:
            continue

        detections.append(
            CameraDetection(
                text=result.text,
                score=float(result.score),
                box=box,
            )
        )

    return sort_camera_detections(detections)


def filter_camera_detections(
    detections: Sequence[CameraDetection],
) -> tuple[tuple[CameraDetection, ...], dict[str, float | int | str]]:
    """Filter low-confidence camera detections using per-frame and per-box thresholds."""
    if not detections:
        return (), {
            "reason": "no_candidates",
            "average_confidence": 0.0,
            "total_count": 0,
            "visible_count": 0,
            "hidden_count": 0,
        }

    average_confidence = sum(item.score for item in detections) / float(len(detections))
    if average_confidence < config.CAMERA_MIN_AVERAGE_CONFIDENCE:
        return (), {
            "reason": "average_low",
            "average_confidence": average_confidence,
            "total_count": len(detections),
            "visible_count": 0,
            "hidden_count": len(detections),
        }

    visible_items = tuple(item for item in detections if item.score >= config.CAMERA_MIN_CONFIDENCE)
    hidden_count = len(detections) - len(visible_items)
    return visible_items, {
        "reason": "ok" if visible_items else "all_low",
        "average_confidence": average_confidence,
        "total_count": len(detections),
        "visible_count": len(visible_items),
        "hidden_count": hidden_count,
    }


def stabilize_camera_detections(
    detections: Sequence[CameraDetection],
    tracks: Sequence[CameraTrack],
    *,
    next_track_id: int,
    allow_missed_tracks: bool = True,
) -> tuple[tuple[CameraDetection, ...], list[CameraTrack], int]:
    """Match current detections to previous tracks and smooth their boxes."""
    if not tracks and not detections:
        return (), [], next_track_id

    tracks_by_id = {track.track_id: track for track in tracks}
    available_track_ids = set(tracks_by_id)
    updated_tracks: list[CameraTrack] = []
    stabilized_items: list[CameraDetection] = []

    for detection in sort_camera_detections(tuple(detections)):
        matched_track_id = _match_camera_track(detection, tracks_by_id, available_track_ids)
        if matched_track_id is None:
            track_id = next_track_id
            next_track_id += 1
            stabilized_detection = detection
        else:
            previous_track = tracks_by_id[matched_track_id]
            stabilized_detection = _blend_camera_detection(previous_track.detection, detection)
            available_track_ids.remove(matched_track_id)
            track_id = matched_track_id

        updated_tracks.append(CameraTrack(track_id=track_id, detection=stabilized_detection, misses=0))
        stabilized_items.append(stabilized_detection)

    if allow_missed_tracks:
        for track_id in available_track_ids:
            previous_track = tracks_by_id[track_id]
            misses = previous_track.misses + 1
            if misses > config.CAMERA_TRACK_MAX_MISSES:
                continue

            updated_tracks.append(
                CameraTrack(
                    track_id=track_id,
                    detection=previous_track.detection,
                    misses=misses,
                )
            )
            stabilized_items.append(previous_track.detection)

    stabilized_items = list(sort_camera_detections(tuple(stabilized_items)))
    return tuple(stabilized_items), updated_tracks, next_track_id


def stable_camera_sequence(
    history: deque[str],
    current_sequence: str,
) -> str:
    """Return a stable sequence once it appears enough times in recent history."""
    if not history:
        return current_sequence

    counter = Counter(history)
    stable_sequence, count = counter.most_common(1)[0]
    if count >= config.CAMERA_STABLE_MIN_COUNT:
        return stable_sequence
    return current_sequence


def camera_detection_signature(
    stable_sequence: str,
    detections: Sequence[CameraDetection],
) -> tuple[object, ...]:
    """Build a coarse signature so the GUI can skip redundant redraws."""
    grid = max(1, config.CAMERA_SIGNATURE_GRID)
    return (
        stable_sequence,
        tuple(
            (
                item.text,
                round(item.score, 2),
                *(int(round(coordinate / grid)) for coordinate in item.box),
            )
            for item in detections
        ),
    )


def camera_result_is_fresh(
    completed_at: float,
    *,
    now: float | None = None,
    max_age_seconds: float = config.CAMERA_OVERLAY_MAX_AGE_SECONDS,
) -> bool:
    """Return whether one camera OCR result is still fresh enough to overlay."""
    if completed_at <= 0.0:
        return False

    current_time = time.perf_counter() if now is None else now
    return (current_time - completed_at) <= max(0.0, max_age_seconds)


def _scale_ocr_results(
    results: Sequence[OCRResult],
    *,
    scale_x: float,
    scale_y: float,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[OCRResult]:
    if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6 and offset_x == 0 and offset_y == 0:
        return [OCRResult(text=result.text, score=result.score, box=[point[:] for point in result.box]) for result in results]

    scaled_results: list[OCRResult] = []
    for result in results:
        scaled_box = [
            [
                int(round(point[0] * scale_x + offset_x)),
                int(round(point[1] * scale_y + offset_y)),
            ]
            for point in result.box
        ]
        scaled_results.append(OCRResult(text=result.text, score=result.score, box=scaled_box))
    return scaled_results


def _map_perspective_results(
    results: Sequence[OCRResult],
    *,
    inverse_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    to_shape: tuple[int, ...],
) -> list[OCRResult]:
    if not results:
        return []

    target_height, target_width = to_shape[:2]
    matrix = np.array(inverse_matrix, dtype=np.float32)
    mapped_results: list[OCRResult] = []
    for result in results:
        points = np.array(result.box, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        mapped_box = [
            [
                int(round(np.clip(point[0], 0, max(0, target_width - 1)))),
                int(round(np.clip(point[1], 0, max(0, target_height - 1)))),
            ]
            for point in transformed
        ]
        mapped_results.append(OCRResult(text=result.text, score=result.score, box=mapped_box))
    return mapped_results


def _order_camera_boxes(boxes: list[CameraBox]) -> list[CameraBox]:
    if not boxes:
        return []
    center_x = float(np.mean([(box[0] + box[2]) / 2.0 for box in boxes]))
    center_y = float(np.mean([(box[1] + box[3]) / 2.0 for box in boxes]))
    return sorted(
        boxes,
        key=lambda box: (
            -((box[2] - box[0]) * (box[3] - box[1])),
            abs(((box[0] + box[2]) / 2.0) - center_x) + abs(((box[1] + box[3]) / 2.0) - center_y),
            box[1],
            box[0],
        ),
    )


def _dedupe_camera_boxes(boxes: Sequence[CameraBox]) -> list[CameraBox]:
    unique_boxes: list[CameraBox] = []
    for box in boxes:
        if any(_camera_boxes_equivalent(box, existing) for existing in unique_boxes):
            continue
        unique_boxes.append(box)
    return unique_boxes


def _result_to_camera_box(result: OCRResult, frame_shape: tuple[int, ...]) -> CameraBox | None:
    xs = [point[0] for point in result.box]
    ys = [point[1] for point in result.box]
    box = (
        max(0, min(xs)),
        max(0, min(ys)),
        min(frame_shape[1] - 1, max(xs)),
        min(frame_shape[0] - 1, max(ys)),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


def _bbox_iou(first_box: CameraBox, second_box: CameraBox) -> float:
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    first_area = max(1, (first_box[2] - first_box[0]) * (first_box[3] - first_box[1]))
    second_area = max(1, (second_box[2] - second_box[0]) * (second_box[3] - second_box[1]))
    union = first_area + second_area - intersection
    return intersection / float(max(union, 1))


def _camera_box_area(box: CameraBox) -> int:
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _bbox_overlap_ratio(first_box: CameraBox, second_box: CameraBox) -> float:
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    smaller_area = min(_camera_box_area(first_box), _camera_box_area(second_box))
    return intersection / float(max(smaller_area, 1))


def _camera_boxes_equivalent(first_box: CameraBox, second_box: CameraBox) -> bool:
    return _bbox_iou(first_box, second_box) >= 0.45 or _bbox_overlap_ratio(first_box, second_box) >= 0.70


def _bbox_center_distance(first_box: CameraBox, second_box: CameraBox) -> float:
    first_center = ((first_box[0] + first_box[2]) / 2.0, (first_box[1] + first_box[3]) / 2.0)
    second_center = ((second_box[0] + second_box[2]) / 2.0, (second_box[1] + second_box[3]) / 2.0)
    delta_x = first_center[0] - second_center[0]
    delta_y = first_center[1] - second_center[1]
    return float((delta_x * delta_x + delta_y * delta_y) ** 0.5)


def _blend_camera_detection(previous: CameraDetection, current: CameraDetection) -> CameraDetection:
    smoothing = min(max(config.CAMERA_TRACK_SMOOTHING, 0.0), 0.95)
    blended_box = tuple(
        int(round(previous_coord * smoothing + current_coord * (1.0 - smoothing)))
        for previous_coord, current_coord in zip(previous.box, current.box)
    )
    return CameraDetection(
        text=current.text,
        score=current.score,
        box=blended_box,  # type: ignore[arg-type]
    )


def _match_camera_track(
    detection: CameraDetection,
    tracks_by_id: dict[int, CameraTrack],
    available_track_ids: set[int],
) -> int | None:
    best_track_id: int | None = None
    best_score = float("-inf")

    for track_id in available_track_ids:
        previous_detection = tracks_by_id[track_id].detection
        iou = _bbox_iou(previous_detection.box, detection.box)
        distance = _bbox_center_distance(previous_detection.box, detection.box)
        max_span = max(
            previous_detection.box[2] - previous_detection.box[0],
            previous_detection.box[3] - previous_detection.box[1],
            detection.box[2] - detection.box[0],
            detection.box[3] - detection.box[1],
            1,
        )

        if iou < config.CAMERA_MATCH_MIN_IOU and distance > max_span * config.CAMERA_TRACK_MAX_DISTANCE_RATIO:
            continue

        score = iou - (distance / float(max_span)) * config.CAMERA_TRACK_DISTANCE_PENALTY
        if previous_detection.text == detection.text:
            score += config.CAMERA_TRACK_SAME_DIGIT_BONUS

        if score > best_score:
            best_score = score
            best_track_id = track_id

    return best_track_id


__all__ = [
    "build_camera_detections_from_results",
    "camera_detection_signature",
    "camera_result_is_fresh",
    "extract_camera_fast_candidates",
    "filter_camera_detections",
    "stable_camera_sequence",
    "stabilize_camera_detections",
]
