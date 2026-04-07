"""Image- and handwriting-oriented helper methods for the service facade."""

from __future__ import annotations

import cv2
import numpy as np

from .messages import append_unique_warning
from .ocr_engine import OCRResult, TextOnlyResult
from .service_types import RegionBox


class ServiceImageUtilityMixin:
    """Hold service helpers that support the split pipelines."""

    def _should_retry_image_results(self, results: list[OCRResult]) -> bool:
        if not results:
            return True
        return self._image_result_max_score(results) < self.image_rotation_retry_score

    def _should_retry_full_image_rotation(
        self,
        image: np.ndarray,
        results: list[OCRResult],
    ) -> bool:
        del results
        candidate_boxes = self._extract_image_candidate_boxes(image)
        return len(candidate_boxes) < 2

    @staticmethod
    def _text_result_quality(result: TextOnlyResult) -> tuple[bool, float, int]:
        return bool(result.text), float(result.score), len(result.text)

    @staticmethod
    def _is_better_image_result_set(candidate: list[OCRResult], current: list[OCRResult]) -> bool:
        return ServiceImageUtilityMixin._image_results_quality(candidate) > ServiceImageUtilityMixin._image_results_quality(
            current
        )

    @staticmethod
    def _image_results_quality(results: list[OCRResult]) -> tuple[bool, float, int, int]:
        if not results:
            return False, 0.0, 0, 0
        return (
            True,
            ServiceImageUtilityMixin._image_result_max_score(results),
            len(results),
            sum(len(result.text) for result in results),
        )

    @staticmethod
    def _image_result_max_score(results: list[OCRResult]) -> float:
        if not results:
            return 0.0
        return max(float(result.score) for result in results)

    def _sort_region_boxes(self, boxes: list[RegionBox]) -> list[RegionBox]:
        if not boxes:
            return []
        temp_results = [OCRResult(text="", score=0.0, box=self._region_box_to_polygon(box)) for box in boxes]
        sorted_results = self._sort_results(temp_results)
        return [self._polygon_to_region_box(result.box) for result in sorted_results]

    def _rotate_image(self, image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self._ensure_bgr(image)
        height, width = source.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        rotated_width = max(1, int(round((height * sin) + (width * cos))))
        rotated_height = max(1, int(round((height * cos) + (width * sin))))
        matrix[0, 2] += (rotated_width / 2.0) - center[0]
        matrix[1, 2] += (rotated_height / 2.0) - center[1]
        rotated = cv2.warpAffine(
            source,
            matrix,
            (rotated_width, rotated_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        inverse_matrix = cv2.invertAffineTransform(matrix)
        return rotated, matrix, inverse_matrix

    @staticmethod
    def _map_results_with_affine(
        results: list[OCRResult],
        matrix: np.ndarray,
        *,
        to_shape: tuple[int, ...],
    ) -> list[OCRResult]:
        if not results:
            return []
        target_height, target_width = to_shape[:2]
        mapped_results: list[OCRResult] = []
        for result in results:
            points = np.array(result.box, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.transform(points, matrix).reshape(-1, 2)
            mapped_box = [
                [
                    int(round(np.clip(point[0], 0, max(0, target_width - 1)))),
                    int(round(np.clip(point[1], 0, max(0, target_height - 1)))),
                ]
                for point in transformed
            ]
            mapped_results.append(OCRResult(text=result.text, score=result.score, box=mapped_box))
        return ServiceImageUtilityMixin._sort_results(mapped_results)

    @staticmethod
    def _warp_polygon_crop(
        image: np.ndarray,
        polygon: list[list[int]],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if len(polygon) < 4:
            return None, None
        ordered = ServiceImageUtilityMixin._order_quad_points(np.array(polygon[:4], dtype=np.float32))
        top_width = np.linalg.norm(ordered[1] - ordered[0])
        bottom_width = np.linalg.norm(ordered[2] - ordered[3])
        left_height = np.linalg.norm(ordered[3] - ordered[0])
        right_height = np.linalg.norm(ordered[2] - ordered[1])
        warped_width = max(1, int(round(max(top_width, bottom_width))))
        warped_height = max(1, int(round(max(left_height, right_height))))
        destination = np.array(
            [
                [0.0, 0.0],
                [warped_width - 1.0, 0.0],
                [warped_width - 1.0, warped_height - 1.0],
                [0.0, warped_height - 1.0],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, destination)
        inverse_matrix = cv2.getPerspectiveTransform(destination, ordered)
        warped = cv2.warpPerspective(
            ServiceImageUtilityMixin._ensure_bgr(image),
            matrix,
            (warped_width, warped_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return warped, inverse_matrix

    @staticmethod
    def _map_polygon_with_perspective(
        polygon: list[list[int]],
        matrix: np.ndarray,
        *,
        to_shape: tuple[int, ...],
    ) -> list[list[int]]:
        target_height, target_width = to_shape[:2]
        points = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        return [
            [
                int(round(np.clip(point[0], 0, max(0, target_width - 1)))),
                int(round(np.clip(point[1], 0, max(0, target_height - 1)))),
            ]
            for point in transformed
        ]

    @staticmethod
    def _order_quad_points(points: np.ndarray) -> np.ndarray:
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        return np.array(
            [
                points[np.argmin(sums)],
                points[np.argmin(diffs)],
                points[np.argmax(sums)],
                points[np.argmax(diffs)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _pad_image_border(image: np.ndarray, padding: int) -> np.ndarray:
        return cv2.copyMakeBorder(
            image,
            padding,
            padding,
            padding,
            padding,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    @staticmethod
    def _add_warning(warnings: list[str], message: str) -> None:
        append_unique_warning(warnings, message)

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Canvas image is empty.")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _validate_handwriting_content(self, image: np.ndarray) -> None:
        self._get_handwriting_pipeline()._validate_handwriting_content(image)

    @staticmethod
    def _segmentation_kernel_size(image_shape: tuple[int, ...]) -> int:
        kernel_size = max(3, int(round(min(image_shape[:2]) * 0.008)))
        kernel_size = min(kernel_size, 7)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    @staticmethod
    def _image_adaptive_block_size(image_shape: tuple[int, ...]) -> int:
        block_size = max(15, int(round(min(image_shape[:2]) * 0.14)))
        block_size = min(block_size, 61)
        if block_size % 2 == 0:
            block_size += 1
        return block_size

    @staticmethod
    def _image_mask_kernel_size(image_shape: tuple[int, ...]) -> int:
        kernel_size = max(1, int(round(min(image_shape[:2]) * 0.01)))
        kernel_size = min(kernel_size, 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    @staticmethod
    def _resize_to_min_side(image: np.ndarray, *, min_short_side: int) -> np.ndarray:
        height, width = image.shape[:2]
        short_side = min(height, width)
        if short_side >= min_short_side:
            return image.copy()
        scale = min_short_side / float(max(1, short_side))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
