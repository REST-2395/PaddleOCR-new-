"""Resolution helpers for the handwriting pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..ocr_engine import OCRResult, TextOnlyResult


class HandwritingResolutionMixin:
    """Resolve handwriting blocks into OCR results."""

    def _recognize_handwriting_regions(
        self,
        image: np.ndarray,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[OCRResult], list[str]]:
        blocks = self._build_handwriting_blocks(image)
        if not blocks:
            return [], []

        warnings: list[str] = []
        results = self._resolve_handwriting_blocks(
            blocks,
            warnings=warnings,
            progress_callback=progress_callback,
            depth=0,
        )
        return self._sort_results(results), warnings

    def _resolve_handwriting_blocks(
        self,
        blocks: list["HandwritingBlock"],
        *,
        warnings: list[str],
        progress_callback: Callable[[str], None] | None,
        depth: int,
    ) -> list[OCRResult]:
        if not blocks:
            return []

        if depth > self.handwriting_split_depth_limit:
            self._add_warning(warnings, self.handwriting_warning_text)
            return []

        selected_results = self._recognize_handwriting_block_texts(
            blocks,
            progress_callback=progress_callback if depth == 0 else None,
        )

        resolved_results: list[OCRResult] = []
        for block, recognized in zip(blocks, selected_results):
            if self._is_single_digit_result(recognized):
                resolved_results.append(
                    OCRResult(
                        text=recognized.text,
                        score=recognized.score,
                        box=self._region_box_to_polygon(block.display_box),
                    )
                )
                continue

            if len(recognized.text) > 1:
                child_blocks = self._split_handwriting_block(block, char_hint=len(recognized.text))
                if child_blocks:
                    child_results = self._resolve_handwriting_blocks(
                        child_blocks,
                        warnings=warnings,
                        progress_callback=None,
                        depth=depth + 1,
                    )
                    if child_results:
                        resolved_results.extend(child_results)
                    else:
                        self._add_warning(warnings, self.handwriting_warning_text)
                    continue

                self._add_warning(warnings, self.handwriting_warning_text)
                continue

            if depth > 0:
                self._add_warning(warnings, self.handwriting_warning_text)

        return resolved_results

    def _recognize_handwriting_block_texts(
        self,
        blocks: list["HandwritingBlock"],
        *,
        progress_callback: Callable[[str], None] | None,
    ) -> list[TextOnlyResult]:
        primary_results = self.engine.recognize_handwriting_blocks([block.primary_image for block in blocks])
        selected_results = primary_results[:]

        fallback_indices = [
            index for index, result in enumerate(primary_results) if self._should_retry_handwriting_result(result)
        ]
        if fallback_indices:
            if progress_callback is not None:
                progress_callback("正在重试低置信度手写块...")

            fallback_results = self.engine.recognize_handwriting_blocks(
                [blocks[index].fallback_image for index in fallback_indices]
            )
            for block_index, fallback_result in zip(fallback_indices, fallback_results):
                if self._is_usable_handwriting_result(fallback_result):
                    selected_results[block_index] = fallback_result

        rotation_retry_indices = [
            index for index, result in enumerate(selected_results) if self._should_retry_rotated_handwriting_result(result)
        ]
        if rotation_retry_indices:
            if progress_callback is not None:
                progress_callback("正在纠正旋转手写块...")

            retry_images: list[np.ndarray] = []
            retry_owners: list[int] = []
            for block_index in rotation_retry_indices:
                for candidate_image in self._build_rotated_handwriting_candidate_images(blocks[block_index]):
                    retry_images.append(candidate_image)
                    retry_owners.append(block_index)

            retry_results = self.engine.recognize_handwriting_blocks(retry_images)
            grouped_results: dict[int, list[TextOnlyResult]] = {}
            for block_index, retry_result in zip(retry_owners, retry_results):
                grouped_results.setdefault(block_index, []).append(retry_result)

            for block_index in rotation_retry_indices:
                best_result = selected_results[block_index]
                for candidate_result in grouped_results.get(block_index, []):
                    if self._is_better_handwriting_result(candidate_result, best_result):
                        best_result = candidate_result
                selected_results[block_index] = best_result

        return selected_results

    def _build_rotated_handwriting_candidate_images(self, block: "HandwritingBlock") -> list[np.ndarray]:
        candidate_images: list[np.ndarray] = []
        estimated_angle = self._estimate_foreground_angle(block.region_image)
        for angle in self._build_rotation_retry_angles(estimated_angle):
            rotated_image, _, _ = self._rotate_image(block.region_image, angle)
            candidate_images.append(self._build_rotation_preview_image(rotated_image))
        return candidate_images
