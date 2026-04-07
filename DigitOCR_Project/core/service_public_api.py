"""Public service entrypoints extracted from the service facade."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .ocr_engine import OCRResult
from .service_types import RecognitionOutput


class ServicePublicApiMixin:
    """Keep the service facade thin while preserving public entrypoints."""

    def recognize_image(
        self,
        image: np.ndarray,
        *,
        source_name: str = "Image",
        annotate_on_original: bool = True,
    ) -> RecognitionOutput:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")
        input_image = image.copy()
        processed_image = self.processor.enhance(input_image)
        recognized_results = self.engine.recognize(processed_image)
        best_processed_results, warnings = self._get_image_pipeline().run(
            processed_image,
            recognized_results,
        )
        if annotate_on_original:
            results = self._remap_results(
                best_processed_results,
                from_shape=processed_image.shape,
                to_shape=input_image.shape,
            )
            base_image = input_image
        else:
            results = best_processed_results
            base_image = processed_image
        annotated_image = self.engine.draw_results(base_image, results)
        return RecognitionOutput(
            source_name=source_name,
            input_image=input_image,
            processed_image=processed_image,
            annotated_image=annotated_image,
            results=results,
            warnings=warnings,
        )

    def recognize_camera_frame(
        self,
        image: np.ndarray,
        *,
        source_name: str = "Camera",
        allow_fallback: bool = True,
    ) -> list[OCRResult]:
        results, _ = self._recognize_camera_frame_internal(
            image,
            source_name=source_name,
            allow_fallback=allow_fallback,
        )
        return results

    def recognize_board_frame(
        self,
        image: np.ndarray,
        *,
        source_name: str = "Blackboard",
        return_warnings: bool = False,
    ) -> list[OCRResult] | tuple[list[OCRResult], list[str]]:
        del source_name
        return self._get_board_sequence_pipeline().run(image, return_warnings=return_warnings)

    def _recognize_camera_frame_internal(
        self,
        image: np.ndarray,
        *,
        source_name: str = "Camera",
        allow_fallback: bool = True,
    ) -> tuple[list[OCRResult], bool]:
        del source_name
        input_image = self._ensure_bgr(image)
        processor = getattr(self, "camera_processor", self.processor)
        processed_image = processor.enhance(input_image)
        selected_results, fallback_used = self._get_camera_digit_pipeline().run(
            processed_image,
            allow_fallback=allow_fallback,
        )
        return self._remap_results(
            selected_results,
            from_shape=processed_image.shape,
            to_shape=input_image.shape,
        ), fallback_used

    def recognize_image_path(self, image_path: str | Path) -> RecognitionOutput:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        return self.recognize_image(image, source_name=path.name, annotate_on_original=True)

    def recognize_handwriting(
        self,
        canvas_image: np.ndarray,
        *,
        source_name: str = "Handwriting",
        progress_callback: Callable[[str], None] | None = None,
    ) -> RecognitionOutput:
        input_image = self._ensure_bgr(canvas_image)
        processed_image = input_image.copy()
        results, warnings = self._get_handwriting_pipeline().run(
            input_image,
            progress_callback=progress_callback,
        )
        annotated_image = self.engine.draw_results(input_image, results)
        return RecognitionOutput(
            source_name=source_name,
            input_image=input_image,
            processed_image=processed_image,
            annotated_image=annotated_image,
            results=results,
            warnings=warnings,
        )
