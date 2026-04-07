"""Board-sequence OCR pipeline extracted from the service facade."""

from __future__ import annotations

import numpy as np

from ..ocr_engine import OCRResult


class BoardSequencePipeline:
    """Board-mode OCR pipeline that keeps multi-character sequence outputs intact."""

    def __init__(self, service) -> None:
        self.service = service

    def __getattr__(self, name: str):
        return getattr(self.service, name)

    @property
    def processor(self):
        return self.service.processor

    @property
    def camera_processor(self):
        return self.service.camera_processor

    @property
    def engine(self):
        return self.service.engine

    def run(
        self,
        image: np.ndarray,
        *,
        return_warnings: bool = False,
    ) -> list[OCRResult] | tuple[list[OCRResult], list[str]]:
        """Recognize digit sequences from one pre-cropped blackboard frame."""
        input_image = self._ensure_bgr(image)
        processor = getattr(self, "camera_processor", self.processor)
        processed_image = processor.enhance(input_image)
        recognized_results = self.engine.recognize(processed_image)
        remapped_results = self._remap_results(
            self._sort_results(recognized_results),
            from_shape=processed_image.shape,
            to_shape=input_image.shape,
        )
        warnings: list[str] = []
        if not remapped_results:
            self._add_warning(warnings, self.board_warning_text)
        if return_warnings:
            return remapped_results, warnings
        return remapped_results


__all__ = ["BoardSequencePipeline"]
