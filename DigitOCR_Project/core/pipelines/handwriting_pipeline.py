"""Handwriting OCR pipeline extracted from the service facade."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..ocr_engine import OCRResult
from .handwriting_blocks import HandwritingBlockBuilderMixin
from .handwriting_resolution import HandwritingResolutionMixin
from .handwriting_scoring import HandwritingScoringMixin
from .handwriting_segmentation import HandwritingSegmentationMixin


class HandwritingPipeline(
    HandwritingResolutionMixin,
    HandwritingBlockBuilderMixin,
    HandwritingScoringMixin,
    HandwritingSegmentationMixin,
):
    """Handwriting OCR pipeline that delegates shared utilities to the service facade."""

    def __init__(self, service) -> None:
        self.service = service

    def __getattr__(self, name: str):
        return getattr(self.service, name)

    @property
    def engine(self):
        return self.service.engine

    def run(
        self,
        image: np.ndarray,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[OCRResult], list[str]]:
        """Run the handwriting pipeline on one prepared BGR image."""
        self._validate_handwriting_content(image)
        return self._recognize_handwriting_regions(image, progress_callback=progress_callback)

    @staticmethod
    def _service_types():
        from .. import service_types

        return service_types


__all__ = ["HandwritingPipeline"]
