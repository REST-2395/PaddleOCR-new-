"""Shared OCR service value objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ocr_engine import OCRResult

RegionBox = tuple[int, int, int, int]


@dataclass(slots=True)
class RecognitionOutput:
    """Container for one end-to-end OCR run."""

    source_name: str
    input_image: np.ndarray
    processed_image: np.ndarray
    annotated_image: np.ndarray
    results: list[OCRResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def combined_text(self) -> str:
        return " ".join(item.text for item in self.results)

    @property
    def summary_text(self) -> str:
        from .messages import format_result_summary

        return format_result_summary(self.results)


@dataclass(slots=True)
class HandwritingBlock:
    """Prepared OCR inputs for one segmented handwriting region."""

    display_box: RegionBox
    region_image: np.ndarray
    primary_image: np.ndarray
    fallback_image: np.ndarray


@dataclass(slots=True)
class ImageCandidate:
    """One image-mode OCR candidate gathered from OCR or segmentation."""

    display_box: RegionBox
    text_hint: str = ""
    score_hint: float = 0.0
    sources: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ImageCandidateBlock:
    """Prepared crop and foreground mask for one image-mode candidate."""

    display_box: RegionBox
    region_image: np.ndarray
    foreground_mask: np.ndarray


@dataclass(slots=True)
class ImageReviewResult:
    """Aggregated single-digit prediction for one image-mode candidate."""

    text: str
    score: float
    support: int
    attempts: int
    runner_up_support: int = 0
    runner_up_score: float = 0.0


@dataclass(slots=True)
class StructuredSequence:
    """One structured multi-digit OCR region that should be split into single digits."""

    polygon: list[list[int]]
    display_box: RegionBox
    text_hint: str
    score_hint: float


__all__ = [
    "HandwritingBlock",
    "ImageCandidate",
    "ImageCandidateBlock",
    "ImageReviewResult",
    "RecognitionOutput",
    "RegionBox",
    "StructuredSequence",
]
