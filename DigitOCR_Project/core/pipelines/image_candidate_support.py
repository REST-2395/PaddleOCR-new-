"""Shared image-mode candidate extraction and review helpers."""

from __future__ import annotations

from .image_candidate_core import ImageCandidateCollectionMixin
from .image_candidate_review import ImageCandidateReviewMixin
from .image_candidate_segmentation import ImageCandidateSegmentationMixin


class ImageCandidateSupportMixin(
    ImageCandidateCollectionMixin,
    ImageCandidateReviewMixin,
    ImageCandidateSegmentationMixin,
):
    """Mixin with the segmentation-first image candidate workflow."""


__all__ = ["ImageCandidateSupportMixin"]
