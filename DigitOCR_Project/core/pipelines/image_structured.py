"""Structured-photo helpers for the image OCR pipeline."""

from __future__ import annotations

from .image_structured_panel import StructuredPanelMixin
from .image_structured_sequence import StructuredSequenceMixin


class StructuredImageMixin(StructuredSequenceMixin, StructuredPanelMixin):
    """Mixin with the structured multi-digit photo route."""


__all__ = ["StructuredImageMixin"]
