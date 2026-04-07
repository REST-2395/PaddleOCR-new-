"""Shared OCR result text, warnings, and summary formatting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ocr_engine import OCRResult


NO_DIGITS_FOUND_TEXT = "未识别到数字"
HANDWRITING_WARNING_TEXT = "检测到连写数字，请尽量分开书写。"
IMAGE_WARNING_TEXT = "检测到多位数字区域，请尽量分开排布。"
BOARD_WARNING_TEXT = "检测到黑板模式下的复杂数字区域，请尽量保持数字清晰并分开。"


def coalesce_result_text(text: str | None) -> str:
    return text or NO_DIGITS_FOUND_TEXT


def format_result_summary(results: Sequence["OCRResult"]) -> str:
    if not results:
        return NO_DIGITS_FOUND_TEXT
    return ", ".join(f"{item.text} ({item.score:.2f})" for item in results)


def format_recognition_status(
    source_name: str,
    detail: str,
    warnings: Sequence[str],
    *,
    has_results: bool,
) -> str:
    if warnings:
        warning_text = "; ".join(warnings)
        if has_results:
            return f"{source_name}: {detail}; {warning_text}"
        return f"{source_name}: {warning_text}"
    return f"{source_name}: {detail}"


def append_unique_warning(warnings: list[str], message: str) -> None:
    if message and message not in warnings:
        warnings.append(message)


__all__ = [
    "BOARD_WARNING_TEXT",
    "HANDWRITING_WARNING_TEXT",
    "IMAGE_WARNING_TEXT",
    "NO_DIGITS_FOUND_TEXT",
    "append_unique_warning",
    "coalesce_result_text",
    "format_recognition_status",
    "format_result_summary",
]
