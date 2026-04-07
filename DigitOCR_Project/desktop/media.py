"""Background-friendly image loading and preprocessing helpers for the Tk GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


DEFAULT_WORKING_MAX_SIDE = 1600
DEFAULT_RESULT_PREVIEW_SIZE = (900, 650)


@dataclass(slots=True)
class LoadedImagePayload:
    """Decoded upload image plus a preprocessed working image and preview copy."""

    path: Path
    working_image: np.ndarray
    preview_image: np.ndarray
    original_size: tuple[int, int]

    @property
    def image(self) -> np.ndarray:
        """Backward-compatible alias for the working image."""
        return self.working_image


def resize_bgr_for_preview(image: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    """Resize one BGR image to fit inside the requested preview box."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    max_width, max_height = max_size
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image.copy()

    scale = min(max_width / float(max(1, width)), max_height / float(max(1, height)))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def preprocess_upload_image(
    image: Image.Image,
    *,
    max_working_side: int = DEFAULT_WORKING_MAX_SIDE,
) -> np.ndarray:
    """Build a consistent BGR working image for downstream OCR."""
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    width, height = normalized.size
    longest_side = max(width, height)
    if longest_side > max_working_side:
        scale = max_working_side / float(longest_side)
        resized_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        normalized = normalized.resize(resized_size, Image.Resampling.LANCZOS)

    rgb_array = np.array(normalized, dtype=np.uint8)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def load_image_for_preview(
    image_path: str | Path,
    *,
    max_size: tuple[int, int],
    max_working_side: int = DEFAULT_WORKING_MAX_SIDE,
) -> LoadedImagePayload:
    """Load one image from disk and prepare both OCR and preview copies."""
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            original_size = image.size
            working_image = preprocess_upload_image(image, max_working_side=max_working_side)
    except Exception as exc:
        raise ValueError(f"无法读取图片: {path}") from exc

    preview_image = resize_bgr_for_preview(working_image, max_size=max_size)
    return LoadedImagePayload(
        path=path,
        working_image=working_image,
        preview_image=preview_image,
        original_size=original_size,
    )


def prepare_result_preview_image(
    image: np.ndarray,
    *,
    max_size: tuple[int, int] = DEFAULT_RESULT_PREVIEW_SIZE,
) -> np.ndarray:
    """Build a smaller BGR preview image for the result panel."""
    return resize_bgr_for_preview(image, max_size=max_size)
