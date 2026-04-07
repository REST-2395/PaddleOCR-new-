"""Image enhancement helpers for OCR pre-processing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageProcessor:
    """Apply lightweight enhancement without binarization."""

    def __init__(
        self,
        min_short_side: int = 720,
        bilateral_diameter: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        clahe_clip_limit: float = 2.0,
    ) -> None:
        self.min_short_side = min_short_side
        self.bilateral_diameter = bilateral_diameter
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe_clip_limit = clahe_clip_limit

    def enhance_path(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk and apply the enhancement pipeline."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return self.enhance(image)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Upscale, denoise, and improve local contrast for OCR."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        enhanced = self._upscale_if_needed(image)
        enhanced = cv2.bilateralFilter(
            enhanced,
            self.bilateral_diameter,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space,
        )
        enhanced = self._apply_clahe(enhanced)
        enhanced = self._sharpen(enhanced)
        return enhanced

    def _upscale_if_needed(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        short_side = min(height, width)
        if short_side >= self.min_short_side:
            return image.copy()

        scale = self.min_short_side / float(short_side)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(8, 8),
        )
        l_channel = clahe.apply(l_channel)

        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def enhance_image_for_complex_env(image_path: str | Path) -> np.ndarray:
    """Compatibility helper that mirrors the path-based enhancement workflow."""
    return ImageProcessor().enhance_path(image_path)
