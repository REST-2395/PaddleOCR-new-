from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from desktop.media import load_image_for_preview, prepare_result_preview_image, resize_bgr_for_preview


class GuiMediaTests(unittest.TestCase):
    def test_resize_bgr_for_preview_keeps_small_image(self) -> None:
        image = np.full((120, 200, 3), 255, dtype=np.uint8)

        preview = resize_bgr_for_preview(image, (400, 300))

        self.assertEqual(preview.shape, image.shape)
        self.assertFalse(preview is image)

    def test_resize_bgr_for_preview_scales_large_image_inside_bounds(self) -> None:
        image = np.full((1200, 1600, 3), 255, dtype=np.uint8)

        preview = resize_bgr_for_preview(image, (400, 300))

        self.assertLessEqual(preview.shape[1], 400)
        self.assertLessEqual(preview.shape[0], 300)

    def test_load_image_for_preview_reads_and_prepares_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image = np.full((720, 1080, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            payload = load_image_for_preview(image_path, max_size=(500, 300))

        self.assertEqual(payload.path.name, "sample.png")
        self.assertEqual(payload.image.shape, (720, 1080, 3))
        self.assertLessEqual(payload.preview_image.shape[1], 500)
        self.assertLessEqual(payload.preview_image.shape[0], 300)

    def test_load_image_for_preview_raises_for_missing_image(self) -> None:
        with self.assertRaises(ValueError):
            load_image_for_preview("missing-file.png", max_size=(500, 300))

    def test_prepare_result_preview_image_scales_large_result_inside_bounds(self) -> None:
        image = np.full((1200, 1800, 3), 40, dtype=np.uint8)

        preview = prepare_result_preview_image(image, max_size=(900, 650))

        self.assertEqual(preview.shape[2], 3)
        self.assertLessEqual(preview.shape[1], 900)
        self.assertLessEqual(preview.shape[0], 650)
        self.assertEqual(image.shape, (1200, 1800, 3))

    def test_prepare_result_preview_image_copies_small_image(self) -> None:
        image = np.full((240, 320, 3), 180, dtype=np.uint8)

        preview = prepare_result_preview_image(image, max_size=(900, 650))

        self.assertEqual(preview.shape, image.shape)
        self.assertFalse(preview is image)


if __name__ == "__main__":
    unittest.main()
