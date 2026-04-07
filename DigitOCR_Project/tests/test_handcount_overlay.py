from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handcount.overlay import overlay_hand_count_frame
from handcount.types import HandCountItem, HandCountPayload, HandLandmarkPoint


class HandCountOverlayTests(unittest.TestCase):
    def test_overlay_uses_ascii_labels_for_detected_hands(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        payload = HandCountPayload(
            items=(_make_hand("Right", 2),),
            total_count=2,
            too_many_hands=False,
            fps=12.0,
            warnings=(),
        )
        texts: list[str] = []

        def capture_text(image, text, org, font_face, font_scale, color, thickness=1, lineType=None, bottomLeftOrigin=False):
            del org, font_face, font_scale, color, thickness, lineType, bottomLeftOrigin
            texts.append(text)
            return image

        with patch("handcount.overlay.cv2.putText", side_effect=capture_text):
            overlay_hand_count_frame(
                frame,
                payload,
                capture_fps=18.5,
                count_fps=11.2,
                prompt_text="ignored",
                roi_width_ratio=0.7,
                roi_height_ratio=0.5,
            )

        self.assertTrue(texts)
        self.assertTrue(all(text.isascii() for text in texts))

    def test_overlay_uses_ascii_warning_for_empty_frame(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        payload = HandCountPayload(
            items=(),
            total_count=0,
            too_many_hands=False,
            fps=12.0,
            warnings=(),
        )
        texts: list[str] = []

        def capture_text(image, text, org, font_face, font_scale, color, thickness=1, lineType=None, bottomLeftOrigin=False):
            del org, font_face, font_scale, color, thickness, lineType, bottomLeftOrigin
            texts.append(text)
            return image

        with patch("handcount.overlay.cv2.putText", side_effect=capture_text):
            overlay_hand_count_frame(
                frame,
                payload,
                capture_fps=18.5,
                count_fps=0.0,
                prompt_text="ignored",
                roi_width_ratio=0.7,
                roi_height_ratio=0.5,
            )

        self.assertIn("No hands", texts)
        self.assertTrue(all(text.isascii() for text in texts))


def _make_hand(
    handedness: str,
    count: int,
    *,
    box: tuple[int, int, int, int] = (30, 30, 70, 70),
    score: float = 0.95,
) -> HandCountItem:
    finger_states = tuple(1 if index < count else 0 for index in range(5))
    landmarks = tuple(HandLandmarkPoint(x=box[0] + 5 + index, y=box[1] + 5 + index) for index in range(21))
    return HandCountItem(
        handedness=handedness,  # type: ignore[arg-type]
        score=score,
        finger_states=finger_states,  # type: ignore[arg-type]
        count=count,
        box=box,
        landmarks=landmarks,
    )


if __name__ == "__main__":
    unittest.main()
