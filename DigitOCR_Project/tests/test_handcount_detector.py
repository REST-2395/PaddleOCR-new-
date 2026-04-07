from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from handcount.constants import (
    INDEX_FINGER_PIP_INDEX,
    INDEX_FINGER_TIP_INDEX,
    MIDDLE_FINGER_PIP_INDEX,
    MIDDLE_FINGER_TIP_INDEX,
    PINKY_PIP_INDEX,
    PINKY_TIP_INDEX,
    RING_FINGER_PIP_INDEX,
    RING_FINGER_TIP_INDEX,
    THUMB_MCP_INDEX,
    THUMB_TIP_INDEX,
)
from handcount.detector import HandDetector


class _HandsStub:
    def __init__(self, result: object) -> None:
        self._result = result
        self.closed = False

    def process(self, _image):
        return self._result

    def close(self) -> None:
        self.closed = True


class _HandsSolutionStub:
    HAND_CONNECTIONS = ((0, 1), (1, 2))

    def __init__(self, result: object) -> None:
        self._result = result
        self.instances: list[_HandsStub] = []

    def Hands(self, **_kwargs) -> _HandsStub:
        instance = _HandsStub(self._result)
        self.instances.append(instance)
        return instance


def _classification(label: str, score: float = 0.99) -> object:
    return SimpleNamespace(classification=[SimpleNamespace(label=label, score=score)])


def _build_landmark_list(points: dict[int, tuple[int, int]], *, image_shape: tuple[int, int, int]) -> object:
    height, width = image_shape[:2]
    raw_points = [(width // 2, height // 2) for _ in range(21)]
    for index, value in points.items():
        raw_points[index] = value
    landmarks = [
        SimpleNamespace(x=x / max(width - 1, 1), y=y / max(height - 1, 1))
        for x, y in raw_points
    ]
    return SimpleNamespace(landmark=landmarks)


class HandDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image_shape = (100, 100, 3)

    def test_right_hand_thumb_rule_and_four_finger_rule(self) -> None:
        result = SimpleNamespace(
            multi_hand_landmarks=[
                _build_landmark_list(
                    {
                        THUMB_MCP_INDEX: (40, 55),
                        THUMB_TIP_INDEX: (20, 50),
                        INDEX_FINGER_PIP_INDEX: (45, 60),
                        INDEX_FINGER_TIP_INDEX: (45, 38),
                        MIDDLE_FINGER_PIP_INDEX: (55, 60),
                        MIDDLE_FINGER_TIP_INDEX: (55, 36),
                        RING_FINGER_PIP_INDEX: (65, 60),
                        RING_FINGER_TIP_INDEX: (65, 78),
                        PINKY_PIP_INDEX: (75, 60),
                        PINKY_TIP_INDEX: (75, 80),
                    },
                    image_shape=self.image_shape,
                )
            ],
            multi_handedness=[_classification("Right", 0.91)],
        )
        detector = HandDetector(hands_solution=_HandsSolutionStub(result))

        items = detector.detect(_dummy_frame(self.image_shape))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].handedness, "Right")
        self.assertEqual(items[0].finger_states, (1, 1, 1, 0, 0))
        self.assertEqual(items[0].count, 3)

    def test_left_hand_thumb_rule_uses_mirrored_orientation(self) -> None:
        result = SimpleNamespace(
            multi_hand_landmarks=[
                _build_landmark_list(
                    {
                        THUMB_MCP_INDEX: (40, 55),
                        THUMB_TIP_INDEX: (62, 45),
                        INDEX_FINGER_PIP_INDEX: (45, 60),
                        INDEX_FINGER_TIP_INDEX: (45, 38),
                        MIDDLE_FINGER_PIP_INDEX: (55, 60),
                        MIDDLE_FINGER_TIP_INDEX: (55, 80),
                        RING_FINGER_PIP_INDEX: (65, 60),
                        RING_FINGER_TIP_INDEX: (65, 80),
                        PINKY_PIP_INDEX: (75, 60),
                        PINKY_TIP_INDEX: (75, 80),
                    },
                    image_shape=self.image_shape,
                )
            ],
            multi_handedness=[_classification("Left", 0.87)],
        )
        detector = HandDetector(hands_solution=_HandsSolutionStub(result))

        items = detector.detect(_dummy_frame(self.image_shape))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].handedness, "Left")
        self.assertEqual(items[0].finger_states, (1, 1, 0, 0, 0))
        self.assertEqual(items[0].count, 2)

    def test_detect_returns_empty_when_no_hands_are_present(self) -> None:
        result = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
        detector = HandDetector(hands_solution=_HandsSolutionStub(result))

        items = detector.detect(_dummy_frame(self.image_shape))

        self.assertEqual(items, ())

    def test_extract_hand_box_uses_pixel_landmarks(self) -> None:
        result = SimpleNamespace(
            multi_hand_landmarks=[
                _build_landmark_list(
                    {
                        THUMB_MCP_INDEX: (10, 15),
                        THUMB_TIP_INDEX: (12, 18),
                        INDEX_FINGER_TIP_INDEX: (70, 20),
                        PINKY_TIP_INDEX: (66, 85),
                    },
                    image_shape=self.image_shape,
                )
            ],
            multi_handedness=[_classification("Right")],
        )
        detector = HandDetector(hands_solution=_HandsSolutionStub(result))

        items = detector.detect(_dummy_frame(self.image_shape))

        self.assertEqual(items[0].box, (10, 15, 70, 85))


def _dummy_frame(image_shape: tuple[int, int, int]):
    import numpy as np

    return np.zeros(image_shape, dtype=np.uint8)


if __name__ == "__main__":
    unittest.main()
