from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera import config
from handcount.runtime import HandCountRuntime
from handcount.types import HandCountItem, HandCountPayload, HandLandmarkPoint


class _CaptureStub:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self.released = False

    def read(self):
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


class _DetectorStub:
    def __init__(self, sequences: list[tuple[HandCountItem, ...]]) -> None:
        self._sequences = list(sequences)
        self.closed = False
        self.call_count = 0

    def detect(self, _image_bgr) -> tuple[HandCountItem, ...]:
        self.call_count += 1
        if self.call_count <= len(self._sequences):
            return self._sequences[self.call_count - 1]
        return self._sequences[-1] if self._sequences else ()

    def close(self) -> None:
        self.closed = True


class HandCountRuntimeTests(unittest.TestCase):
    def test_start_stop_and_snapshot_publish_payload(self) -> None:
        detector = _DetectorStub([(_make_hand("Right", 2),)])
        capture = _CaptureStub(np.zeros((120, 160, 3), dtype=np.uint8))
        runtime = HandCountRuntime(
            detector_factory=lambda: detector,
            inference_interval_seconds=0.01,
            startup_grace_seconds=0.0,
            idle_seconds=0.001,
        )

        runtime.start_with_capture(capture, device_index=2, backend_name="TEST")
        snapshot = _wait_for_snapshot(runtime)

        self.assertTrue(snapshot.running)
        self.assertEqual(snapshot.device_index, 2)
        self.assertEqual(snapshot.backend_name, "TEST")
        self.assertEqual(snapshot.latest_result.mode, config.CAMERA_MODE_HAND_COUNT)
        self.assertIsInstance(snapshot.latest_result.payload, HandCountPayload)
        self.assertEqual(snapshot.latest_result.payload.total_count, 2)

        runtime.stop()

        self.assertTrue(capture.released)
        self.assertTrue(detector.closed)

    def test_stable_payload_holds_latest_confirmed_total(self) -> None:
        runtime = HandCountRuntime(detector_factory=lambda: _DetectorStub([]))
        frame_shape = (100, 100, 3)
        two = (_make_hand("Left", 2),)
        four = (_make_hand("Left", 4),)

        runtime._build_payload(two, frame_shape=frame_shape, count_fps=10.0)
        runtime._build_payload(two, frame_shape=frame_shape, count_fps=10.0)
        payload, _status = runtime._build_payload(two, frame_shape=frame_shape, count_fps=10.0)

        self.assertEqual(payload.total_count, 2)
        self.assertIsNotNone(runtime._stable_payload)

        unstable_payload, _status = runtime._build_payload(four, frame_shape=frame_shape, count_fps=10.0)

        self.assertEqual(unstable_payload.total_count, 2)
        self.assertEqual(runtime._stable_payload.total_count, 2)

    def test_empty_frames_clear_stable_result_after_threshold(self) -> None:
        runtime = HandCountRuntime(detector_factory=lambda: _DetectorStub([]))
        frame_shape = (100, 100, 3)
        two = (_make_hand("Left", 2),)

        for _ in range(config.CAMERA_HAND_STABLE_MIN_COUNT):
            runtime._build_payload(two, frame_shape=frame_shape, count_fps=10.0)

        hold_payload = None
        for _ in range(max(1, config.CAMERA_HAND_EMPTY_RESET_FRAMES - 1)):
            hold_payload, _status = runtime._build_payload((), frame_shape=frame_shape, count_fps=10.0)
        cleared_payload, _status = runtime._build_payload((), frame_shape=frame_shape, count_fps=10.0)

        self.assertIsNotNone(hold_payload)
        self.assertEqual(hold_payload.total_count, 2)
        self.assertEqual(cleared_payload.total_count, 0)
        self.assertFalse(cleared_payload.items)
        self.assertIsNone(runtime._stable_payload)

    def test_too_many_hands_warns_without_polluting_stable_result(self) -> None:
        runtime = HandCountRuntime(detector_factory=lambda: _DetectorStub([]))
        frame_shape = (100, 100, 3)
        two = (_make_hand("Left", 2),)
        too_many = (
            _make_hand("Left", 1, box=(30, 30, 50, 60)),
            _make_hand("Right", 1, box=(52, 30, 72, 60)),
            _make_hand("Unknown", 1, box=(40, 35, 68, 66)),
        )

        for _ in range(config.CAMERA_HAND_STABLE_MIN_COUNT):
            runtime._build_payload(two, frame_shape=frame_shape, count_fps=10.0)

        payload, status = runtime._build_payload(too_many, frame_shape=frame_shape, count_fps=10.0)

        self.assertTrue(payload.too_many_hands)
        self.assertEqual(payload.warnings, ("请仅保留两只手在画面中",))
        self.assertEqual(status, "请仅保留两只手在画面中")
        self.assertEqual(runtime._stable_payload.total_count, 2)

    def test_update_roi_size_clears_runtime_state(self) -> None:
        runtime = HandCountRuntime(detector_factory=lambda: _DetectorStub([]))
        frame_shape = (100, 100, 3)
        for _ in range(config.CAMERA_HAND_STABLE_MIN_COUNT):
            runtime._build_payload((_make_hand("Left", 2),), frame_shape=frame_shape, count_fps=10.0)

        runtime.update_roi_size(0.7, 0.5)

        self.assertEqual(runtime.roi_width_ratio, 0.7)
        self.assertEqual(runtime.roi_height_ratio, 0.5)
        self.assertIsNone(runtime.get_snapshot().latest_result)


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


def _wait_for_snapshot(runtime: HandCountRuntime):
    deadline = time.time() + 2.0
    while time.time() < deadline:
        snapshot = runtime.get_snapshot()
        if snapshot.latest_result is not None:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for hand-count snapshot")


if __name__ == "__main__":
    unittest.main()
