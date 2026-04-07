"""MediaPipe-backed hand detector for the live hand-count mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from camera import config

from .constants import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_COUNT,
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
from .types import FingerStateTuple, HandCountItem, HandLandmarkPoint, HandednessLabel


def _load_mediapipe_module() -> Any:
    try:
        import mediapipe as mp
    except Exception as exc:  # pragma: no cover - exercised by integration tests
        raise RuntimeError(
            "MediaPipe 初始化失败，请先安装 requirements.txt 中的 mediapipe 依赖。"
        ) from exc
    return mp


class HandDetector:
    """Convert MediaPipe Hands outputs into project-native count items."""

    def __init__(
        self,
        *,
        max_num_hands: int = config.CAMERA_HAND_MAX_HANDS,
        model_complexity: int = config.CAMERA_HAND_MODEL_COMPLEXITY,
        min_detection_confidence: float = config.CAMERA_HAND_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.CAMERA_HAND_MIN_TRACKING_CONFIDENCE,
        hands_solution: Any | None = None,
    ) -> None:
        internal_max_hands = max(int(max_num_hands), config.CAMERA_HAND_MAX_HANDS + 1)
        self._backend = "solutions"
        self._mp_module = _load_mediapipe_module()
        if hands_solution is not None:
            self._hands_solution = hands_solution
            self._hands = self._hands_solution.Hands(
                static_image_mode=False,
                max_num_hands=internal_max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._connections = tuple(getattr(self._hands_solution, "HAND_CONNECTIONS", HAND_CONNECTIONS))
            return

        mp_solutions = getattr(self._mp_module, "solutions", None)
        hands_api = getattr(mp_solutions, "hands", None) if mp_solutions is not None else None
        if hands_api is not None:
            self._hands_solution = hands_api
            self._hands = self._hands_solution.Hands(
                static_image_mode=False,
                max_num_hands=internal_max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._connections = tuple(getattr(self._hands_solution, "HAND_CONNECTIONS", HAND_CONNECTIONS))
            return

        self._backend = "tasks"
        self._hands_solution = None
        self._hands = self._create_tasks_hand_landmarker(
            num_hands=internal_max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._connections = tuple(self._tasks_connections())

    @property
    def connections(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(start), int(end)) for start, end in self._connections)

    def close(self) -> None:
        if hasattr(self._hands, "close"):
            self._hands.close()

    def find_hands(self, image_bgr) -> Any:
        if self._backend == "tasks":
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self._mp_module.Image(
                image_format=self._mp_module.ImageFormat.SRGB,
                data=image_rgb,
            )
            return self._hands.detect(mp_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._hands.process(image_rgb)

    def detect(self, image_bgr) -> tuple[HandCountItem, ...]:
        return self.to_hand_items(self.find_hands(image_bgr), image_bgr.shape)

    def fingers_up(
        self,
        handedness: HandednessLabel,
        landmarks: tuple[HandLandmarkPoint, ...],
    ) -> FingerStateTuple:
        thumb_tip = landmarks[THUMB_TIP_INDEX]
        thumb_base = landmarks[THUMB_MCP_INDEX]
        if handedness == "Right":
            thumb_up = int(thumb_tip.x < thumb_base.x)
        elif handedness == "Left":
            thumb_up = int(thumb_tip.x > thumb_base.x)
        else:
            thumb_up = 0

        return (
            thumb_up,
            int(landmarks[INDEX_FINGER_TIP_INDEX].y < landmarks[INDEX_FINGER_PIP_INDEX].y),
            int(landmarks[MIDDLE_FINGER_TIP_INDEX].y < landmarks[MIDDLE_FINGER_PIP_INDEX].y),
            int(landmarks[RING_FINGER_TIP_INDEX].y < landmarks[RING_FINGER_PIP_INDEX].y),
            int(landmarks[PINKY_TIP_INDEX].y < landmarks[PINKY_PIP_INDEX].y),
        )

    def count_fingers(self, finger_states: FingerStateTuple) -> int:
        return int(sum(int(state) for state in finger_states))

    def extract_hand_box(
        self,
        landmarks: tuple[HandLandmarkPoint, ...],
        *,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        height, width = image_shape[:2]
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        x0 = max(0, min(xs))
        y0 = max(0, min(ys))
        x1 = min(width, max(xs))
        y1 = min(height, max(ys))
        return x0, y0, x1, y1

    def to_hand_items(
        self,
        results: Any,
        image_shape: tuple[int, ...],
    ) -> tuple[HandCountItem, ...]:
        landmark_lists, handedness_lists = self._adapt_results(results)
        if not landmark_lists:
            return ()

        items: list[HandCountItem] = []
        for index, landmark_list in enumerate(landmark_lists):
            landmarks = self._pixel_landmarks(landmark_list, image_shape=image_shape)
            handedness_info = handedness_lists[index] if index < len(handedness_lists) else None
            handedness = self._handedness_label(handedness_info)
            score = self._handedness_score(handedness_info)
            finger_states = self.fingers_up(handedness, landmarks)
            items.append(
                HandCountItem(
                    handedness=handedness,
                    score=score,
                    finger_states=finger_states,
                    count=self.count_fingers(finger_states),
                    box=self.extract_hand_box(landmarks, image_shape=image_shape),
                    landmarks=landmarks,
                )
            )
        return tuple(items)

    def _adapt_results(self, results: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        if self._backend == "tasks":
            landmarks = tuple(getattr(results, "hand_landmarks", ()) or ())
            handedness = tuple(getattr(results, "handedness", ()) or ())
            return landmarks, handedness

        landmarks = tuple(getattr(results, "multi_hand_landmarks", ()) or ())
        handedness = tuple(getattr(results, "multi_handedness", ()) or ())
        return landmarks, handedness

    def _pixel_landmarks(
        self,
        landmark_list: Any,
        *,
        image_shape: tuple[int, ...],
    ) -> tuple[HandLandmarkPoint, ...]:
        height, width = image_shape[:2]
        landmarks = tuple(getattr(landmark_list, "landmark", ()) or landmark_list or ())
        points = []
        for landmark in landmarks[:HAND_LANDMARK_COUNT]:
            x = int(round(float(getattr(landmark, "x", 0.0)) * max(width - 1, 1)))
            y = int(round(float(getattr(landmark, "y", 0.0)) * max(height - 1, 1)))
            points.append(HandLandmarkPoint(x=x, y=y))
        return tuple(points)

    def _handedness_label(self, handedness_info: Any) -> HandednessLabel:
        classifications = self._handedness_entries(handedness_info)
        if not classifications:
            return "Unknown"
        category = classifications[0]
        label_candidates = (
            getattr(category, "label", None),
            getattr(category, "category_name", None),
            getattr(category, "display_name", None),
        )
        for label in label_candidates:
            if label in {"Left", "Right"}:
                return self._mirrored_handedness(label)
        return "Unknown"

    def _handedness_score(self, handedness_info: Any) -> float:
        classifications = self._handedness_entries(handedness_info)
        if not classifications:
            return 0.0
        return float(getattr(classifications[0], "score", 0.0) or 0.0)

    def _handedness_entries(self, handedness_info: Any) -> tuple[Any, ...]:
        if handedness_info is None:
            return ()

        classifications = getattr(handedness_info, "classification", None)
        if classifications is not None:
            return tuple(classifications or ())

        try:
            return tuple(handedness_info or ())
        except TypeError:
            return (handedness_info,)

    def _mirrored_handedness(self, label: str) -> HandednessLabel:
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return "Unknown"

    def _create_tasks_hand_landmarker(
        self,
        *,
        num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> Any:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("当前 Mediapipe 安装不包含可用的 hand landmarker API。") from exc

        model_path = self._task_model_path()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _tasks_connections(self) -> tuple[tuple[int, int], ...]:
        from mediapipe.tasks.python import vision

        connections = getattr(vision.HandLandmarksConnections, "HAND_CONNECTIONS", ())
        return tuple((int(item.start), int(item.end)) for item in connections)

    def _task_model_path(self) -> Path:
        model_path = Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"
        if not model_path.exists():
            raise RuntimeError(f"Hand landmarker model not found: {model_path}")
        return model_path


__all__ = ["HandDetector"]
