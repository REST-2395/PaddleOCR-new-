"""Typed data objects for the hand-count camera mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias


HandednessLabel: TypeAlias = Literal["Left", "Right", "Unknown"]
HandBox: TypeAlias = tuple[int, int, int, int]
FingerStateTuple: TypeAlias = tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class HandLandmarkPoint:
    """One pixel-space hand landmark."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class HandCountItem:
    """One detected hand and its finger-count metadata."""

    handedness: HandednessLabel
    score: float
    finger_states: FingerStateTuple
    count: int
    box: HandBox
    landmarks: tuple[HandLandmarkPoint, ...]


@dataclass(frozen=True, slots=True)
class HandCountPayload:
    """Mode-specific payload published through CameraInferenceResult."""

    items: tuple[HandCountItem, ...] = field(default_factory=tuple)
    total_count: int = 0
    too_many_hands: bool = False
    fps: float = 0.0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_items(self) -> bool:
        return bool(self.items)


__all__ = [
    "FingerStateTuple",
    "HandBox",
    "HandCountItem",
    "HandCountPayload",
    "HandLandmarkPoint",
    "HandednessLabel",
]
