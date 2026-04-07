"""Desktop GUI controllers."""

from __future__ import annotations

from .camera_controller import CameraController
from .handwriting_controller import HandwritingController
from .image_controller import ImageController
from .recognition_controller import RecognitionController
from .result_panel_controller import ResultPanelController

__all__ = [
    "CameraController",
    "HandwritingController",
    "ImageController",
    "RecognitionController",
    "ResultPanelController",
]
