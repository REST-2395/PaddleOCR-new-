"""Shared OCR workflow helpers for CLI and desktop UI."""

from __future__ import annotations

from pathlib import Path

from .image_processor import ImageProcessor
from .messages import BOARD_WARNING_TEXT, HANDWRITING_WARNING_TEXT, IMAGE_WARNING_TEXT
from .ocr_engine import DigitOCREngine, OCRResult, TextOnlyResult
from .service_geometry_support import ServiceGeometryMixin
from .service_image_utils import ServiceImageUtilityMixin
from .service_public_api import ServicePublicApiMixin
from .service_types import (
    HandwritingBlock,
    ImageCandidate,
    ImageCandidateBlock,
    ImageReviewResult,
    RecognitionOutput,
    RegionBox,
    StructuredSequence,
)


class DigitOCRService(ServicePublicApiMixin, ServiceImageUtilityMixin, ServiceGeometryMixin):
    """High-level service that performs image enhancement and OCR."""

    handwriting_threshold = 245
    handwriting_candidate_min_side = 220
    handwriting_fallback_score = 0.90
    image_rotation_retry_score = 0.90
    handwriting_split_depth_limit = 4
    image_candidate_duplicate_iou = 0.65
    image_candidate_split_score = 0.84
    image_candidate_review_score = 0.72
    image_candidate_strong_review_score = 0.96
    image_candidate_max_segments = 5
    structured_sequence_min_score = 0.78
    structured_route_min_row_items = 3
    structured_route_min_panel_count = 4
    structured_panel_min_area_ratio = 0.42
    structured_panel_min_height_ratio = 0.58
    structured_hint_override_margin = 0.14
    handwriting_warning_text = HANDWRITING_WARNING_TEXT
    image_warning_text = IMAGE_WARNING_TEXT
    board_warning_text = BOARD_WARNING_TEXT
    rotation_retry_angles = (0.0, 90.0, 180.0, 270.0)

    def __init__(
        self,
        dict_path: str | Path,
        use_gpu: bool = False,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        cls_model_dir: str | None = None,
        ocr_version: str = "PP-OCRv5",
        score_threshold: float = 0.3,
        cpu_threads: int | None = None,
        enable_mkldnn: bool = False,
        use_textline_orientation: bool = True,
        language: str = "en",
    ) -> None:
        self.processor = ImageProcessor()
        self.camera_processor = ImageProcessor(
            min_short_side=256,
            bilateral_diameter=5,
            bilateral_sigma_color=45,
            bilateral_sigma_space=45,
            clahe_clip_limit=1.6,
        )
        self.engine = DigitOCREngine(
            dict_path=dict_path,
            use_gpu=use_gpu,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            ocr_version=ocr_version,
            score_threshold=score_threshold,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            use_textline_orientation=use_textline_orientation,
            language=language,
        )
        self._handwriting_pipeline = None
        self._image_pipeline = None
        self._camera_digit_pipeline = None
        self._board_sequence_pipeline = None

    def _get_handwriting_pipeline(self):
        from .pipelines.handwriting_pipeline import HandwritingPipeline

        pipeline = self.__dict__.get("_handwriting_pipeline")
        if pipeline is None:
            pipeline = HandwritingPipeline(self)
            self._handwriting_pipeline = pipeline
        return pipeline

    def _get_image_pipeline(self):
        from .pipelines.image_pipeline import ImageRecognitionPipeline

        pipeline = self.__dict__.get("_image_pipeline")
        if pipeline is None:
            pipeline = ImageRecognitionPipeline(self)
            self._image_pipeline = pipeline
        return pipeline

    def _get_camera_digit_pipeline(self):
        from .pipelines.camera_digit_pipeline import CameraDigitPipeline

        pipeline = self.__dict__.get("_camera_digit_pipeline")
        if pipeline is None:
            pipeline = CameraDigitPipeline(self)
            self._camera_digit_pipeline = pipeline
        return pipeline

    def _get_board_sequence_pipeline(self):
        from .pipelines.board_sequence_pipeline import BoardSequencePipeline

        pipeline = self.__dict__.get("_board_sequence_pipeline")
        if pipeline is None:
            pipeline = BoardSequencePipeline(self)
            self._board_sequence_pipeline = pipeline
        return pipeline


__all__ = [
    "DigitOCRService",
    "HandwritingBlock",
    "ImageCandidate",
    "ImageCandidateBlock",
    "ImageReviewResult",
    "OCRResult",
    "RecognitionOutput",
    "RegionBox",
    "StructuredSequence",
    "TextOnlyResult",
]
