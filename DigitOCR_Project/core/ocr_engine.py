"""Wrapper around PaddleOCR for digit-only recognition."""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEX_CACHE_DIR = PROJECT_ROOT / ".runtime" / "paddlex_cache"
PADDLEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PADDLEX_CACHE_DIR))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

HANDWRITING_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
HANDWRITING_MODEL_DIR = PADDLEX_CACHE_DIR / "official_models" / HANDWRITING_MODEL_NAME
HANDWRITING_BATCH_SIZE = 8
DEFAULT_CPU_THREADS = max(1, min(6, int(os.cpu_count() or 4)))


def _resolve_cpu_threads(value: int | None) -> int:
    """Resolve OCR CPU threads with a safe default for desktop usage."""
    if value is None:
        return DEFAULT_CPU_THREADS
    return max(1, int(value))


def _apply_cpu_thread_env_limits(cpu_threads: int) -> None:
    """Apply common BLAS/OpenMP thread limits for CPU OCR workloads."""
    thread_value = str(max(1, int(cpu_threads)))
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = thread_value


@dataclass(slots=True)
class OCRResult:
    """Normalized OCR output for one detected text region."""

    text: str
    score: float
    box: list[list[int]]


@dataclass(slots=True)
class TextOnlyResult:
    """Normalized text-only recognition result for one cropped image."""

    text: str
    score: float


class DigitOCREngine:
    """Digit-only OCR engine backed by PaddleOCR."""

    def __init__(
        self,
        dict_path: str | Path,
        use_gpu: bool = False,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        cls_model_dir: str | None = None,
        ocr_version: str = "PP-OCRv5",
        score_threshold: float = 0.3,
        load_detection_engine: bool = True,
        cpu_threads: int | None = None,
        enable_mkldnn: bool = False,
        use_textline_orientation: bool = True,
        language: str = "en",
    ) -> None:
        self.score_threshold = score_threshold
        self.allowed_chars = self._load_allowed_chars(dict_path)
        self.device = "gpu" if use_gpu else "cpu"
        self.cpu_threads = _resolve_cpu_threads(cpu_threads)
        self.enable_mkldnn = bool(enable_mkldnn)
        self.use_textline_orientation = bool(use_textline_orientation)
        self.language = language
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.cls_model_dir = cls_model_dir
        self.ocr_version = ocr_version
        self.load_detection_engine = bool(load_detection_engine)
        self._handwriting_recognizer: object | None = None
        self._handwriting_lock = threading.Lock()
        self._ocr_lock = threading.Lock()
        self.ocr: object | None = None

        if bool(det_model_dir) != bool(rec_model_dir):
            raise ValueError(
                "Please provide both detection and recognition model directories together, or leave both unset."
            )

        if not load_detection_engine:
            return

        if self.device == "cpu":
            _apply_cpu_thread_env_limits(self.cpu_threads)

        self.ocr = self._create_detection_engine()

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Run the general OCR pipeline and return normalized result items."""
        if self.ocr is None:
            raise RuntimeError("Detection OCR engine is not initialized.")
        try:
            raw_results = self.ocr.predict(
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except Exception as exc:
            if not self._should_retry_without_mkldnn(exc):
                raise
            raw_results = self._retry_recognize_without_mkldnn(image)
        return self._normalize_results(raw_results)

    def recognize_handwriting_blocks(self, images: Sequence[np.ndarray]) -> list[TextOnlyResult]:
        """Run the lightweight text recognition model on cropped handwriting blocks."""
        if not images:
            return []

        recognizer = self._get_handwriting_recognizer()
        try:
            raw_results = recognizer.predict(
                list(images),
                batch_size=max(1, min(len(images), HANDWRITING_BATCH_SIZE)),
            )
        except Exception as exc:
            if not self._should_retry_without_mkldnn(exc):
                raise
            raw_results = self._retry_handwriting_without_mkldnn(images)
        return self._normalize_text_only_results(raw_results, expected_count=len(images))

    def draw_results(self, image: np.ndarray, results: Sequence[OCRResult]) -> np.ndarray:
        """Draw boxes and recognized digits on the image."""
        annotated = image.copy()

        for result in results:
            points = np.array(result.box, dtype=np.int32)
            cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            x = int(np.min(points[:, 0]))
            y = int(np.min(points[:, 1])) - 10
            y = max(y, 20)
            label = f"{result.text} ({result.score:.2f})"
            cv2.putText(
                annotated,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def _get_handwriting_recognizer(self) -> object:
        """Lazy-load the lightweight recognition model used for handwriting blocks."""
        if self._handwriting_recognizer is not None:
            return self._handwriting_recognizer

        with self._handwriting_lock:
            if self._handwriting_recognizer is None:
                self._handwriting_recognizer = self._create_handwriting_recognizer()

        return self._handwriting_recognizer

    def _create_detection_engine(self) -> object:
        if self.device == "cpu":
            _apply_cpu_thread_env_limits(self.cpu_threads)

        from paddleocr import PaddleOCR

        return self._init_with_optional_cpu_threads(PaddleOCR, self._build_detection_init_kwargs())

    def _build_detection_init_kwargs(self) -> dict[str, object]:
        init_kwargs: dict[str, object] = {
            "device": self.device,
            "enable_mkldnn": self.enable_mkldnn,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self.use_textline_orientation,
            "text_rec_score_thresh": self.score_threshold,
        }
        if self.device == "cpu":
            init_kwargs["cpu_threads"] = self.cpu_threads

        if self.det_model_dir and self.rec_model_dir:
            init_kwargs["text_detection_model_dir"] = self.det_model_dir
            init_kwargs["text_recognition_model_dir"] = self.rec_model_dir
            if self.cls_model_dir:
                init_kwargs["textline_orientation_model_dir"] = self.cls_model_dir
        else:
            init_kwargs["lang"] = self.language
            init_kwargs["ocr_version"] = self.ocr_version
        return init_kwargs

    def _create_handwriting_recognizer(self) -> object:
        from paddleocr import TextRecognition

        init_kwargs: dict[str, object] = {
            "model_name": HANDWRITING_MODEL_NAME,
            "device": self.device,
            "enable_mkldnn": self.enable_mkldnn,
        }
        if self.device == "cpu":
            _apply_cpu_thread_env_limits(self.cpu_threads)
            init_kwargs["cpu_threads"] = self.cpu_threads
        if HANDWRITING_MODEL_DIR.exists():
            init_kwargs["model_dir"] = str(HANDWRITING_MODEL_DIR)
        return self._init_with_optional_cpu_threads(TextRecognition, init_kwargs)

    def _should_retry_without_mkldnn(self, exc: Exception) -> bool:
        if self.device != "cpu" or not self.enable_mkldnn:
            return False
        message = str(exc)
        lowered = message.lower()
        indicators = (
            "onednn",
            "mkldnn",
            "convertpirattribute2runtimeattribute",
            "pir::arrayattribute",
        )
        return any(item in lowered for item in indicators)

    def _disable_mkldnn_and_rebuild(self) -> None:
        self.enable_mkldnn = False
        self.ocr = self._create_detection_engine() if self.load_detection_engine else None
        self._handwriting_recognizer = None

    def _retry_recognize_without_mkldnn(self, image: np.ndarray) -> object:
        with self._ocr_lock:
            if self.enable_mkldnn:
                self._disable_mkldnn_and_rebuild()
            if self.ocr is None:
                raise RuntimeError("Detection OCR engine is not initialized.")
            return self.ocr.predict(
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )

    def _retry_handwriting_without_mkldnn(self, images: Sequence[np.ndarray]) -> object:
        with self._handwriting_lock:
            if self.enable_mkldnn:
                self._disable_mkldnn_and_rebuild()
            self._handwriting_recognizer = self._create_handwriting_recognizer()
            recognizer = self._handwriting_recognizer
        return recognizer.predict(
            list(images),
            batch_size=max(1, min(len(images), HANDWRITING_BATCH_SIZE)),
        )

    @staticmethod
    def _init_with_optional_cpu_threads(factory: object, init_kwargs: dict[str, object]) -> object:
        """Initialize Paddle objects while gracefully handling missing cpu_threads support."""
        try:
            return factory(**init_kwargs)
        except TypeError as exc:
            if "cpu_threads" not in str(exc):
                raise
            fallback_kwargs = dict(init_kwargs)
            fallback_kwargs.pop("cpu_threads", None)
            return factory(**fallback_kwargs)

    def _normalize_text_only_results(
        self,
        raw_results: object,
        *,
        expected_count: int,
    ) -> list[TextOnlyResult]:
        """Normalize batched text-only recognition results while preserving input order."""
        normalized = [TextOnlyResult(text="", score=0.0) for _ in range(expected_count)]
        if raw_results is None:
            return normalized

        raw_items = raw_results if isinstance(raw_results, list) else [raw_results]
        for index, item in enumerate(raw_items):
            if index >= expected_count:
                break
            normalized[index] = self._parse_text_only_result(item)

        return normalized

    def _parse_text_only_result(self, value: object) -> TextOnlyResult:
        payload = self._extract_text_only_payload(value)
        clean_text = self._sanitize_text(payload.get("rec_text"))
        if not clean_text:
            return TextOnlyResult(text="", score=0.0)

        score_value = float(payload.get("rec_score") or 0.0)
        return TextOnlyResult(text=clean_text, score=score_value)

    @staticmethod
    def _extract_text_only_payload(value: object) -> Mapping[str, object]:
        """Extract the plain text recognition payload from a PaddleX result object."""
        if isinstance(value, Mapping):
            payload = value.get("res")
            if isinstance(payload, Mapping):
                return payload
            return value

        json_payload = getattr(value, "json", None)
        if isinstance(json_payload, Mapping):
            payload = json_payload.get("res")
            if isinstance(payload, Mapping):
                return payload
            return json_payload

        return {}

    def _normalize_results(self, raw_results: object) -> list[OCRResult]:
        normalized: list[OCRResult] = []

        for page_result in self._iter_page_results(raw_results):
            if self._looks_like_page_result(page_result):
                normalized.extend(self._parse_page_result(page_result))
                continue

            normalized.extend(self._parse_legacy_result(page_result))

        normalized.sort(
            key=lambda item: (
                min(point[1] for point in item.box),
                min(point[0] for point in item.box),
            )
        )
        return normalized

    def _iter_page_results(self, raw_results: object) -> Iterable[object]:
        if raw_results is None:
            return []

        if isinstance(raw_results, list):
            return raw_results

        return [raw_results]

    @staticmethod
    def _looks_like_page_result(value: object) -> bool:
        if not isinstance(value, Mapping):
            return False
        return "rec_texts" in value and "rec_scores" in value and "rec_polys" in value

    @staticmethod
    def _looks_like_line(value: object) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False

        box, rec = value[0], value[1]
        return isinstance(box, (list, tuple)) and isinstance(rec, (list, tuple)) and len(rec) >= 2

    def _parse_page_result(self, page_result: Mapping[str, object]) -> list[OCRResult]:
        normalized: list[OCRResult] = []
        texts = page_result.get("rec_texts") or []
        scores = page_result.get("rec_scores") or []
        boxes = page_result.get("rec_polys") or []

        for text, score, box in zip(texts, scores, boxes):
            clean_text = self._sanitize_text(text)
            if not clean_text:
                continue

            score_value = float(score)
            if score_value < self.score_threshold:
                continue

            points = [[int(point[0]), int(point[1])] for point in box]
            normalized.append(OCRResult(text=clean_text, score=score_value, box=points))

        return normalized

    def _parse_legacy_result(self, page_result: object) -> list[OCRResult]:
        normalized: list[OCRResult] = []

        if not isinstance(page_result, list):
            return normalized

        lines: list[Sequence[object]] = []
        if page_result and self._looks_like_line(page_result[0]):
            lines = [line for line in page_result if self._looks_like_line(line)]
        else:
            for item in page_result:
                if isinstance(item, list):
                    lines.extend(line for line in item if self._looks_like_line(line))

        for line in lines:
            box, (text, score) = line[0], line[1]
            clean_text = self._sanitize_text(text)
            if not clean_text:
                continue

            score_value = float(score)
            if score_value < self.score_threshold:
                continue

            points = [[int(point[0]), int(point[1])] for point in box]
            normalized.append(OCRResult(text=clean_text, score=score_value, box=points))

        return normalized

    def _sanitize_text(self, text: object) -> str:
        if isinstance(text, (list, tuple)) and text:
            text = text[0]

        filtered = "".join(char for char in str(text) if char in self.allowed_chars)
        return filtered.strip()

    @staticmethod
    def _load_allowed_chars(dict_path: str | Path) -> frozenset[str]:
        path = Path(dict_path)
        entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not entries:
            raise ValueError(f"Dictionary file is empty: {path}")
        return frozenset("".join(entries))
