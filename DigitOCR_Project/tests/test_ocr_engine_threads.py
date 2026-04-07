from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ocr_engine import DigitOCREngine


class OcrEngineCpuThreadTests(unittest.TestCase):
    def test_engine_passes_cpu_threads_to_paddle_components(self) -> None:
        captured: dict[str, list[dict[str, object]]] = {"ocr": [], "rec": []}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured["ocr"].append(kwargs)

            def predict(self, _image, **_kwargs):
                return []

        class FakeTextRecognition:
            def __init__(self, **kwargs):
                captured["rec"].append(kwargs)

            def predict(self, images, **_kwargs):
                return [{"res": {"rec_text": "1", "rec_score": 0.99}} for _ in images]

        fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR, TextRecognition=FakeTextRecognition)
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            engine = DigitOCREngine(
                dict_path=PROJECT_ROOT / "config" / "digits_dict.txt",
                use_gpu=False,
                cpu_threads=3,
            )
            _ = engine.recognize_handwriting_blocks([np.full((16, 16, 3), 255, dtype=np.uint8)])

        self.assertEqual(captured["ocr"][0]["cpu_threads"], 3)
        self.assertEqual(captured["rec"][0]["cpu_threads"], 3)
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "3")

    def test_engine_falls_back_when_cpu_threads_not_supported(self) -> None:
        captured: dict[str, list[dict[str, object]]] = {"ocr": [], "rec": []}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                if "cpu_threads" in kwargs:
                    raise TypeError("unexpected keyword argument 'cpu_threads'")
                captured["ocr"].append(kwargs)

            def predict(self, _image, **_kwargs):
                return []

        class FakeTextRecognition:
            def __init__(self, **kwargs):
                if "cpu_threads" in kwargs:
                    raise TypeError("unexpected keyword argument 'cpu_threads'")
                captured["rec"].append(kwargs)

            def predict(self, images, **_kwargs):
                return [{"res": {"rec_text": "2", "rec_score": 0.95}} for _ in images]

        fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR, TextRecognition=FakeTextRecognition)
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            engine = DigitOCREngine(
                dict_path=PROJECT_ROOT / "config" / "digits_dict.txt",
                use_gpu=False,
                cpu_threads=4,
            )
            _ = engine.recognize_handwriting_blocks([np.full((16, 16, 3), 255, dtype=np.uint8)])

        self.assertNotIn("cpu_threads", captured["ocr"][0])
        self.assertNotIn("cpu_threads", captured["rec"][0])

    def test_engine_retries_detection_without_mkldnn_after_onednn_error(self) -> None:
        captured: list[dict[str, object]] = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                self.enable_mkldnn = bool(kwargs.get("enable_mkldnn"))
                captured.append(kwargs)

            def predict(self, _image, **_kwargs):
                if self.enable_mkldnn:
                    raise RuntimeError("ConvertPirAttribute2RuntimeAttribute not support in oneDNN")
                return []

        fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR, TextRecognition=object)
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            engine = DigitOCREngine(
                dict_path=PROJECT_ROOT / "config" / "digits_dict.txt",
                use_gpu=False,
                enable_mkldnn=True,
            )
            results = engine.recognize(np.full((24, 24, 3), 255, dtype=np.uint8))

        self.assertEqual(results, [])
        self.assertEqual(len(captured), 2)
        self.assertTrue(captured[0]["enable_mkldnn"])
        self.assertFalse(captured[1]["enable_mkldnn"])
        self.assertFalse(engine.enable_mkldnn)

    def test_engine_retries_handwriting_without_mkldnn_after_onednn_error(self) -> None:
        captured: dict[str, list[dict[str, object]]] = {"rec": []}

        class FakeTextRecognition:
            def __init__(self, **kwargs):
                self.enable_mkldnn = bool(kwargs.get("enable_mkldnn"))
                captured["rec"].append(kwargs)

            def predict(self, images, **_kwargs):
                if self.enable_mkldnn:
                    raise RuntimeError("oneDNN ConvertPirAttribute2RuntimeAttribute not support")
                return [{"res": {"rec_text": "3", "rec_score": 0.96}} for _ in images]

        fake_module = types.SimpleNamespace(PaddleOCR=object, TextRecognition=FakeTextRecognition)
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            engine = DigitOCREngine(
                dict_path=PROJECT_ROOT / "config" / "digits_dict.txt",
                use_gpu=False,
                load_detection_engine=False,
                enable_mkldnn=True,
            )
            results = engine.recognize_handwriting_blocks([np.full((16, 16, 3), 255, dtype=np.uint8)])

        self.assertEqual([item.text for item in results], ["3"])
        self.assertEqual(len(captured["rec"]), 2)
        self.assertTrue(captured["rec"][0]["enable_mkldnn"])
        self.assertFalse(captured["rec"][1]["enable_mkldnn"])
        self.assertFalse(engine.enable_mkldnn)


if __name__ == "__main__":
    unittest.main()
