"""Entry point for batch digit OCR."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

from bootstrap.support import ensure_runtime_ready


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run digit-only OCR on images from a folder.")
    parser.add_argument("--skip-bootstrap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "input",
        help="Directory containing images to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "output",
        help="Directory where annotated images will be saved.",
    )
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=project_root / "config" / "digits_dict.txt",
        help="Digit whitelist dictionary path.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU inference if PaddlePaddle GPU is installed.",
    )
    parser.add_argument(
        "--ocr-version",
        type=str,
        default="PP-OCRv5",
        choices=["PP-OCRv3", "PP-OCRv4", "PP-OCRv5"],
        help="OCR model family to use when official models are loaded automatically.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Filter out OCR results below this confidence score.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="result_",
        help="Filename prefix for annotated output images.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="CPU thread count for Paddle OCR on CPU devices (default: auto).",
    )
    parser.add_argument("--det-model-dir", type=str, default=None, help="Optional custom detection model directory.")
    parser.add_argument("--rec-model-dir", type=str, default=None, help="Optional custom recognition model directory.")
    parser.add_argument("--cls-model-dir", type=str, default=None, help="Optional custom textline orientation directory.")
    return parser.parse_args()


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    return sorted(
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def main() -> int:
    mp.freeze_support()
    ensure_runtime_ready(project_root=Path(__file__).resolve().parent)

    import cv2

    from core.recognition_service import DigitOCRService

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        print(f"No images found in: {args.input_dir}")
        return 0

    print(f"Loading OCR engine ({args.ocr_version})...")
    service = DigitOCRService(
        dict_path=args.dict_path,
        use_gpu=args.use_gpu,
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        cls_model_dir=args.cls_model_dir,
        ocr_version=args.ocr_version,
        score_threshold=args.score_threshold,
        cpu_threads=args.cpu_threads,
    )

    save_failures = False
    for image_path in image_paths:
        output = service.recognize_image_path(image_path)
        output_path = args.output_dir / f"{args.output_prefix}{image_path.name}"
        print(f"{image_path.name}: {output.summary_text}")
        if cv2.imwrite(str(output_path), output.annotated_image):
            print(f"Saved: {output_path}")
            continue

        save_failures = True
        print(f"Failed to save: {output_path}")

    return 1 if save_failures else 0


if __name__ == "__main__":
    sys.exit(main())
