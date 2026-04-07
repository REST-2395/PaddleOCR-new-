"""Camera mode profiles and mode-specific worker defaults."""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass(frozen=True, slots=True)
class CameraModeProfile:
    """Mode-specific camera worker and timing defaults."""

    mode: str
    capture_size: tuple[int, int]
    cpu_threads: int
    enable_mkldnn: bool
    use_textline_orientation: bool
    ocr_interval_seconds: float
    max_ocr_side: int
    fast_worker_threads: int | None = None
    fallback_worker_threads: int | None = None

    @property
    def uses_fast_workers(self) -> bool:
        return self.fast_worker_threads is not None

    @property
    def primary_worker_kind(self) -> str:
        return "board" if self.mode == config.CAMERA_MODE_BOARD else "fallback"

    @property
    def primary_worker_name(self) -> str:
        return "camera-board-worker" if self.mode == config.CAMERA_MODE_BOARD else "camera-fallback-worker"

    @property
    def primary_worker_threads(self) -> int:
        if self.mode == config.CAMERA_MODE_BOARD:
            return self.cpu_threads
        return int(self.fallback_worker_threads or self.cpu_threads)


_MODE_PROFILES: dict[str, CameraModeProfile] = {
    config.CAMERA_MODE_DIGIT: CameraModeProfile(
        mode=config.CAMERA_MODE_DIGIT,
        capture_size=(config.CAMERA_FRAME_WIDTH, config.CAMERA_FRAME_HEIGHT),
        cpu_threads=2,
        enable_mkldnn=False,
        use_textline_orientation=True,
        ocr_interval_seconds=config.CAMERA_INFERENCE_INTERVAL_MS / 1000.0,
        max_ocr_side=config.CAMERA_MAX_OCR_SIZE,
        fast_worker_threads=config.CAMERA_FAST_WORKER_THREADS,
        fallback_worker_threads=config.CAMERA_FALLBACK_WORKER_THREADS,
    ),
    config.CAMERA_MODE_BOARD: CameraModeProfile(
        mode=config.CAMERA_MODE_BOARD,
        capture_size=(config.CAMERA_BOARD_FRAME_WIDTH, config.CAMERA_BOARD_FRAME_HEIGHT),
        cpu_threads=4,
        enable_mkldnn=True,
        use_textline_orientation=False,
        ocr_interval_seconds=config.CAMERA_BOARD_INFERENCE_INTERVAL_MS / 1000.0,
        max_ocr_side=config.CAMERA_BOARD_MAX_OCR_SIZE,
    ),
}


def get_camera_mode_profile(mode: str) -> CameraModeProfile:
    """Return the shared defaults for one camera mode."""
    try:
        return _MODE_PROFILES[mode]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported camera mode: {mode}") from exc

__all__ = [
    "CameraModeProfile",
    "get_camera_mode_profile",
]
