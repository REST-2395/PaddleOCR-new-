"""Support helpers for the camera runtime."""

from __future__ import annotations

import os
import time

import cv2

from . import config


def replace_queue_item_latest_only(task_queue: object, task: object) -> bool:
    try:
        task_queue.put_nowait(task)
        return True
    except Exception:
        pass
    try:
        task_queue.get_nowait()
    except Exception:
        return False
    try:
        task_queue.put_nowait(task)
        return True
    except Exception:
        return False


def drain_task_queue(task_queue: object | None) -> None:
    if task_queue is None:
        return
    while True:
        try:
            task_queue.get_nowait()
        except Exception:
            break


def open_camera_capture(
    cv2_module: object,
    *,
    preferred_index: int,
    capture_size: tuple[int, int],
    target_capture_fps: float,
    warmup_frames: int,
) -> tuple[cv2.VideoCapture | None, int | None, str, str]:
    attempt_messages: list[str] = []
    for backend_code, backend_name in _camera_backend_candidates(cv2_module):
        capture = _create_capture(cv2_module, preferred_index, backend_code)
        if capture is None:
            attempt_messages.append(f"{backend_name}/#{preferred_index}: create failed")
            continue
        if not capture.isOpened():
            capture.release()
            attempt_messages.append(f"{backend_name}/#{preferred_index}: open failed")
            continue
        width, height = capture_size
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, float(height))
        if hasattr(cv2_module, "CAP_PROP_FPS"):
            capture.set(cv2_module.CAP_PROP_FPS, float(target_capture_fps))
        if hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2_module.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
        if not _warm_up_camera_capture(capture, warmup_frames=warmup_frames):
            capture.release()
            attempt_messages.append(f"{backend_name}/#{preferred_index}: no frame received")
            continue
        return capture, preferred_index, backend_name, f"{backend_name}/#{preferred_index}: ok"
    detail = " | ".join(attempt_messages) if attempt_messages else "no backend candidates"
    return None, None, "", detail


def _camera_backend_candidates(cv2_module: object) -> list[tuple[int | None, str]]:
    candidates: list[tuple[int | None, str]] = []
    if os.name == "nt":
        for attribute_name, display_name in (("CAP_DSHOW", "DSHOW"), ("CAP_MSMF", "MSMF")):
            if hasattr(cv2_module, attribute_name):
                candidates.append((getattr(cv2_module, attribute_name), display_name))
    if hasattr(cv2_module, "CAP_ANY"):
        candidates.append((getattr(cv2_module, "CAP_ANY"), "AUTO"))
    candidates.append((None, "DEFAULT"))
    unique_candidates: list[tuple[int | None, str]] = []
    seen_codes: set[int | None] = set()
    for backend_code, backend_name in candidates:
        if backend_code not in seen_codes:
            seen_codes.add(backend_code)
            unique_candidates.append((backend_code, backend_name))
    return unique_candidates


def _create_capture(cv2_module: object, index: int, backend_code: int | None) -> cv2.VideoCapture | None:
    try:
        if backend_code is None:
            return cv2_module.VideoCapture(index)
        return cv2_module.VideoCapture(index, backend_code)
    except Exception:
        return None


def _warm_up_camera_capture(capture: cv2.VideoCapture, *, warmup_frames: int) -> bool:
    for _ in range(max(warmup_frames, 1)):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True
        time.sleep(0.05)
    return False


__all__ = ["drain_task_queue", "open_camera_capture", "replace_queue_item_latest_only"]
