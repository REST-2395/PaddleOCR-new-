"""Handwriting controller for the staged GUI refactor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from tkinter import ttk


@dataclass(slots=True)
class StrokeRecord:
    """One stored handwriting stroke that can be replayed after a resize."""

    points: list[tuple[float, float]]
    width: float


class HandwritingController:
    """Handle handwriting canvas assembly, drawing, and recognition submission."""

    def __init__(self, app: Any, recognition_controller: Any) -> None:
        self.app = app
        self.recognition_controller = recognition_controller

    def _build_handwriting_tab(self) -> None:
        tab = ttk.Frame(self.app.notebook, padding=20)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        self.app.notebook.add(tab, text="手写识别")

        ttk.Label(
            tab,
            text="按住鼠标左键，在画板上手写数字。",
            foreground="#6b7280",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        canvas_frame = ttk.Frame(tab)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.app.handwriting_canvas = tk.Canvas(
            canvas_frame,
            width=self.app.canvas_width,
            height=self.app.canvas_height,
            bg="white",
            cursor="crosshair",
            highlightthickness=1,
            highlightbackground="#d1d5db",
            relief="flat",
        )
        self.app.handwriting_canvas.grid(row=0, column=0, sticky="nsew")
        self.app.handwriting_canvas.bind("<ButtonPress-1>", self._start_drawing)
        self.app.handwriting_canvas.bind("<B1-Motion>", self._continue_drawing)
        self.app.handwriting_canvas.bind("<ButtonRelease-1>", self._stop_drawing)
        self.app.handwriting_canvas.bind("<Configure>", self._handle_handwriting_canvas_configure)

        controls = ttk.Frame(tab, padding=(0, 20, 0, 0))
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="画笔粗细").grid(row=0, column=0, sticky="w", padx=(0, 10))
        brush_scale = ttk.Scale(controls, from_=8, to=30, variable=self.app.brush_width, orient="horizontal")
        brush_scale.grid(row=0, column=1, sticky="ew", padx=(0, 20))

        self.app.clear_canvas_button = ttk.Button(controls, text="清空画板", command=self._clear_canvas)
        self.app.clear_canvas_button.grid(row=0, column=2, sticky="ew", padx=(0, 10))

        self.app.recognize_canvas_button = ttk.Button(
            controls,
            text="识别手写内容",
            command=self._recognize_handwriting,
            style="Accent.TButton",
        )
        self.app.recognize_canvas_button.grid(row=0, column=3, sticky="ew")

    def _reset_handwriting_surface(self) -> None:
        self.app.handwriting_image = Image.new("RGB", self.app.handwriting_surface_size, "white")
        self.app.handwriting_draw = ImageDraw.Draw(self.app.handwriting_image)

    def _handle_handwriting_canvas_configure(self, event: tk.Event) -> None:
        self._sync_handwriting_surface_to_size(event.width, event.height)

    def _sync_handwriting_surface_to_widget(self) -> None:
        self._sync_handwriting_surface_to_size(
            self.app.handwriting_canvas.winfo_width(),
            self.app.handwriting_canvas.winfo_height(),
        )

    def _sync_handwriting_surface_to_size(self, width: int, height: int) -> None:
        if width < 32 or height < 32:
            return

        new_size = (int(width), int(height))
        old_size = self.app.handwriting_surface_size
        if new_size == old_size:
            return

        self.app.active_stroke = None
        self._scale_stroke_history(old_size=old_size, new_size=new_size)
        self.app.handwriting_surface_size = new_size
        self._rebuild_handwriting_surface()

    def _scale_stroke_history(
        self,
        *,
        old_size: tuple[int, int],
        new_size: tuple[int, int],
    ) -> None:
        old_width, old_height = old_size
        new_width, new_height = new_size
        if old_width <= 0 or old_height <= 0:
            return

        scale_x = new_width / old_width
        scale_y = new_height / old_height
        width_scale = min(scale_x, scale_y)

        for stroke in self.app.stroke_history:
            stroke.points = [(x * scale_x, y * scale_y) for x, y in stroke.points]
            stroke.width = max(1.0, stroke.width * width_scale)

    def _rebuild_handwriting_surface(self) -> None:
        self._reset_handwriting_surface()
        self.app.handwriting_canvas.delete("all")
        for stroke in self.app.stroke_history:
            self._render_full_stroke(stroke)

    def _render_full_stroke(self, stroke: StrokeRecord) -> None:
        width = max(1, int(round(stroke.width)))
        points = [self._clamp_point_to_surface(x, y) for x, y in stroke.points]
        if not points:
            return

        if len(points) == 1:
            self._draw_dot(points[0], width)
            return

        flat_points = [coordinate for point in points for coordinate in point]
        self.app.handwriting_canvas.create_line(
            *flat_points,
            fill="black",
            width=width,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=24,
        )
        self.app.handwriting_draw.line(points, fill="black", width=width)
        self._draw_dot(points[0], width)
        self._draw_dot(points[-1], width)

    def _draw_line_segment(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        width: int,
    ) -> None:
        x0, y0 = self._clamp_point_to_surface(*start)
        x1, y1 = self._clamp_point_to_surface(*end)
        self.app.handwriting_canvas.create_line(
            x0,
            y0,
            x1,
            y1,
            fill="black",
            width=width,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=24,
        )
        self.app.handwriting_draw.line(((x0, y0), (x1, y1)), fill="black", width=width)

    def _clamp_point_to_surface(self, x: float, y: float) -> tuple[float, float]:
        width, height = self.app.handwriting_surface_size
        clamped_x = min(max(float(x), 0.0), max(0.0, width - 1))
        clamped_y = min(max(float(y), 0.0), max(0.0, height - 1))
        return clamped_x, clamped_y

    def _start_drawing(self, event: tk.Event) -> None:
        self._sync_handwriting_surface_to_widget()
        width = float(self.app.brush_width.get())
        start_point = self._clamp_point_to_surface(event.x, event.y)
        stroke = StrokeRecord(points=[start_point], width=width)
        self.app.stroke_history.append(stroke)
        self.app.active_stroke = stroke
        self._draw_dot(start_point, max(1, int(round(width))))

    def _continue_drawing(self, event: tk.Event) -> None:
        if self.app.active_stroke is None:
            self._start_drawing(event)
            return

        width = max(1, int(round(self.app.active_stroke.width)))
        previous_point = self.app.active_stroke.points[-1]
        current_point = self._clamp_point_to_surface(event.x, event.y)
        if current_point == previous_point:
            return

        self.app.active_stroke.points.append(current_point)
        self._draw_line_segment(previous_point, current_point, width)
        self._draw_dot(current_point, width)

    def _stop_drawing(self, _: tk.Event) -> None:
        self.app.active_stroke = None

    def _draw_dot(self, point: tuple[float, float], width: int) -> None:
        x, y = self._clamp_point_to_surface(*point)
        radius = max(4, width // 2)
        self.app.handwriting_canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="black",
            outline="",
        )
        self.app.handwriting_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="black")

    def _clear_canvas(self) -> None:
        self.app.handwriting_canvas.delete("all")
        self.app.stroke_history.clear()
        self.app.active_stroke = None
        self._reset_handwriting_surface()
        self.recognition_controller._set_status("画板已清空。")

    def _recognize_handwriting(self) -> None:
        self._sync_handwriting_surface_to_widget()
        canvas_rgb = np.array(self.app.handwriting_image, dtype=np.uint8)
        canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
        self.recognition_controller._submit_recognition(
            task_name="正在识别手写内容...",
            worker=lambda service: service.recognize_handwriting(
                canvas_bgr,
                source_name="手写画板",
                progress_callback=self.recognition_controller._queue_status_update,
            ),
        )


__all__ = ["HandwritingController", "StrokeRecord"]
