"""Windows desktop GUI for handwritten and image-based digit OCR."""

from __future__ import annotations

from collections import deque
import multiprocessing as mp
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import ImageTk

from bootstrap.support import ensure_runtime_ready

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ensure_runtime_ready(project_root=PROJECT_ROOT)

from camera.runtime import (  # noqa: E402
    CAMERA_MODE_BOARD,
    CAMERA_MODE_DIGIT,
    CAMERA_MODE_HAND_COUNT,
    CAMERA_ROI_HEIGHT_RATIO,
    CAMERA_ROI_WIDTH_RATIO,
    CAMERA_SEQUENCE_HISTORY,
    CameraDetection,
    CameraOCRRuntime,
    CameraTrack,
)
from core.recognition_service import DigitOCRService, RecognitionOutput  # noqa: E402
from desktop.controllers import (  # noqa: E402
    CameraController,
    HandwritingController,
    ImageController,
    RecognitionController,
    ResultPanelController,
)
from desktop.messages import APP_READY_STATUS, CAMERA_IDLE_STATUS, CAMERA_MODE_LABELS, RESULT_PLACEHOLDER_TEXT  # noqa: E402


def _initialize_app_state(app: "DigitOCRGuiApp") -> None:
    app.service = None
    app.busy = False
    app.worker_lock = threading.Lock()

    app.canvas_width = 560
    app.canvas_height = 360
    app.brush_width = tk.IntVar(value=18)
    app.status_var = tk.StringVar(value=APP_READY_STATUS)
    app.summary_var = tk.StringVar(value=RESULT_PLACEHOLDER_TEXT)
    app.upload_path_var = tk.StringVar(value="尚未选择图片。")
    app.camera_device_var = tk.StringVar(value="0")
    app.camera_mode_var = tk.StringVar(value=CAMERA_MODE_LABELS[CAMERA_MODE_DIGIT])
    app.camera_status_var = tk.StringVar(value=CAMERA_IDLE_STATUS)
    app.camera_roi_width_var = tk.DoubleVar(value=CAMERA_ROI_WIDTH_RATIO)
    app.camera_roi_height_var = tk.DoubleVar(value=CAMERA_ROI_HEIGHT_RATIO)
    app.camera_roi_label_var = tk.StringVar(value="")

    app.handwriting_surface_size = (app.canvas_width, app.canvas_height)
    app.stroke_history = []
    app.active_stroke = None
    app.current_upload_path = None
    app.current_upload_image = None
    app.current_output = None
    app.camera_session = None
    app.camera_tab = None
    app.camera_preview_label = None
    app.camera_start_button = None
    app.camera_stop_button = None
    app.camera_device_combo = None
    app.camera_mode_combo = None
    app.camera_roi_width_scale = None
    app.camera_roi_height_scale = None
    app.camera_apply_roi_button = None
    app.camera_poll_after_id = None
    app.camera_preview_photo = None
    app.current_camera_preview = None
    app.current_camera_frame = None
    app.current_camera_detections = ()
    app.current_camera_payload = None
    app.current_camera_result_mode = CAMERA_MODE_DIGIT
    app.camera_tracks = []
    app.camera_next_track_id = 1
    app.camera_empty_streak = 0
    app.camera_filter_state = None
    app.camera_result_signature = None
    app.camera_last_render_signature = None
    app.camera_last_snapshot_frame_id = -1
    app.camera_last_result_completed_at = 0.0
    app.camera_last_consumed_result_id = -1
    app.camera_sequence_history = deque(maxlen=CAMERA_SEQUENCE_HISTORY)
    app.camera_error_message = None
    app.camera_starting = False
    app.camera_stopping = False
    app.camera_operation_token = 0
    app.image_loading = False
    app.image_load_token = 0
    app.service_warmup_started = False
    app.service_warmup_finished = False
    app.preview_photo = None
    app.upload_preview_photo = None
    app.preview_image_item = None
    app.preview_placeholder_item = None
    app.result_preview_after_id = None

    app.notebook = None
    app.handwriting_canvas = None
    app.clear_canvas_button = None
    app.recognize_canvas_button = None
    app.choose_image_button = None
    app.clear_image_button = None
    app.recognize_image_button = None
    app.upload_preview_label = None
    app.result_table = None
    app.preview_canvas = None
    app.save_result_button = None
    app.copy_result_button = None


def _build_preview_canvas(app: "DigitOCRGuiApp", right_frame: ttk.Frame) -> None:
    preview_frame = ttk.Frame(right_frame, style="Card.TFrame")
    preview_frame.grid(row=5, column=0, columnspan=2, sticky="nsew")
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)

    app.preview_canvas = tk.Canvas(
        preview_frame,
        bg="white",
        highlightthickness=0,
        relief="flat",
    )
    app.preview_canvas.grid(row=0, column=0, sticky="nsew")
    app.preview_canvas.bind("<Configure>", app.result_panel_controller._handle_preview_canvas_configure)

    preview_y_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=app.preview_canvas.yview)
    preview_y_scroll.grid(row=0, column=1, sticky="ns")
    preview_x_scroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=app.preview_canvas.xview)
    preview_x_scroll.grid(row=1, column=0, sticky="ew")
    app.preview_canvas.configure(
        xscrollcommand=preview_x_scroll.set,
        yscrollcommand=preview_y_scroll.set,
    )


def _build_result_action_row(app: "DigitOCRGuiApp", right_frame: ttk.Frame) -> None:
    button_row = ttk.Frame(right_frame, style="Card.TFrame")
    button_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(20, 0))
    button_row.columnconfigure(0, weight=1)
    button_row.columnconfigure(1, weight=1)

    app.save_result_button = ttk.Button(
        button_row,
        text="保存结果图片",
        command=app.result_panel_controller._save_result_image,
    )
    app.save_result_button.grid(row=0, column=0, sticky="ew", padx=(0, 10))

    app.copy_result_button = ttk.Button(
        button_row,
        text="复制识别文本",
        command=app.result_panel_controller._copy_result_text,
    )
    app.copy_result_button.grid(row=0, column=1, sticky="ew", padx=(10, 0))


def _build_result_panel(app: "DigitOCRGuiApp", container: ttk.Frame) -> None:
    right_frame = ttk.Frame(container, style="Card.TFrame", padding=20)
    right_frame.grid(row=0, column=1, sticky="nsew")
    right_frame.columnconfigure(0, weight=1)
    right_frame.rowconfigure(3, weight=1)
    right_frame.rowconfigure(5, weight=2)

    ttk.Label(right_frame, text="识别结果", font=("Microsoft YaHei UI", 14, "bold"), background="white").grid(
        row=0,
        column=0,
        sticky="w",
        pady=(0, 10),
    )

    summary_label = ttk.Label(
        right_frame,
        textvariable=app.summary_var,
        font=("Microsoft YaHei UI", 16, "bold"),
        foreground="#0f4c81",
        wraplength=480,
        justify="left",
        background="white",
    )
    summary_label.grid(row=1, column=0, sticky="ew", pady=(0, 20))

    columns = ("text", "score", "box")
    app.result_table = ttk.Treeview(right_frame, columns=columns, show="headings", height=8)
    app.result_table.heading("text", text="文本")
    app.result_table.heading("score", text="置信度")
    app.result_table.heading("box", text="位置框")
    app.result_table.column("text", width=80, anchor="center")
    app.result_table.column("score", width=100, anchor="center")
    app.result_table.column("box", width=260, anchor="w")
    app.result_table.grid(row=3, column=0, sticky="nsew")

    table_scroll = ttk.Scrollbar(right_frame, orient="vertical", command=app.result_table.yview)
    app.result_table.configure(yscrollcommand=table_scroll.set)
    table_scroll.grid(row=3, column=1, sticky="ns")

    ttk.Label(right_frame, text="结果预览", font=("Microsoft YaHei UI", 14, "bold"), background="white").grid(
        row=4,
        column=0,
        columnspan=2,
        sticky="w",
        pady=(20, 10),
    )

    _build_preview_canvas(app, right_frame)
    _build_result_action_row(app, right_frame)


class DigitOCRGuiApp(tk.Tk):
    """Desktop app for digit OCR."""

    def __init__(self) -> None:
        super().__init__()
        self.project_root = PROJECT_ROOT
        self.dict_path = self.project_root / "config" / "digits_dict.txt"
        _initialize_app_state(self)

        self.result_panel_controller = ResultPanelController(self)
        self.recognition_controller = RecognitionController(self, self.result_panel_controller)
        self.camera_controller = CameraController(self, self.recognition_controller, self.result_panel_controller)
        self.image_controller = ImageController(self, self.recognition_controller, self.result_panel_controller)
        self.handwriting_controller = HandwritingController(self, self.recognition_controller)
        self.camera_roi_label_var.set(self.camera_controller._camera_roi_label_text())

        self.title("数字识别工作台")
        self.geometry("1380x860")
        self.minsize(1220, 760)

        self._configure_styles()
        self._build_ui()
        self.notebook.bind("<<NotebookTabChanged>>", self._handle_notebook_tab_changed)
        self.protocol("WM_DELETE_WINDOW", self._handle_close)
        self.handwriting_controller._reset_handwriting_surface()
        self.result_panel_controller._reset_result_preview()
        self.after(300, self.recognition_controller._start_service_warmup)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = ("Microsoft YaHei UI", 10)
        bg_color = "#f0f2f5"
        fg_color = "#1f2937"
        accent_color = "#2563eb"
        accent_active = "#1d4ed8"

        style.configure(".", background=bg_color, foreground=fg_color, font=base_font)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color, font=base_font)

        style.configure("TButton", padding=(12, 8), font=base_font, borderwidth=0, relief="flat", background="white")
        style.map(
            "TButton",
            background=[("active", "#e5e7eb"), ("disabled", "#f3f4f6")],
            foreground=[("disabled", "#9ca3af")],
        )

        style.configure("Accent.TButton", background=accent_color, foreground="white", font=("Microsoft YaHei UI", 10, "bold"))
        style.map(
            "Accent.TButton",
            background=[("active", accent_active), ("disabled", "#93c5fd")],
            foreground=[("disabled", "white")],
        )

        style.configure("Treeview", rowheight=32, font=base_font, background="white", fieldbackground="white", borderwidth=0)
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(10, 8),
            background="#e5e7eb",
            foreground=fg_color,
        )
        style.map("Treeview", background=[("selected", accent_color)], foreground=[("selected", "white")])

        style.configure("TNotebook", background=bg_color, tabposition="n", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(20, 10), font=("Microsoft YaHei UI", 11), background="#e5e7eb", foreground="#6b7280")
        style.map(
            "TNotebook.Tab",
            background=[("selected", "white")],
            foreground=[("selected", accent_color)],
            expand=[("selected", [0, 0, 0, 0])],
        )

        style.configure("Card.TFrame", background="white", relief="flat")
        style.configure("Status.TLabel", font=("Microsoft YaHei UI", 9), foreground="#6b7280", background="#e5e7eb", padding=8)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self, padding=20)
        container.grid(sticky="nsew")
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=0)

        left_frame = ttk.Frame(container)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.grid(sticky="nsew")

        self.handwriting_controller._build_handwriting_tab()
        self.image_controller._build_upload_tab()
        self.camera_controller._build_camera_tab()
        _build_result_panel(self, container)

        status_bar = ttk.Label(
            container,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
        )
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(20, 0))

    def _handle_notebook_tab_changed(self, _: tk.Event) -> None:
        if self.camera_tab is None:
            return

        selected_tab = self.notebook.select()
        if self.camera_controller._is_camera_active() and str(self.camera_tab) != selected_tab:
            self.camera_controller._stop_camera_session(reset_preview=True)

    def _handle_close(self) -> None:
        self.camera_controller._stop_camera_session(reset_preview=False)
        self.destroy()


def main() -> None:
    mp.freeze_support()
    app = DigitOCRGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
