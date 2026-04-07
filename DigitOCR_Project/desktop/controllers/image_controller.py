"""Uploaded image controller for the staged GUI refactor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tkinter import filedialog, messagebox, ttk

from desktop.media import LoadedImagePayload, load_image_for_preview


SUPPORTED_FILE_TYPES = [
    ("图片文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("所有文件", "*.*"),
]


class ImageController:
    """Handle uploaded image tab assembly and preview loading."""

    def __init__(self, app: Any, recognition_controller: Any, result_panel_controller: Any) -> None:
        self.app = app
        self.recognition_controller = recognition_controller
        self.result_panel_controller = result_panel_controller

    def _build_upload_tab(self) -> None:
        tab = ttk.Frame(self.app.notebook, padding=20)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        self.app.notebook.add(tab, text="图片识别")

        button_row = ttk.Frame(tab)
        button_row.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        button_row.columnconfigure(2, weight=1)

        self.app.choose_image_button = ttk.Button(button_row, text="选择图片", command=self._choose_image)
        self.app.choose_image_button.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.app.clear_image_button = ttk.Button(button_row, text="清除图片", command=self._clear_uploaded_image)
        self.app.clear_image_button.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        self.app.recognize_image_button = ttk.Button(
            button_row,
            text="识别上传图片",
            command=self._recognize_uploaded_image,
            style="Accent.TButton",
        )
        self.app.recognize_image_button.grid(row=0, column=2, sticky="ew")

        path_label = ttk.Label(tab, textvariable=self.app.upload_path_var, wraplength=620, foreground="#6b7280")
        path_label.grid(row=1, column=0, sticky="w", pady=(0, 10))

        self.app.upload_preview_label = ttk.Label(
            tab,
            text="上传图片的预览会显示在这里。",
            anchor="center",
            relief="flat",
            background="#e5e7eb",
            foreground="#6b7280",
        )
        self.app.upload_preview_label.grid(row=2, column=0, sticky="nsew")

    def _choose_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择要识别的图片",
            initialdir=str(self.app.project_root / "data" / "input"),
            filetypes=SUPPORTED_FILE_TYPES,
        )
        if not file_path:
            return

        self.app.image_loading = True
        self.app.image_load_token += 1
        current_token = self.app.image_load_token
        self._set_image_controls_enabled(enabled=False)
        self.app.upload_path_var.set("正在加载图片...")
        self.app.upload_preview_label.configure(image="", text="正在加载图片预览...")
        self.app.upload_preview_photo = None
        self.recognition_controller._set_status("正在加载图片...")

        self.recognition_controller._run_background_task(
            worker=lambda: load_image_for_preview(file_path, max_size=(650, 450)),
            on_success=lambda payload: self._handle_image_loaded(payload, current_token),
            on_error=lambda error: self._handle_image_load_error(error, current_token, Path(file_path)),
        )

    def _clear_uploaded_image(self) -> None:
        self.app.current_upload_path = None
        self.app.current_upload_image = None
        self.app.upload_path_var.set("尚未选择图片。")
        self.app.upload_preview_label.configure(image="", text="上传图片的预览会显示在这里。")
        self.app.upload_preview_photo = None
        self.app.image_loading = False
        self.recognition_controller._set_status("已清除上传图片。")

    def _handle_image_loaded(self, payload: LoadedImagePayload, token: int) -> None:
        if token != self.app.image_load_token:
            return

        self.app.image_loading = False
        self.app.current_upload_path = payload.path
        self.app.current_upload_image = payload.image
        self.app.upload_path_var.set(str(payload.path))
        self._set_image_controls_enabled(enabled=True)
        self.result_panel_controller._show_bgr_image(
            self.app.upload_preview_label,
            payload.preview_image,
            max_size=None,
            photo_slot="upload_preview_photo",
        )
        self.recognition_controller._set_status(f"已加载图片：{payload.path.name}")

    def _handle_image_load_error(self, error: Exception, token: int, path: Path) -> None:
        if token != self.app.image_load_token:
            return

        self.app.image_loading = False
        self._set_image_controls_enabled(enabled=True)
        self.app.current_upload_path = None
        self.app.current_upload_image = None
        self.app.upload_path_var.set("尚未选择图片。")
        self.app.upload_preview_label.configure(image="", text="上传图片的预览会显示在这里。")
        self.app.upload_preview_photo = None
        self.recognition_controller._set_status("图片加载失败。")
        messagebox.showerror("打开图片失败", f"无法读取所选图片：\n{path}\n\n{error}")

    def _recognize_uploaded_image(self) -> None:
        if self.app.current_upload_image is None or self.app.current_upload_path is None:
            messagebox.showinfo("未选择图片", "请先选择一张图片。")
            return

        image = self.app.current_upload_image.copy()
        source_name = self.app.current_upload_path.name
        self.recognition_controller._submit_recognition(
            task_name=f"正在识别图片：{source_name}",
            worker=lambda service: service.recognize_image(image, source_name=source_name, annotate_on_original=True),
        )

    def _set_image_controls_enabled(self, *, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.app.choose_image_button, self.app.clear_image_button, self.app.recognize_image_button):
            if widget is not None:
                widget.configure(state=state)


__all__ = ["ImageController"]
