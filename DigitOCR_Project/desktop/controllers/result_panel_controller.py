"""Result panel controller for the staged GUI refactor."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from core.messages import coalesce_result_text
from desktop.messages import CAMERA_TEXT_COPIED_STATUS, RESULT_PLACEHOLDER_TEXT


class ResultPanelController:
    """Manage result table, preview rendering, clipboard, and saving."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def _copy_result_text(self) -> None:
        if self.app.camera_controller._is_camera_active():
            text = coalesce_result_text(self.app.camera_controller._camera_detection_text())
        elif self.app.current_output is not None:
            text = coalesce_result_text(self.app.current_output.combined_text)
        else:
            messagebox.showinfo("暂无结果", "当前还没有可复制的识别文本。")
            return

        self.app.clipboard_clear()
        self.app.clipboard_append(text)
        self.app.update_idletasks()
        self.app.recognition_controller._set_status(CAMERA_TEXT_COPIED_STATUS)

    def _restore_result_panel_from_current_output(self) -> None:
        if self.app.current_output is not None:
            self.app.summary_var.set(coalesce_result_text(self.app.current_output.combined_text))
            self._populate_result_table(self.app.current_output)
            return

        self.app.summary_var.set(RESULT_PLACEHOLDER_TEXT)
        for item_id in self.app.result_table.get_children():
            self.app.result_table.delete(item_id)
        self.app.result_table.insert("", "end", values=("-", "-", "未检测到数字"))

    def _populate_result_table(self, output) -> None:
        for item_id in self.app.result_table.get_children():
            self.app.result_table.delete(item_id)

        for result in output.results:
            box_text = " ".join(f"({x},{y})" for x, y in result.box)
            self.app.result_table.insert(
                "",
                "end",
                values=(result.text, f"{result.score:.2f}", box_text),
            )

        if not output.results:
            self.app.result_table.insert("", "end", values=("-", "-", "未检测到数字"))

    def _show_result_preview(self, image: np.ndarray) -> None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb_image))

        self.app.preview_canvas.delete("all")
        self.app.preview_image_item = self.app.preview_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.app.preview_canvas.configure(scrollregion=(0, 0, photo.width(), photo.height()))
        self.app.preview_canvas.xview_moveto(0)
        self.app.preview_canvas.yview_moveto(0)
        self.app.preview_photo = photo
        self.app.preview_placeholder_item = None

    def _schedule_result_preview(self, image: np.ndarray) -> None:
        if self.app.result_preview_after_id is not None:
            try:
                self.app.after_cancel(self.app.result_preview_after_id)
            except Exception:
                pass
            self.app.result_preview_after_id = None

        delay_ms = 80 if max(image.shape[:2]) >= 600 else 20
        self.app.result_preview_after_id = self.app.after(
            delay_ms,
            lambda preview=image: self._show_result_preview_when_idle(preview),
        )

    def _show_result_preview_when_idle(self, image: np.ndarray) -> None:
        self.app.result_preview_after_id = None
        self.app.after_idle(lambda preview=image: self._show_result_preview(preview))

    def _reset_result_preview(self, message: str = "识别后的预览图会显示在这里。") -> None:
        if self.app.result_preview_after_id is not None:
            try:
                self.app.after_cancel(self.app.result_preview_after_id)
            except Exception:
                pass
            self.app.result_preview_after_id = None

        self.app.preview_canvas.delete("all")
        self.app.preview_canvas.configure(scrollregion=(0, 0, 1, 1))
        self.app.preview_canvas.xview_moveto(0)
        self.app.preview_canvas.yview_moveto(0)
        self.app.preview_photo = None
        self.app.preview_image_item = None
        self.app.preview_placeholder_item = self.app.preview_canvas.create_text(
            0,
            0,
            text=message,
            fill="#667085",
            font=("Segoe UI", 11),
            justify="center",
        )
        self._center_preview_placeholder()

    def _handle_preview_canvas_configure(self, _: tk.Event) -> None:
        if self.app.preview_image_item is None:
            self._center_preview_placeholder()

    def _center_preview_placeholder(self) -> None:
        if self.app.preview_placeholder_item is None:
            return

        width = max(1, self.app.preview_canvas.winfo_width())
        height = max(1, self.app.preview_canvas.winfo_height())
        self.app.preview_canvas.coords(self.app.preview_placeholder_item, width / 2.0, height / 2.0)
        self.app.preview_canvas.itemconfigure(self.app.preview_placeholder_item, width=max(160, width - 24))

    def _show_bgr_image(
        self,
        target: ttk.Label,
        image: np.ndarray,
        *,
        max_size: tuple[int, int] | None,
        photo_slot: str,
    ) -> None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        if max_size is not None:
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)

        target.configure(image=photo, text="")
        setattr(self.app, photo_slot, photo)

    def _save_result_image(self) -> None:
        image_to_save: np.ndarray | None = None
        initial_name = "result_ocr.png"
        if self.app.camera_controller._is_camera_active() and self.app.current_camera_frame is not None:
            image_to_save = self.app.current_camera_frame
            initial_name = f"camera_{self.app.camera_device_var.get()}.png"
        elif self.app.current_output is not None:
            image_to_save = self.app.current_output.annotated_image
            initial_name = f"result_{self.app.current_output.source_name or 'ocr'}.png"
        else:
            messagebox.showinfo("暂无结果", "请先执行识别，再保存结果图片。")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存识别结果图片",
            defaultextension=".png",
            initialfile=initial_name,
            filetypes=[("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg"), ("所有文件", "*.*")],
        )
        if not file_path:
            return

        try:
            saved = cv2.imwrite(file_path, image_to_save)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return

        if not saved:
            messagebox.showerror("保存失败", f"无法保存图片到：{file_path}")
            return
        self.app.recognition_controller._set_status(f"结果图片已保存到：{file_path}")


__all__ = ["ResultPanelController"]
