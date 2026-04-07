from __future__ import annotations

import ast
from collections import deque
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from desktop.controllers.camera_controller import CameraController
from desktop.controllers.result_panel_controller import ResultPanelController
import desktop.controllers.result_panel_controller as result_panel_module
from handcount.types import HandCountItem, HandCountPayload, HandLandmarkPoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_gui_app_module():
    module_path = PROJECT_ROOT / "gui_app.pyw"
    spec = importlib.util.spec_from_file_location("digit_ocr_gui_app_characterization", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load gui app module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gui_app = _load_gui_app_module()


class _VarStub:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _WidgetStub:
    def __init__(self) -> None:
        self.state = None

    def configure(self, **kwargs) -> None:
        if "state" in kwargs:
            self.state = kwargs["state"]


class GuiCameraCharacterizationTests(unittest.TestCase):
    def test_gui_camera_code_only_depends_on_runtime_facade(self) -> None:
        source = (PROJECT_ROOT / "gui_app.pyw").read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(PROJECT_ROOT / "gui_app.pyw"))

        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_camera_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "camera"
            for alias in node.names
        }

        self.assertNotIn("camera.state", imported_modules)
        self.assertNotIn("config", imported_camera_names)
        self.assertIn("camera.runtime", imported_modules)

    def test_gui_app_only_keeps_assembly_methods_after_phase_5(self) -> None:
        source = (PROJECT_ROOT / "gui_app.pyw").read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(PROJECT_ROOT / "gui_app.pyw"))

        method_names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "DigitOCRGuiApp":
                method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
                break

        self.assertEqual(
            method_names,
            ["__init__", "_configure_styles", "_build_ui", "_handle_notebook_tab_changed", "_handle_close"],
        )

    def test_board_mode_keeps_roi_controls_enabled(self) -> None:
        app = SimpleNamespace(
            camera_start_button=_WidgetStub(),
            camera_stop_button=_WidgetStub(),
            camera_device_combo=_WidgetStub(),
            camera_mode_combo=_WidgetStub(),
            camera_roi_width_scale=_WidgetStub(),
            camera_roi_height_scale=_WidgetStub(),
            camera_apply_roi_button=_WidgetStub(),
        )
        controller = CameraController(app, recognition_controller=SimpleNamespace(), result_panel_controller=SimpleNamespace())

        controller._set_camera_controls_state(start_enabled=False, stop_enabled=True, device_enabled=False)

        self.assertEqual(app.camera_roi_width_scale.state, "normal")
        self.assertEqual(app.camera_roi_height_scale.state, "normal")
        self.assertEqual(app.camera_apply_roi_button.state, "normal")

    def test_apply_camera_roi_size_updates_running_board_runtime(self) -> None:
        applied_sizes: list[tuple[float, float]] = []
        status_updates: list[str] = []
        summary_updates: list[str] = []
        table_updates: list[tuple] = []
        app = SimpleNamespace(
            camera_roi_width_var=_VarStub(0.68),
            camera_roi_height_var=_VarStub(0.52),
            camera_roi_label_var=_VarStub(""),
            camera_mode_var=_VarStub(gui_app.CAMERA_MODE_LABELS[gui_app.CAMERA_MODE_BOARD]),
            camera_session=SimpleNamespace(
                is_running=True,
                update_roi_size=lambda width, height: applied_sizes.append((width, height)),
            ),
            current_camera_detections=(object(),),
            current_camera_payload=None,
            camera_tracks=[object()],
            camera_result_signature="old",
            camera_last_render_signature="old",
            camera_last_result_completed_at=1.0,
            camera_last_consumed_result_id=7,
            camera_sequence_history=deque(["12"], maxlen=6),
            summary_var=SimpleNamespace(set=lambda value: summary_updates.append(value)),
        )
        recognition_controller = SimpleNamespace(_set_status=lambda message: status_updates.append(message))
        controller = CameraController(app, recognition_controller=recognition_controller, result_panel_controller=SimpleNamespace())
        controller._populate_camera_result_table = lambda values=(), **_kwargs: table_updates.append(tuple(values))

        controller._apply_camera_roi_size()

        self.assertEqual(applied_sizes, [(0.68, 0.52)])
        self.assertEqual(summary_updates[-1], "请将黑板上的数字放入识别框。")
        self.assertEqual(table_updates[-1], ())
        self.assertEqual(status_updates[-1], "已应用新的识别框大小。")
        self.assertEqual(app.camera_roi_label_var.get(), "识别框大小：宽 68% / 高 52%")

    def test_board_digit_and_hand_count_start_prompts_stay_separate(self) -> None:
        def build_controller() -> tuple[CameraController, list[str], list[str], object]:
            summary_updates: list[str] = []
            status_updates: list[str] = []
            app = SimpleNamespace(
                camera_operation_token=3,
                camera_starting=True,
                camera_stopping=False,
                summary_var=SimpleNamespace(set=lambda value: summary_updates.append(value)),
                camera_status_var=SimpleNamespace(set=lambda _value: None),
            )
            recognition_controller = SimpleNamespace(
                _run_background_task=lambda **_kwargs: None,
                _set_status=lambda message: status_updates.append(message),
            )
            controller = CameraController(app, recognition_controller=recognition_controller, result_panel_controller=SimpleNamespace())
            controller._reset_camera_display_state = lambda: None
            controller._set_camera_controls_state = lambda **_kwargs: None
            controller._populate_camera_result_table = lambda values=(), **_kwargs: None
            controller._schedule_camera_poll = lambda: None
            return controller, summary_updates, status_updates, app

        board_controller, board_summary, board_status, _ = build_controller()
        digit_controller, digit_summary, digit_status, _ = build_controller()
        hand_controller, hand_summary, hand_status, _ = build_controller()

        board_runtime = SimpleNamespace(
            camera_mode=gui_app.CAMERA_MODE_BOARD,
            backend_name="TEST",
            device_index=2,
        )
        digit_runtime = SimpleNamespace(
            camera_mode=gui_app.CAMERA_MODE_DIGIT,
            backend_name="TEST",
            device_index=2,
        )
        hand_runtime = SimpleNamespace(
            camera_mode=gui_app.CAMERA_MODE_HAND_COUNT,
            backend_name="TEST",
            device_index=2,
        )

        board_controller._handle_camera_started(board_runtime, 3)
        digit_controller._handle_camera_started(digit_runtime, 3)
        hand_controller._handle_camera_started(hand_runtime, 3)

        self.assertEqual(board_summary[-1], "请将黑板上的数字放入识别框。")
        self.assertEqual(digit_summary[-1], "请将数字放入识别框。")
        self.assertEqual(hand_summary[-1], "请将手放入计数框内。")
        self.assertIn("黑板模式", board_status[-1])
        self.assertNotIn("黑板模式", digit_status[-1])
        self.assertIn("手势计数模式", hand_status[-1])

    def test_hand_count_detection_text_uses_total_summary(self) -> None:
        payload = HandCountPayload(
            items=(
                HandCountItem(
                    handedness="Left",
                    score=0.96,
                    finger_states=(1, 1, 1, 0, 0),
                    count=3,
                    box=(10, 10, 40, 60),
                    landmarks=tuple(HandLandmarkPoint(x=10 + index, y=20 + index) for index in range(21)),
                ),
            ),
            total_count=3,
            too_many_hands=False,
            fps=12.0,
            warnings=(),
        )
        app = SimpleNamespace(
            current_camera_result_mode=gui_app.CAMERA_MODE_HAND_COUNT,
            current_camera_detections=(),
            current_camera_payload=payload,
        )
        controller = CameraController(app, recognition_controller=SimpleNamespace(), result_panel_controller=SimpleNamespace())

        self.assertEqual(controller._camera_detection_text(), "总数：3")

    def test_switching_tabs_stops_active_camera_session(self) -> None:
        app = gui_app.DigitOCRGuiApp.__new__(gui_app.DigitOCRGuiApp)
        stop_calls: list[bool] = []
        app.camera_tab = ".!notebook.!frame3"
        app.notebook = SimpleNamespace(select=lambda: ".!notebook.!frame1")
        app.camera_controller = SimpleNamespace(
            _is_camera_active=lambda: True,
            _stop_camera_session=lambda *, reset_preview: stop_calls.append(reset_preview),
        )

        app._handle_notebook_tab_changed(None)

        self.assertEqual(stop_calls, [True])

    def test_handle_close_stops_camera_and_destroys_window(self) -> None:
        app = gui_app.DigitOCRGuiApp.__new__(gui_app.DigitOCRGuiApp)
        close_calls: list[bool] = []
        destroyed: list[str] = []
        app.camera_controller = SimpleNamespace(
            _stop_camera_session=lambda *, reset_preview: close_calls.append(reset_preview),
        )
        app.destroy = lambda: destroyed.append("destroyed")

        app._handle_close()

        self.assertEqual(close_calls, [False])
        self.assertEqual(destroyed, ["destroyed"])

    def test_save_result_image_surfaces_imwrite_failure(self) -> None:
        status_updates: list[str] = []
        app = SimpleNamespace(
            current_camera_frame=np.zeros((16, 16, 3), dtype=np.uint8),
            current_output=None,
            camera_device_var=SimpleNamespace(get=lambda: "0"),
            camera_controller=SimpleNamespace(_is_camera_active=lambda: True),
            recognition_controller=SimpleNamespace(_set_status=lambda message: status_updates.append(message)),
        )
        controller = ResultPanelController(app)

        with patch.object(result_panel_module.filedialog, "asksaveasfilename", return_value="C:/temp/result.png"), \
             patch.object(result_panel_module.cv2, "imwrite", return_value=False), \
             patch.object(result_panel_module.messagebox, "showerror") as showerror_mock:
            controller._save_result_image()

        showerror_mock.assert_called_once()
        self.assertEqual(status_updates, [])


if __name__ == "__main__":
    unittest.main()
