from __future__ import annotations

from collections import deque
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from desktop.controllers.camera_controller import CameraController
from desktop.controllers.recognition_controller import RecognitionController
from desktop.controllers.result_panel_controller import ResultPanelController
import desktop.controllers.result_panel_controller as result_panel_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as cli_main


def _load_gui_app_module():
    module_path = PROJECT_ROOT / "gui_app.pyw"
    spec = importlib.util.spec_from_file_location("digit_ocr_gui_app", module_path)
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


class SaveBehaviorTests(unittest.TestCase):
    def test_cli_main_returns_nonzero_when_image_save_fails(self) -> None:
        output_image = np.zeros((12, 12, 3), dtype=np.uint8)
        service = Mock()
        service.recognize_image_path.return_value = SimpleNamespace(
            annotated_image=output_image,
            summary_text="ok",
        )
        args = SimpleNamespace(
            input_dir=PROJECT_ROOT / "data" / "input",
            output_dir=PROJECT_ROOT / "data" / "output",
            dict_path=PROJECT_ROOT / "config" / "digits_dict.txt",
            use_gpu=False,
            det_model_dir=None,
            rec_model_dir=None,
            cls_model_dir=None,
            ocr_version="PP-OCRv5",
            score_threshold=0.3,
            output_prefix="result_",
            cpu_threads=None,
        )

        with patch.object(cli_main, "parse_args", return_value=args), \
             patch.object(cli_main, "collect_images", return_value=[Path("sample.png")]), \
             patch("core.recognition_service.DigitOCRService", return_value=service), \
             patch("cv2.imwrite", return_value=False), \
             patch("builtins.print") as print_mock:
            exit_code = cli_main.main()

        self.assertEqual(exit_code, 1)
        print_mock.assert_any_call("Failed to save: {}".format(args.output_dir / "result_sample.png"))

    def test_gui_save_result_image_surfaces_imwrite_failure(self) -> None:
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

    def test_handle_recognition_success_updates_results_before_scheduling_preview(self) -> None:
        call_order: list[str] = []
        app = SimpleNamespace(
            summary_var=SimpleNamespace(set=lambda _value: call_order.append("summary")),
            status_var=SimpleNamespace(set=lambda _message: call_order.append("status")),
            busy=True,
            current_output=None,
        )
        result_panel_controller = SimpleNamespace(
            _populate_result_table=lambda _output: call_order.append("table"),
            _schedule_result_preview=lambda _image: call_order.append("preview"),
        )
        controller = RecognitionController(app, result_panel_controller)
        controller._toggle_actions = lambda *, enabled: call_order.append(f"toggle:{enabled}")
        output = SimpleNamespace(
            combined_text="123",
            summary_text="123 (0.99)",
            warnings=[],
            results=[SimpleNamespace(text="1", score=0.99, box=[[0, 0], [1, 0], [1, 1], [0, 1]])],
            source_name="structured",
        )

        controller._handle_recognition_success(output, np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertEqual(call_order, ["summary", "table", "toggle:True", "status", "preview"])
        self.assertFalse(app.busy)

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

    def test_handle_camera_started_uses_manual_board_prompt(self) -> None:
        summary_updates: list[str] = []
        status_updates: list[str] = []
        camera_status_updates: list[str] = []
        control_calls: list[tuple[bool, bool, bool]] = []
        scheduled: list[str] = []

        app = SimpleNamespace(
            camera_operation_token=3,
            camera_starting=True,
            camera_stopping=True,
            summary_var=SimpleNamespace(set=lambda value: summary_updates.append(value)),
            camera_status_var=SimpleNamespace(set=lambda value: camera_status_updates.append(value)),
        )
        recognition_controller = SimpleNamespace(
            _run_background_task=lambda **_kwargs: None,
            _set_status=lambda message: status_updates.append(message),
        )
        controller = CameraController(app, recognition_controller=recognition_controller, result_panel_controller=SimpleNamespace())
        controller._reset_camera_display_state = lambda: None
        controller._set_camera_controls_state = lambda *, start_enabled, stop_enabled, device_enabled: control_calls.append(
            (start_enabled, stop_enabled, device_enabled)
        )
        controller._populate_camera_result_table = lambda values=(), **_kwargs: None
        controller._schedule_camera_poll = lambda: scheduled.append("poll")

        runtime = SimpleNamespace(camera_mode=gui_app.CAMERA_MODE_BOARD, backend_name="TEST", device_index=2)

        controller._handle_camera_started(runtime, 3)

        self.assertIs(app.camera_session, runtime)
        self.assertEqual(summary_updates[-1], "请将黑板上的数字放入识别框。")
        self.assertEqual(status_updates[-1], "已启动摄像头 2，当前为黑板模式。")
        self.assertIn("摄像头 2 (TEST)", camera_status_updates[-1])
        self.assertEqual(control_calls[-1], (False, True, False))
        self.assertEqual(scheduled, ["poll"])


if __name__ == "__main__":
    unittest.main()
