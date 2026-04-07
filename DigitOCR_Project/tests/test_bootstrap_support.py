from __future__ import annotations
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bootstrap import support as bootstrap_helpers


class BootstrapSupportTests(unittest.TestCase):
    def test_should_skip_bootstrap_when_flag_or_env_present(self) -> None:
        self.assertTrue(bootstrap_helpers.should_skip_bootstrap(["--skip-bootstrap"]))
        with patch.dict(os.environ, {"DIGITOCR_SKIP_BOOTSTRAP": "1"}, clear=False):
            self.assertTrue(bootstrap_helpers.should_skip_bootstrap([]))

    def test_build_bootstrap_command_targets_project_root(self) -> None:
        project_root = Path("/tmp/demo")

        command = bootstrap_helpers.build_bootstrap_command(project_root)

        self.assertEqual(command[1], str(project_root / "bootstrap_env.py"))
        self.assertIn("--skip-installed", command)

    def test_ensure_runtime_ready_execs_into_project_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            venv_python = project_root / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("bootstrap.support.subprocess.run") as mocked_run, patch(
                    "bootstrap.support.default_venv_python", return_value=venv_python
                ), patch(
                    "bootstrap.support.runtime_dependencies_ready", return_value=False
                ), patch("bootstrap.support.os.execve") as mocked_execve, patch(
                    "bootstrap.support.sys.executable", str(project_root / "system-python")
                ), patch("bootstrap.support.sys.argv", [str(project_root / "main.py"), "--demo"]):
                    bootstrap_helpers.ensure_runtime_ready(project_root=project_root, argv=[])

        mocked_run.assert_called_once()
        mocked_execve.assert_called_once()

    def test_ensure_runtime_ready_returns_when_already_using_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            venv_python = project_root / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("bootstrap.support.subprocess.run") as mocked_run, patch(
                    "bootstrap.support.default_venv_python", return_value=venv_python
                ), patch(
                    "bootstrap.support.runtime_dependencies_ready", return_value=True
                ), patch("bootstrap.support.os.execve") as mocked_execve, patch(
                    "bootstrap.support.sys.executable", str(venv_python)
                ):
                    bootstrap_helpers.ensure_runtime_ready(project_root=project_root, argv=[])

        mocked_run.assert_not_called()
        mocked_execve.assert_not_called()

    def test_ensure_runtime_ready_treats_pythonw_as_same_venv_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            scripts_dir = project_root / ".venv" / "Scripts"
            venv_python = scripts_dir / "python.exe"
            venv_pythonw = scripts_dir / "pythonw.exe"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")
            venv_pythonw.write_text("", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("bootstrap.support.subprocess.run") as mocked_run, patch(
                    "bootstrap.support.default_venv_python", return_value=venv_python
                ), patch(
                    "bootstrap.support.runtime_dependencies_ready", return_value=True
                ), patch("bootstrap.support.os.execve") as mocked_execve, patch(
                    "bootstrap.support.sys.executable", str(venv_pythonw)
                ):
                    bootstrap_helpers.ensure_runtime_ready(project_root=project_root, argv=[])

        mocked_run.assert_not_called()
        mocked_execve.assert_not_called()

    def test_runtime_dependencies_ready_checks_installed_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            requirements_path = project_root / "requirements.txt"
            requirements_path.write_text("numpy>=1.26.0\nmediapipe\n", encoding="utf-8")

            with patch("bootstrap.support.importlib.util.find_spec") as mocked_find_spec, patch(
                "bootstrap.support.importlib_metadata.version"
            ) as mocked_version:
                mocked_find_spec.side_effect = lambda name: object() if name in {"numpy", "mediapipe"} else None
                mocked_version.side_effect = lambda name: {
                    "numpy": "2.2.6",
                    "mediapipe": "0.10.33",
                }[name]
                self.assertTrue(bootstrap_helpers.runtime_dependencies_ready(project_root))

            with patch("bootstrap.support.importlib.util.find_spec", return_value=None):
                self.assertFalse(bootstrap_helpers.runtime_dependencies_ready(project_root))

    def test_iter_runtime_requirements_supports_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "requirements.txt"
            extra = root / "extra.txt"
            extra.write_text("Pillow>=10.0.0\n", encoding="utf-8")
            base.write_text("numpy>=1.26.0\n-r extra.txt\n", encoding="utf-8")

            requirements = bootstrap_helpers.iter_runtime_requirements(base)

        self.assertEqual(requirements, ["numpy>=1.26.0", "Pillow>=10.0.0"])


if __name__ == "__main__":
    unittest.main()
