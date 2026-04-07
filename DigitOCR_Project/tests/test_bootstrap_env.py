from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap_env


class _RecordingRunner:
    def __init__(self, *, fail_download: bool = False) -> None:
        self.fail_download = fail_download
        self.calls: list[dict[str, object]] = []

    def run(self, command, *, description: str, cwd=None, env=None, retryable: bool = True):
        self.calls.append(
            {
                "command": command,
                "description": description,
                "cwd": cwd,
                "env": env,
                "retryable": retryable,
            }
        )
        if self.fail_download and isinstance(command, list) and "download" in command:
            raise bootstrap_env.BootstrapError("simulated download failure")
        return subprocess.CompletedProcess(command, 0, "", "")


class BootstrapEnvTests(unittest.TestCase):
    def test_parse_requirements_file_supports_nested_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "requirements.txt"
            extra = root / "extra.txt"
            extra.write_text("Pillow>=10.0.0\n", encoding="utf-8")
            base.write_text(
                "\n".join(
                    [
                        "# comment",
                        "numpy>=1.26.0",
                        "-r extra.txt",
                        "--extra-index-url https://example.com/simple",
                    ]
                ),
                encoding="utf-8",
            )

            packages = bootstrap_env.parse_requirements_file(base)

        self.assertEqual(packages, ["numpy>=1.26.0", "Pillow>=10.0.0"])

    def test_parse_package_json_respects_dev_dependency_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_json = Path(temp_dir) / "package.json"
            package_json.write_text(
                json.dumps(
                    {
                        "dependencies": {"react": "^18.0.0"},
                        "optionalDependencies": {"sharp": "^1.0.0"},
                        "devDependencies": {"vitest": "^2.0.0"},
                    }
                ),
                encoding="utf-8",
            )

            runtime_packages = bootstrap_env.parse_package_json(package_json, include_dev_dependencies=False)
            all_packages = bootstrap_env.parse_package_json(package_json, include_dev_dependencies=True)

        self.assertEqual(runtime_packages, ["react@^18.0.0", "sharp@^1.0.0"])
        self.assertEqual(all_packages, ["react@^18.0.0", "sharp@^1.0.0", "vitest@^2.0.0"])

    def test_command_runner_retries_after_failure(self) -> None:
        logger = logging.getLogger("bootstrap_env_test_retry")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        runner = bootstrap_env.CommandRunner(
            logger=logger,
            retries=1,
            retry_delay=0.01,
            timeout_seconds=10,
            dry_run=False,
        )
        results = [
            subprocess.CompletedProcess(["python"], 1, "", "network error"),
            subprocess.CompletedProcess(["python"], 0, "ok", ""),
        ]

        with patch("bootstrap_env.subprocess.run", side_effect=results) as mocked_run, patch("bootstrap_env.time.sleep") as mocked_sleep:
            completed = runner.run(["python", "--version"], description="retry test")

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(mocked_run.call_count, 2)
        mocked_sleep.assert_called_once()

    def test_install_pip_manifest_falls_back_to_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "requirements.txt"
            manifest_path.write_text("numpy>=1.26.0\n", encoding="utf-8")
            cache_dir = root / "cache"
            cached_wheel_dir = cache_dir / "pip"
            cached_wheel_dir.mkdir(parents=True, exist_ok=True)
            (cached_wheel_dir / "numpy.whl").write_text("cached", encoding="utf-8")

            manifest = bootstrap_env.ManifestSpec(
                name="requirements.txt",
                path=manifest_path,
                manager="pip",
                manifest_type="requirements",
                cache_subdir="pip",
            )
            runner = _RecordingRunner(fail_download=True)

            bootstrap_env.install_pip_manifest(
                manifest,
                runner=runner,
                venv_python=root / ".venv" / "python",
                cache_dir=cache_dir,
                offline=False,
                force_reinstall=False,
            )

        self.assertEqual(len(runner.calls), 2)
        install_command = runner.calls[1]["command"]
        assert isinstance(install_command, list)
        self.assertIn("--no-index", install_command)

    def test_main_dry_run_generates_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            config_dir = project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (project_root / "requirements.txt").write_text("numpy>=1.26.0\n", encoding="utf-8")
            (config_dir / "env_bootstrap.json").write_text(
                json.dumps(
                    {
                        "project_name": "demo",
                        "python": {"min_version": "3.10", "venv_dir": ".venv", "bootstrap_packages": ["pip"]},
                        "manifests": [
                            {
                                "name": "python-runtime",
                                "path": "requirements.txt",
                                "manager": "pip",
                                "type": "requirements",
                                "optional": False,
                                "cache_subdir": "pip",
                            }
                        ],
                        "system_dependencies": [],
                    }
                ),
                encoding="utf-8",
            )
            log_file = project_root / "bootstrap.log"
            report_file = project_root / "bootstrap.json"

            with patch("bootstrap_env.ensure_virtualenv", return_value=project_root / ".venv" / "bin" / "python"):
                exit_code = bootstrap_env.main(
                    [
                        "--project-root",
                        str(project_root),
                        "--log-file",
                        str(log_file),
                        "--json-report",
                        str(report_file),
                        "--skip-system-deps",
                        "--skip-bootstrap-packages",
                        "--dry-run",
                    ]
                )

            logger = logging.getLogger("bootstrap_env")
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

            self.assertEqual(exit_code, 0)
            self.assertTrue(log_file.exists())
            self.assertTrue(report_file.exists())
            report_payload = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertTrue(report_payload["success"])


if __name__ == "__main__":
    unittest.main()
