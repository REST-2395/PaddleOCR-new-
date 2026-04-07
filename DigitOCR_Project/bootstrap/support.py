"""Helpers for running the bootstrap tool before app startup."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - fallback only used in minimal environments
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover
    Requirement = None  # type: ignore[assignment]


RUNTIME_IMPORT_PROBES = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "paddlepaddle": "paddle",
    "paddleocr": "paddleocr",
    "pillow": "PIL",
    "mediapipe": "mediapipe",
}


def is_frozen_application() -> bool:
    return bool(getattr(sys, "frozen", False))


def detect_platform_name() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def default_venv_python(venv_dir: Path) -> Path:
    if detect_platform_name() == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def default_venv_pythonw(venv_dir: Path) -> Path:
    if detect_platform_name() == "windows":
        return venv_dir / "Scripts" / "pythonw.exe"
    return default_venv_python(venv_dir)


def should_skip_bootstrap(argv: Sequence[str] | None = None) -> bool:
    argv = argv or sys.argv[1:]
    return os.environ.get("DIGITOCR_SKIP_BOOTSTRAP") == "1" or "--skip-bootstrap" in argv


def interpreters_share_runtime(current_python: Path, target_python: Path) -> bool:
    current_resolved = current_python.resolve()
    target_resolved = target_python.resolve()
    if current_resolved == target_resolved:
        return True
    return (
        current_resolved.parent == target_resolved.parent
        and {current_resolved.stem.lower(), target_resolved.stem.lower()} <= {"python", "pythonw"}
    )


def runtime_dependencies_ready(project_root: Path) -> bool:
    requirements_path = project_root / "requirements.txt"
    if not requirements_path.exists():
        return False

    for requirement_text in iter_runtime_requirements(requirements_path):
        package_name = requirement_name(requirement_text)
        if not package_name:
            continue
        if not package_requirement_satisfied(package_name, requirement_text):
            return False
    return True


def iter_runtime_requirements(path: Path, *, _seen: set[Path] | None = None) -> list[str]:
    seen = set() if _seen is None else _seen
    resolved = path.resolve()
    if resolved in seen or not resolved.exists():
        return []
    seen.add(resolved)

    requirements: list[str] = []
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            include_path = line.split(maxsplit=1)[1].strip()
            requirements.extend(iter_runtime_requirements((resolved.parent / include_path).resolve(), _seen=seen))
            continue
        if line.startswith("--"):
            continue
        requirements.append(line)
    return requirements


def requirement_name(requirement_text: str) -> str:
    if Requirement is not None:
        try:
            return Requirement(requirement_text).name.lower()
        except Exception:
            pass

    normalized = requirement_text.split(";", 1)[0].strip()
    if "[" in normalized:
        normalized = normalized.split("[", 1)[0]
    for token in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if token in normalized:
            normalized = normalized.split(token, 1)[0]
            break
    return normalized.strip().lower()


def package_requirement_satisfied(package_name: str, requirement_text: str) -> bool:
    module_name = RUNTIME_IMPORT_PROBES.get(package_name, package_name.replace("-", "_"))
    if importlib.util.find_spec(module_name) is None:
        return False

    if Requirement is None:
        return True

    try:
        requirement = Requirement(requirement_text)
    except Exception:
        return True

    try:
        installed_version = importlib_metadata.version(requirement.name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return installed_version in requirement.specifier if requirement.specifier else True


def build_bootstrap_command(project_root: Path) -> list[str]:
    return [
        sys.executable,
        str(project_root / "bootstrap_env.py"),
        "--project-root",
        str(project_root),
        "--skip-installed",
        "--skip-bootstrap-packages",
    ]


def ensure_runtime_ready(project_root: Path | None = None, argv: Sequence[str] | None = None) -> None:
    """Ensures the runtime environment exists and re-execs into the project venv if needed."""

    if is_frozen_application():
        return

    project_root = project_root or Path(__file__).resolve().parents[1]
    project_root = project_root.resolve()
    target_python = default_venv_python(project_root / ".venv").resolve()
    current_python = Path(sys.executable).resolve()
    if interpreters_share_runtime(current_python, target_python) and runtime_dependencies_ready(project_root):
        return

    if should_skip_bootstrap(argv):
        return

    subprocess.run(build_bootstrap_command(project_root), check=True)

    if interpreters_share_runtime(current_python, target_python):
        return

    script_path = Path(sys.argv[0]).resolve()
    env = os.environ.copy()
    env["DIGITOCR_SKIP_BOOTSTRAP"] = "1"
    os.execve(
        str(target_python),
        [str(target_python), str(script_path), *sys.argv[1:]],
        env,
    )
