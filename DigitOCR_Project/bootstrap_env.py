#!/usr/bin/env python3
"""Cross-platform runtime bootstrap for DigitOCR_Project."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import venv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "env_bootstrap.json"
DEFAULT_RUNTIME_DIR = Path(".runtime") / "bootstrap"
DEFAULT_CACHE_DIR = Path(".runtime") / "cache"
SUPPORTED_MANIFEST_FILES: dict[str, tuple[str, str]] = {
    "requirements.txt": ("pip", "requirements"),
    "package.json": ("npm", "package-json"),
}
SYSTEM_MANAGER_ORDER: dict[str, tuple[str, ...]] = {
    "windows": ("winget",),
    "macos": ("brew",),
    "linux": ("apt", "dnf", "yum", "pacman"),
}


class BootstrapError(RuntimeError):
    """Raised when runtime bootstrapping cannot complete."""


@dataclass(slots=True)
class ManifestSpec:
    name: str
    path: Path
    manager: str
    manifest_type: str
    optional: bool = False
    include_dev_dependencies: bool = False
    cache_subdir: str | None = None


@dataclass(slots=True)
class SystemDependencySpec:
    name: str
    check_commands: list[str]
    package_names: dict[str, str]
    platforms: list[str] = field(default_factory=lambda: ["windows", "macos", "linux"])
    manifest_managers: list[str] = field(default_factory=list)
    optional: bool = False


@dataclass(slots=True)
class BootstrapConfig:
    project_name: str
    min_python_version: str
    venv_dir: Path
    bootstrap_packages: list[str]
    manifests: list[ManifestSpec]
    system_dependencies: list[SystemDependencySpec]


@dataclass(slots=True)
class StepRecord:
    name: str
    status: str
    started_at: float
    finished_at: float
    detail: str = ""


@dataclass(slots=True)
class ManifestState:
    fingerprint: str
    installed_at: str
    manager: str
    packages: list[str]


@dataclass(slots=True)
class BootstrapState:
    platform: str
    python_version: str
    manifests: dict[str, ManifestState] = field(default_factory=dict)


class CommandRunner:
    """Runs subprocess commands with retry, timeout, and logging."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        retries: int,
        retry_delay: float,
        timeout_seconds: int,
        dry_run: bool,
    ) -> None:
        self.logger = logger
        self.retries = max(0, retries)
        self.retry_delay = max(0.0, retry_delay)
        self.timeout_seconds = timeout_seconds
        self.dry_run = dry_run

    @staticmethod
    def _render_command(command: Sequence[str] | str) -> str:
        if isinstance(command, str):
            return command
        return subprocess.list2cmdline([str(part) for part in command])

    def run(
        self,
        command: Sequence[str] | str,
        *,
        description: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        retryable: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        rendered = self._render_command(command)
        attempts = self.retries + 1 if retryable else 1
        last_error: BootstrapError | None = None

        for attempt in range(1, attempts + 1):
            self.logger.info("[%s] %s", description, rendered)
            if self.dry_run:
                return subprocess.CompletedProcess(command, 0, "", "")

            try:
                completed = subprocess.run(
                    command,
                    cwd=str(cwd) if cwd else None,
                    env=env,
                    capture_output=True,
                    text=True,
                    shell=isinstance(command, str),
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_error = BootstrapError(f"{description} failed before completion: {exc}")
                self.logger.warning("%s", last_error)
            else:
                if completed.stdout.strip():
                    self.logger.info("%s stdout:\n%s", description, completed.stdout.strip())
                if completed.stderr.strip():
                    self.logger.warning("%s stderr:\n%s", description, completed.stderr.strip())
                if completed.returncode == 0:
                    return completed
                last_error = BootstrapError(
                    f"{description} failed with exit code {completed.returncode}: {rendered}"
                )
                self.logger.warning("%s", last_error)

            if attempt < attempts:
                delay = self.retry_delay * attempt
                self.logger.info("Retrying %s in %.1f seconds (attempt %s/%s).", description, delay, attempt + 1, attempts)
                time.sleep(delay)

        raise last_error or BootstrapError(f"{description} failed: {rendered}")


def detect_platform_name() -> str:
    system_name = platform.system().lower()
    if system_name.startswith("win"):
        return "windows"
    if system_name == "darwin":
        return "macos"
    return "linux"


def resolve_project_root(project_root_arg: str | None) -> Path:
    if project_root_arg:
        return Path(project_root_arg).resolve()
    return Path(__file__).resolve().parents[1]


def default_config_payload(project_root: Path) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    for file_name, (manager, manifest_type) in SUPPORTED_MANIFEST_FILES.items():
        path = project_root / file_name
        manifests.append(
            {
                "name": path.name,
                "path": file_name,
                "manager": manager,
                "type": manifest_type,
                "optional": not path.exists(),
                "include_dev_dependencies": False,
                "cache_subdir": manager,
            }
        )

    return {
        "project_name": project_root.name,
        "python": {
            "min_version": "3.10",
            "venv_dir": ".venv",
            "bootstrap_packages": ["pip", "setuptools", "wheel"],
        },
        "manifests": manifests,
        "system_dependencies": [
            {
                "name": "node-runtime",
                "optional": True,
                "platforms": ["windows", "macos", "linux"],
                "manifest_managers": ["npm"],
                "check_commands": ["node --version", "npm --version"],
                "package_names": {
                    "winget": "OpenJS.NodeJS.LTS",
                    "brew": "node",
                    "apt": "nodejs npm",
                    "dnf": "nodejs npm",
                    "yum": "nodejs npm",
                    "pacman": "nodejs npm",
                },
            }
        ],
    }


def infer_manifest_definition(path: Path) -> tuple[str, str]:
    file_name = path.name.lower()
    if file_name not in SUPPORTED_MANIFEST_FILES:
        raise BootstrapError(f"Unsupported manifest file: {path}")
    return SUPPORTED_MANIFEST_FILES[file_name]


def load_config(project_root: Path, config_path_arg: str | None) -> BootstrapConfig:
    config_path = project_root / DEFAULT_CONFIG_RELATIVE_PATH
    if config_path_arg:
        config_path = Path(config_path_arg).resolve()

    payload = default_config_payload(project_root)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    python_config = payload.get("python", {})
    manifests: list[ManifestSpec] = []
    for raw_manifest in payload.get("manifests", []):
        manifest_path = project_root / raw_manifest["path"]
        manager = raw_manifest.get("manager")
        manifest_type = raw_manifest.get("type")
        if not manager or not manifest_type:
            manager, manifest_type = infer_manifest_definition(manifest_path)
        manifests.append(
            ManifestSpec(
                name=raw_manifest.get("name", manifest_path.name),
                path=manifest_path,
                manager=manager,
                manifest_type=manifest_type,
                optional=bool(raw_manifest.get("optional", False)),
                include_dev_dependencies=bool(raw_manifest.get("include_dev_dependencies", False)),
                cache_subdir=raw_manifest.get("cache_subdir"),
            )
        )

    system_dependencies = [
        SystemDependencySpec(
            name=item["name"],
            check_commands=list(item.get("check_commands", [])),
            package_names=dict(item.get("package_names", {})),
            platforms=list(item.get("platforms", ["windows", "macos", "linux"])),
            manifest_managers=list(item.get("manifest_managers", [])),
            optional=bool(item.get("optional", False)),
        )
        for item in payload.get("system_dependencies", [])
    ]

    return BootstrapConfig(
        project_name=payload.get("project_name", project_root.name),
        min_python_version=python_config.get("min_version", "3.10"),
        venv_dir=project_root / python_config.get("venv_dir", ".venv"),
        bootstrap_packages=list(python_config.get("bootstrap_packages", ["pip", "setuptools", "wheel"])),
        manifests=manifests,
        system_dependencies=system_dependencies,
    )


def parse_version_string(version_text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version_text.split(".") if part.isdigit())


def ensure_python_version(min_version: str) -> None:
    current = sys.version_info[:3]
    expected = parse_version_string(min_version)
    if current < expected:
        raise BootstrapError(
            f"Python {min_version}+ is required, but the current interpreter is {platform.python_version()}."
        )


def make_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bootstrap_env")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def load_state(state_path: Path) -> BootstrapState:
    if not state_path.exists():
        return BootstrapState(platform=detect_platform_name(), python_version=platform.python_version())

    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    manifests = {
        key: ManifestState(
            fingerprint=value["fingerprint"],
            installed_at=value["installed_at"],
            manager=value["manager"],
            packages=list(value.get("packages", [])),
        )
        for key, value in payload.get("manifests", {}).items()
    }
    return BootstrapState(
        platform=payload.get("platform", detect_platform_name()),
        python_version=payload.get("python_version", platform.python_version()),
        manifests=manifests,
    )


def save_state(state_path: Path, state: BootstrapState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "platform": state.platform,
        "python_version": state.python_version,
        "manifests": {key: asdict(value) for key, value in state.manifests.items()},
    }
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_requirements_file(path: Path, *, _seen: set[Path] | None = None) -> list[str]:
    _seen = _seen or set()
    resolved = path.resolve()
    if resolved in _seen:
        return []
    _seen.add(resolved)

    requirements: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r ", "--requirement ")):
                include_path = line.split(maxsplit=1)[1]
                requirements.extend(parse_requirements_file((path.parent / include_path).resolve(), _seen=_seen))
                continue
            if line.startswith(("--", "-f ", "-i ", "--index-url", "--extra-index-url")):
                continue
            requirements.append(line)
    return requirements


def parse_package_json(path: Path, *, include_dev_dependencies: bool) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    dependencies = dict(payload.get("dependencies", {}))
    dependencies.update(payload.get("optionalDependencies", {}))
    if include_dev_dependencies:
        dependencies.update(payload.get("devDependencies", {}))
    return [f"{name}@{version}" for name, version in sorted(dependencies.items())]


def parse_manifest_packages(manifest: ManifestSpec) -> list[str]:
    if manifest.manager == "pip":
        return parse_requirements_file(manifest.path)
    if manifest.manager == "npm":
        return parse_package_json(manifest.path, include_dev_dependencies=manifest.include_dev_dependencies)
    raise BootstrapError(f"Unsupported manifest manager: {manifest.manager}")


def compute_manifest_fingerprint(manifest: ManifestSpec, packages: list[str], *, platform_name: str, python_version: str) -> str:
    digest = hashlib.sha256()
    digest.update(platform_name.encode("utf-8"))
    digest.update(python_version.encode("utf-8"))
    digest.update(manifest.manager.encode("utf-8"))
    digest.update(manifest.manifest_type.encode("utf-8"))
    digest.update(str(manifest.path.resolve()).encode("utf-8"))
    digest.update(manifest.path.read_bytes())
    digest.update("\n".join(packages).encode("utf-8"))
    return digest.hexdigest()


def default_venv_python(venv_dir: Path) -> Path:
    if detect_platform_name() == "windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_virtualenv(config: BootstrapConfig, logger: logging.Logger) -> Path:
    venv_python = default_venv_python(config.venv_dir)
    if venv_python.exists():
        return venv_python

    logger.info("Creating virtual environment at %s", config.venv_dir)
    builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False)
    builder.create(config.venv_dir)
    if not venv_python.exists():
        raise BootstrapError(f"Virtual environment creation did not produce an interpreter at {venv_python}")
    return venv_python


def detect_system_manager(platform_name: str) -> str | None:
    for manager in SYSTEM_MANAGER_ORDER.get(platform_name, ()):
        executable = "apt-get" if manager == "apt" else manager
        if shutil.which(executable):
            return manager
    return None


def build_system_install_command(manager: str, package_name: str) -> Sequence[str]:
    if manager == "winget":
        return [
            "winget",
            "install",
            "--id",
            package_name,
            "-e",
            "--accept-package-agreements",
            "--accept-source-agreements",
        ]
    if manager == "brew":
        return ["brew", "install", package_name]
    if manager == "apt":
        return ["sudo", "apt-get", "install", "-y", *package_name.split()]
    if manager == "dnf":
        return ["sudo", "dnf", "install", "-y", *package_name.split()]
    if manager == "yum":
        return ["sudo", "yum", "install", "-y", *package_name.split()]
    if manager == "pacman":
        return ["sudo", "pacman", "-S", "--noconfirm", *package_name.split()]
    raise BootstrapError(f"Unsupported system package manager: {manager}")


def is_system_dependency_installed(dep: SystemDependencySpec, runner: CommandRunner, platform_name: str) -> bool:
    if platform_name not in dep.platforms:
        return True
    if not dep.check_commands:
        return False

    for command in dep.check_commands:
        try:
            runner.run(command, description=f"Check system dependency {dep.name}", retryable=False)
        except BootstrapError:
            return False
    return True


def ensure_system_dependencies(
    config: BootstrapConfig,
    *,
    runner: CommandRunner,
    logger: logging.Logger,
    platform_name: str,
    active_managers: set[str],
    skip_system_dependencies: bool,
) -> None:
    if skip_system_dependencies:
        logger.info("Skipping system dependency checks by request.")
        return

    system_manager = detect_system_manager(platform_name)
    for dep in config.system_dependencies:
        if platform_name not in dep.platforms:
            continue
        if dep.manifest_managers and not (active_managers & set(dep.manifest_managers)):
            continue
        if is_system_dependency_installed(dep, runner, platform_name):
            logger.info("System dependency %s is already available.", dep.name)
            continue
        if not system_manager:
            if dep.optional:
                logger.warning("No supported system package manager found for optional dependency %s.", dep.name)
                continue
            raise BootstrapError(f"Cannot install required system dependency {dep.name}: no package manager found.")

        package_name = dep.package_names.get(system_manager)
        if not package_name:
            if dep.optional:
                logger.warning("No package mapping for optional dependency %s under %s.", dep.name, system_manager)
                continue
            raise BootstrapError(f"No package mapping for required dependency {dep.name} under {system_manager}.")

        install_command = build_system_install_command(system_manager, package_name)
        runner.run(install_command, description=f"Install system dependency {dep.name}")


def upgrade_bootstrap_packages(
    *,
    runner: CommandRunner,
    logger: logging.Logger,
    venv_python: Path,
    cache_dir: Path,
    bootstrap_packages: list[str],
    skip_bootstrap_packages: bool,
) -> None:
    if skip_bootstrap_packages:
        logger.info("Skipping bootstrap package upgrade by request.")
        return
    command = [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--cache-dir",
        str(cache_dir / "pip-http"),
        *bootstrap_packages,
    ]
    runner.run(command, description="Upgrade bootstrap packages")


def install_pip_manifest(
    manifest: ManifestSpec,
    *,
    runner: CommandRunner,
    venv_python: Path,
    cache_dir: Path,
    offline: bool,
    force_reinstall: bool,
) -> None:
    wheel_cache_dir = cache_dir / (manifest.cache_subdir or "pip")
    wheel_cache_dir.mkdir(parents=True, exist_ok=True)

    effective_offline = offline
    if not offline:
        download_command = [
            str(venv_python),
            "-m",
            "pip",
            "download",
            "--requirement",
            str(manifest.path),
            "--dest",
            str(wheel_cache_dir),
            "--cache-dir",
            str(cache_dir / "pip-http"),
            "--prefer-binary",
        ]
        try:
            runner.run(download_command, description=f"Cache packages for {manifest.name}")
        except BootstrapError:
            if any(wheel_cache_dir.iterdir()):
                effective_offline = True
            else:
                raise

    install_command = [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--requirement",
        str(manifest.path),
        "--cache-dir",
        str(cache_dir / "pip-http"),
        "--find-links",
        str(wheel_cache_dir),
        "--prefer-binary",
    ]
    if effective_offline:
        install_command.append("--no-index")
    if force_reinstall:
        install_command.extend(["--upgrade", "--force-reinstall"])
    runner.run(install_command, description=f"Install Python dependencies from {manifest.name}")


def install_npm_manifest(
    manifest: ManifestSpec,
    *,
    runner: CommandRunner,
    cache_dir: Path,
    offline: bool,
    force_reinstall: bool,
) -> None:
    npm_executable = shutil.which("npm")
    if not npm_executable:
        raise BootstrapError("npm was not found in PATH.")

    lock_file = manifest.path.parent / "package-lock.json"
    command = [
        npm_executable,
        "ci" if lock_file.exists() else "install",
        "--cache",
        str(cache_dir / (manifest.cache_subdir or "npm")),
        "--prefer-offline",
        "--no-audit",
        "--fund=false",
    ]
    if not manifest.include_dev_dependencies:
        command.append("--omit=dev")
    if offline:
        command.append("--offline")
    if force_reinstall:
        command.append("--force")
    runner.run(command, description=f"Install Node dependencies from {manifest.name}", cwd=manifest.path.parent)


def install_manifest(
    manifest: ManifestSpec,
    *,
    runner: CommandRunner,
    logger: logging.Logger,
    venv_python: Path,
    cache_dir: Path,
    offline: bool,
    force_reinstall: bool,
) -> None:
    logger.info("Installing dependencies declared in %s", manifest.path)
    if manifest.manager == "pip":
        install_pip_manifest(
            manifest,
            runner=runner,
            venv_python=venv_python,
            cache_dir=cache_dir,
            offline=offline,
            force_reinstall=force_reinstall,
        )
        return
    if manifest.manager == "npm":
        install_npm_manifest(
            manifest,
            runner=runner,
            cache_dir=cache_dir,
            offline=offline,
            force_reinstall=force_reinstall,
        )
        return
    raise BootstrapError(f"Unsupported manifest manager: {manifest.manager}")


def write_report(
    report_path: Path,
    *,
    success: bool,
    config: BootstrapConfig,
    steps: list[StepRecord],
    platform_name: str,
    state: BootstrapState,
    log_file: Path,
    error_message: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_name": config.project_name,
        "success": success,
        "platform": platform_name,
        "python_version": platform.python_version(),
        "venv_dir": str(config.venv_dir),
        "log_file": str(log_file),
        "error_message": error_message,
        "steps": [asdict(step) for step in steps],
        "state": {
            "platform": state.platform,
            "python_version": state.python_version,
            "manifests": {key: asdict(value) for key, value in state.manifests.items()},
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and install DigitOCR runtime dependencies.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root to bootstrap. Defaults to this script directory.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config path.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory.")
    parser.add_argument("--log-file", type=str, default=None, help="Override bootstrap log path.")
    parser.add_argument("--json-report", type=str, default=None, help="Override JSON report output path.")
    parser.add_argument("--offline", action="store_true", help="Install only from existing cache when possible.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count for failed download/install commands.")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Base delay in seconds between retries.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-command timeout in seconds.")
    parser.add_argument("--force-reinstall", action="store_true", help="Force reinstallation of manifest dependencies.")
    parser.add_argument("--skip-installed", action="store_true", help="Skip manifests whose fingerprints match the last successful run.")
    parser.add_argument("--skip-system-deps", action="store_true", help="Skip OS-level dependency detection/installation.")
    parser.add_argument("--skip-bootstrap-packages", action="store_true", help="Skip pip/setuptools/wheel upgrade inside the virtual environment.")
    parser.add_argument("--dry-run", action="store_true", help="Log planned actions without executing commands.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = resolve_project_root(args.project_root)
    runtime_dir = project_root / DEFAULT_RUNTIME_DIR
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else project_root / DEFAULT_CACHE_DIR
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = Path(args.log_file).resolve() if args.log_file else runtime_dir / "logs" / f"bootstrap-{timestamp}.log"
    report_path = (
        Path(args.json_report).resolve()
        if args.json_report
        else runtime_dir / "reports" / f"bootstrap-{timestamp}.json"
    )
    state_path = runtime_dir / "state.json"
    config = load_config(project_root, args.config)
    ensure_python_version(config.min_python_version)

    logger = make_logger(log_file)
    runner = CommandRunner(
        logger=logger,
        retries=args.retries,
        retry_delay=args.retry_delay,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
    )
    steps: list[StepRecord] = []
    state = load_state(state_path)
    platform_name = detect_platform_name()

    def run_step(name: str, action: Any) -> None:
        started_at = time.time()
        detail = ""
        try:
            result = action()
            if isinstance(result, str):
                detail = result
        except Exception as exc:
            steps.append(
                StepRecord(
                    name=name,
                    status="failed",
                    started_at=started_at,
                    finished_at=time.time(),
                    detail=str(exc),
                )
            )
            raise
        steps.append(
            StepRecord(
                name=name,
                status="completed",
                started_at=started_at,
                finished_at=time.time(),
                detail=detail,
            )
        )

    try:
        run_step("load-config", lambda: f"Loaded {len(config.manifests)} manifest definition(s).")
        venv_python = ensure_virtualenv(config, logger)
        run_step("ensure-venv", lambda: f"Using interpreter {venv_python}")
        run_step(
            "ensure-system-dependencies",
            lambda: (
                ensure_system_dependencies(
                    config,
                    runner=runner,
                    logger=logger,
                    platform_name=platform_name,
                    active_managers={manifest.manager for manifest in config.manifests if manifest.path.exists()},
                    skip_system_dependencies=args.skip_system_deps,
                )
                or "System dependency checks completed."
            ),
        )
        run_step(
            "upgrade-bootstrap-packages",
            lambda: (
                upgrade_bootstrap_packages(
                    runner=runner,
                    logger=logger,
                    venv_python=venv_python,
                    cache_dir=cache_dir,
                    bootstrap_packages=config.bootstrap_packages,
                    skip_bootstrap_packages=args.skip_bootstrap_packages,
                )
                or "Bootstrap packages ready."
            ),
        )

        for manifest in config.manifests:
            if not manifest.path.exists():
                if manifest.optional:
                    logger.info("Skipping optional manifest %s because it does not exist.", manifest.path)
                    continue
                raise BootstrapError(f"Required manifest not found: {manifest.path}")

            packages = parse_manifest_packages(manifest)
            fingerprint = compute_manifest_fingerprint(
                manifest,
                packages,
                platform_name=platform_name,
                python_version=platform.python_version(),
            )
            state_key = str(manifest.path.relative_to(project_root)).replace("\\", "/")
            previous_state = state.manifests.get(state_key)
            if (
                args.skip_installed
                and previous_state
                and previous_state.fingerprint == fingerprint
                and previous_state.manager == manifest.manager
            ):
                logger.info("Skipping %s because the manifest fingerprint matches the last successful install.", manifest.path)
                continue

            run_step(
                f"install:{state_key}",
                lambda manifest=manifest: (
                    install_manifest(
                        manifest,
                        runner=runner,
                        logger=logger,
                        venv_python=venv_python,
                        cache_dir=cache_dir,
                        offline=args.offline,
                        force_reinstall=args.force_reinstall,
                    )
                    or f"Installed dependencies from {manifest.path}"
                ),
            )
            state.manifests[state_key] = ManifestState(
                fingerprint=fingerprint,
                installed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                manager=manifest.manager,
                packages=packages,
            )

        if not args.dry_run:
            save_state(
                state_path,
                BootstrapState(
                    platform=platform_name,
                    python_version=platform.python_version(),
                    manifests=state.manifests,
                ),
            )
        write_report(
            report_path,
            success=True,
            config=config,
            steps=steps,
            platform_name=platform_name,
            state=state,
            log_file=log_file,
        )
        logger.info("Bootstrap completed successfully.")
        logger.info("JSON report: %s", report_path)
        return 0
    except Exception as exc:
        logger.exception("Bootstrap failed: %s", exc)
        write_report(
            report_path,
            success=False,
            config=config,
            steps=steps,
            platform_name=platform_name,
            state=state,
            log_file=log_file,
            error_message=str(exc),
        )
        logger.info("Failure report: %s", report_path)
        return 1
    finally:
        close_logger(logger)


if __name__ == "__main__":
    sys.exit(main())
