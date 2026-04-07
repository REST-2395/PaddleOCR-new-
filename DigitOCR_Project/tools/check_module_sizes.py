"""Check repository source modules against the Phase 8 size guardrails."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_LINES = 700
MAX_CLASS_LINES = 350
MAX_FUNCTION_LINES = 80
SOURCE_DIRS = ("bootstrap", "camera", "core", "desktop")
ROOT_FILES = (
    "gui_app.pyw",
    "main.py",
)


@dataclass(frozen=True)
class SizeRecord:
    kind: str
    path: Path
    name: str
    lines: int


class SizeCollector(ast.NodeVisitor):
    def __init__(self, path: Path, source_lines: list[str]) -> None:
        self.path = path
        self.source_lines = source_lines
        self.stack: list[str] = []
        self.records: list[SizeRecord] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._record("class", node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._record("function", node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._record("function", node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _record(self, kind: str, node: ast.AST) -> None:
        end_lineno = getattr(node, "end_lineno", None)
        lineno = getattr(node, "lineno", None)
        if end_lineno is None or lineno is None:
            return
        qualified_name = ".".join([*self.stack, getattr(node, "name", "<unknown>")])
        line_count = count_nonblank_lines(self.source_lines[lineno - 1 : end_lineno])
        self.records.append(SizeRecord(kind=kind, path=self.path, name=qualified_name, lines=line_count))


def iter_source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_name in ROOT_FILES:
        path = repo_root / relative_name
        if path.exists():
            files.append(path)
    for relative_dir in SOURCE_DIRS:
        root = repo_root / relative_dir
        if not root.exists():
            continue
        for pattern in ("*.py", "*.pyw"):
            files.extend(path for path in root.rglob(pattern) if "__pycache__" not in path.parts)
    return sorted(set(files))


def relative_path(repo_root: Path, path: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def count_nonblank_lines(lines: list[str]) -> int:
    return sum(1 for line in lines if line.strip())


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    warnings: list[str] = []
    scanned_files = iter_source_files(repo_root)

    for path in scanned_files:
        source = path.read_text(encoding="utf-8-sig")
        source_lines = source.splitlines()
        line_count = count_nonblank_lines(source_lines)
        display_path = relative_path(repo_root, path)
        if line_count > MAX_FILE_LINES:
            failures.append(f"file {display_path} has {line_count} lines (limit {MAX_FILE_LINES})")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"syntax error in {display_path}: {exc.msg} at line {exc.lineno}")
            continue
        collector = SizeCollector(path, source_lines)
        collector.visit(tree)
        for record in collector.records:
            if record.kind == "function" and record.lines > MAX_FUNCTION_LINES:
                warnings.append(
                    f"function {display_path}:{record.name} has {record.lines} lines (limit {MAX_FUNCTION_LINES})"
                )
            if record.kind == "class" and record.lines > MAX_CLASS_LINES:
                warnings.append(
                    f"class {display_path}:{record.name} has {record.lines} lines (limit {MAX_CLASS_LINES})"
                )

    print(f"Scanned {len(scanned_files)} source files.")
    if warnings:
        print("Warnings:")
        for message in sorted(warnings):
            print(f"  - {message}")
    else:
        print("Warnings: none")
    if failures:
        print("Failures:")
        for message in sorted(failures):
            print(f"  - {message}")
        return 1
    print("Failures: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
