"""Check that the public repository contains no broken local Markdown links or local paths."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
WINDOWS_USER_PATH = re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE)
SKIP_PARTS = {
    ".git",
    ".venv",
    ".runtime",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "runs",
}


def public_files(pattern: str):
    return (
        path
        for path in ROOT.rglob(pattern)
        if not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
    )


def markdown_errors() -> list[str]:
    errors: list[str] = []
    for document in public_files("*.md"):
        text = document.read_text(encoding="utf-8")
        if WINDOWS_USER_PATH.search(text):
            errors.append(f"local Windows user path in {document.relative_to(ROOT)}")
        for raw in LINK.findall(text):
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (document.parent / target).resolve().exists():
                errors.append(
                    f"broken link in {document.relative_to(ROOT)}: {raw}"
                )
    return errors


def notebook_errors() -> list[str]:
    errors: list[str] = []
    for notebook in public_files("*.ipynb"):
        text = notebook.read_text(encoding="utf-8")
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid notebook JSON in {notebook.relative_to(ROOT)}: {exc}")
        if WINDOWS_USER_PATH.search(text):
            errors.append(f"local Windows user path in {notebook.relative_to(ROOT)}")
    return errors


def local_path_errors() -> list[str]:
    errors: list[str] = []
    text_suffixes = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
    for path in public_files("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if WINDOWS_USER_PATH.search(text):
            errors.append(f"local Windows user path in {path.relative_to(ROOT)}")
    return errors


def tree_errors() -> list[str]:
    errors: list[str] = []
    for path in public_files("*"):
        relative = path.relative_to(ROOT)
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"file exceeds 50 MiB: {relative}")
    return errors


def main() -> int:
    errors = [*markdown_errors(), *notebook_errors(), *local_path_errors(), *tree_errors()]
    if errors:
        print("Public-repository checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Public-repository checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
