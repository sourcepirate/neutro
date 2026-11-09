"""Lightweight documentation checks.

Run from the repository root::

    python scripts/check_docs.py

Checks:

1. Every local Markdown link points at an existing file or directory.
2. Every backticked ``neutro/...py`` source path exists.

This is intentionally dependency-free so it can run in CI before any docs
tooling is installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_GLOBS = ("README.md", "docs/**/*.md")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SOURCE_RE = re.compile(r"`(neutro/[^`\s]+\.py)`")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:")


def _iter_markdown_files():
    seen = set()
    for pattern in DOC_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _strip_anchor(link: str) -> str:
    return link.split("#", 1)[0].strip()


def _strip_fenced_code(text: str) -> str:
    """Remove fenced code blocks so code like ``fn[0](x)`` is not a link."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def check_docs() -> list[str]:
    """Return a list of human-readable problems (empty means clean)."""
    problems: list[str] = []

    for md_path in _iter_markdown_files():
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        rel_md = md_path.relative_to(ROOT)
        prose = _strip_fenced_code(text)

        for raw_link in LINK_RE.findall(prose):
            link = raw_link.strip()
            if not link or link.startswith(SKIP_PREFIXES):
                continue
            if link.startswith("#"):
                continue
            target = _strip_anchor(link)
            if not target:
                continue
            if target.startswith("/"):
                # Absolute site links are outside this repository check.
                continue
            resolved = (md_path.parent / target).resolve()
            if not resolved.exists():
                problems.append(f"{rel_md}: broken link -> {raw_link}")

        for source in SOURCE_RE.findall(text):
            if not (ROOT / source).exists():
                problems.append(f"{rel_md}: missing source path -> {source}")

        # Also catch bare source-path mentions without backticks in prose.
        for source in re.findall(r"(?<![/\w])neutro/[A-Za-z0-9_./-]+\.py\b", text):
            if not (ROOT / source).exists():
                problems.append(f"{rel_md}: missing source path -> {source}")

    return sorted(set(problems))


def main() -> int:
    problems = check_docs()
    if problems:
        print("Documentation check failed:\n")
        for problem in problems:
            print(f"  - {problem}")
        print(f"\n{len(problems)} problem(s) found.")
        return 1

    print("Documentation check passed: links and source paths are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
