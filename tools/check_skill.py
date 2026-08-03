# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Keep skills/using-zorch in lockstep with the released package.

The skill documents the *released* pyzorch surface — every version string and
docs link in the bundle pins one release, so a release cut without touching the
skill ships a guide describing the previous package. This check makes that
impossible to forget: release.yml runs it from the same gate that already
refuses a tag disagreeing with ``zorch.__version__``.

Two layers:

* **Consistency (default)** — offline and deterministic, safe as a release
  gate: every ``pyzorch <version>`` mention and every
  ``blob/v<version>/<path>`` link in the bundle names the packaged version,
  each linked path exists in the tree, relative links resolve, and the
  frontmatter is well-formed.
* **``--imports``** — additionally executes every ``from zorch...`` line in the
  bundle's python fences against the *installed* package. Run this locally
  (or in a wheel-testing job) when bumping the skill for a new release:
  install the new wheel in a venv, then ``python tools/check_skill.py
  --imports``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO / "skills" / "using-zorch"

_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_BLOB = re.compile(r"https://github\.com/fractalyze/zorch/blob/v([0-9.]+)/([^#)]+)")
_PYZORCH_VERSION = re.compile(r"pyzorch[ =]=?\s?([0-9]+(?:\.[0-9]+)+)")


def packaged_version() -> str:
    text = (REPO / "zorch" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if match is None:
        sys.exit("zorch/__init__.py carries no __version__")
    return match.group(1)


def check_consistency(expect: str) -> list[str]:
    problems: list[str] = []
    for md in sorted(SKILL_DIR.rglob("*.md")):
        rel = md.relative_to(REPO)
        text = md.read_text(encoding="utf-8")
        for version in _PYZORCH_VERSION.findall(text):
            if version != expect:
                problems.append(f"{rel}: names pyzorch {version}, packaged {expect}")
        for version, path in _BLOB.findall(text):
            if version != expect:
                problems.append(f"{rel}: links blob/v{version}/{path}, packaged {expect}")
            if not (REPO / path).exists():
                problems.append(f"{rel}: links {path}, which does not exist in the tree")
        for target in _LINK.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            if not (md.parent / target.partition("#")[0]).exists():
                problems.append(f"{rel}: relative link {target} does not resolve")

    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    if re.match(r"^---\n.*?^name:\s*using-zorch\s*$.*?\n---\n", skill_md, re.MULTILINE | re.DOTALL) is None:
        problems.append("SKILL.md: frontmatter missing or name != using-zorch")
    return problems


def check_imports() -> list[str]:
    problems: list[str] = []
    lines: set[str] = set()
    for md in sorted(SKILL_DIR.rglob("*.md")):
        for fence in re.findall(r"```python\n(.*?)```", md.read_text(encoding="utf-8"), re.DOTALL):
            for raw in fence.splitlines():
                line = raw.split("#")[0].strip()
                if line.startswith(("from zorch", "import zorch", "from zk_dtypes")):
                    lines.add(line)
    for line in sorted(lines):
        try:
            exec(line, {})  # noqa: S102 — the whole point: run the documented import
        except Exception as error:  # noqa: BLE001 — any failure is a stale doc
            problems.append(f"{line} -> {type(error).__name__}: {error}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-version",
        default=None,
        help="version the bundle must pin (default: zorch/__init__.py)",
    )
    parser.add_argument(
        "--imports",
        action="store_true",
        help="also execute the bundle's import lines against the installed zorch",
    )
    args = parser.parse_args()

    expect = args.expect_version or packaged_version()
    problems = check_consistency(expect)
    if args.imports:
        problems += check_imports()

    if problems:
        print(f"skills/using-zorch is out of lockstep with pyzorch {expect}:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(
            "Update the bundle for this release (bump version strings and "
            "blob/v links, re-run with --imports against the new wheel), "
            "then re-cut."
        )
    print(f"skills/using-zorch is in lockstep with pyzorch {expect}")


if __name__ == "__main__":
    main()
