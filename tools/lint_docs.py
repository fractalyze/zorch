# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Check that documentation references resolve, as a pre-commit hook.

Only the two questions the filesystem answers objectively: does a cited symbol
exist anywhere in the tree, and does a relative link resolve from the page
carrying it. Both catch what prose review misses, because a symbol that was
renamed or never existed reads exactly like one that is real.

Whether a page explains the right things is not decidable here and is not
attempted — `.claude/skills/lint-zorch-docs` covers that, by reading.
"""

from __future__ import annotations

import builtins
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Test scaffolding and leaf helpers ship no design block of their own.
SUPPORT_DIRS = frozenset({"testing", "testkit", "utils", "__pycache__"})

# Skill files are in scope because a lint written as prose rots exactly like the
# docs it polices: `commit/jagged` was a path one of them invented.
DOC_GLOBS = ("docs/**/*.md", "README.md", "CLAUDE.md", ".claude/skills/**/*.md")


@dataclass(frozen=True)
class Finding:
    check: str
    path: Path
    message: str
    line: int = 0

    def __str__(self) -> str:
        where = f"{self.path.relative_to(REPO)}"
        if self.line:
            where += f":{self.line}"
        return f"{where}: [{self.check}] {self.message}"


def _docs() -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in DOC_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            seen[path] = None
    return list(seen)


# ---------------------------------------------------------------------------
# dangling-reference
# ---------------------------------------------------------------------------

# Owned by another project, so this tree cannot resolve their spelling.
EXTERNAL_PREFIXES = (
    "jnp.",
    "lax.",
    "np.",
    "jax.",
    "frx.",
    "zk_dtypes.",
    # Config flags and tracing errors owned by JAX, spelled without a dot.
    "jax_",
    "jit_",
    "xla_",
    # StableHLO dialect op spellings (`stablehlo.shift_right_arithmetic`, …):
    # owned by the compiler stack; a doc quotes them from lowering errors.
    "stablehlo.",
)

# Op names owned by the compiler stack rather than by a Python module: docs cite
# the HLO/lowering spelling, which no source file in this tree contains.
EXTERNAL_NAMES = frozenset(
    {
        "reduce_sum",
        "reduce_window",
        "dynamic_slice",
        # Raised by JAX's tracer, not defined here.
        "ConcretizationTypeError",
        "TracerBoolConversionError",
        # A sibling repository, and a harness tool the skills invoke.
        "whir_zorch",
        "AskUserQuestion",
    }
)

# Backticks that are deliberately not citations: a name shown as an example of
# what not to write, or a path in another project. Keyed by page so an alibi
# cannot travel, and expiring once the page stops citing the token.
FOREIGN_CITATIONS = {
    ("docs/reference/conventions.md", "_sp1_input_hash"): (
        "an invented name, shown as the downstream-leaking spelling to avoid"
    ),
    ("docs/reference/conventions.md", "SplitMix64"): (
        "an external generator named as the kind of machinery a golden test "
        "should not drag in"
    ),
    ("docs/blocks/pcs.md", "PolynomialSpace"): (
        "a Plonky3 concept, named as the shape this seam deliberately lacks"
    ),
    ("docs/reference/development.md", "docs/build_from_source.md"): (
        "a page in the XLA repository, cited for the build it documents"
    ),
}

_BACKTICKED = re.compile(r"`([^`\n]+)`")
_MD_LINK = re.compile(r"\[([^\]\n]*)\]\(([^)\s]+)\)")
# A path citation: a tree path, optionally with a line or anchor suffix.
_PATH_LIKE = re.compile(r"^[\w./-]+\.(py|bzl|md|toml|bazel|yaml|yml|in|txt)$")
# An identifier this tree would define: CamelCase types and snake_case callables.
_CAMEL = re.compile(r"^_?[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+$")
_SNAKE = re.compile(r"^_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


# The lint's own tests are excluded: they name fictional symbols on purpose, and
# a fixture must not vouch for the symbol it was written to catch.
SOURCE_GLOBS = (
    "zorch/**/*.py",
    "**/BUILD.bazel",
    "*.bzl",
    "pyproject.toml",
    "requirements*.in",
    ".github/workflows/*.yml",
)


def _source_words() -> set[str]:
    """Every identifier-shaped token in the source tree.

    Coarser than a definition table on purpose — a composite marker name lives in
    a string literal — but it still catches the failure that matters: a page
    describing a symbol that appears nowhere at all.
    """
    words = set(dir(builtins))
    for pattern in SOURCE_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            words.add(path.stem)
            words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return words


def _candidate(token: str) -> str | None:
    """The resolvable core of a backticked span, or None if it is prose."""
    token = token.strip()
    if not token or " " in token:
        return None
    # A glob or a shell placeholder names a shape, not a file in this tree.
    if any(ch in token for ch in "*?<>$~"):
        return None
    token = token.removesuffix("()")
    token = token.split("#", 1)[0]
    if token.startswith(EXTERNAL_PREFIXES) or token in EXTERNAL_NAMES:
        return None
    if _PATH_LIKE.match(token) or token.endswith("/"):
        return token
    # A dotted or slashed span resolves through its last segment; the leading
    # segments are package structure the path check already covers.
    tail = re.split(r"[./]", token)[-1]
    if _CAMEL.match(tail) or _SNAKE.match(tail):
        return tail
    return None


def _resolves_as_path(doc: Path, token: str) -> bool:
    """A citation reads from where it is written, or from a tree root.

    Pages cite siblings relatively and source subsystems by package-relative
    path, so both origins count; a bare filename is shorthand for the one file of
    that name, which still fails once that file is renamed.
    """
    bases = (doc.parent, REPO, REPO / "zorch", REPO / "docs")
    if any((base / token).exists() for base in bases):
        return True
    return "/" not in token and any(REPO.glob(f"**/{token}"))


def check_link_targets() -> Iterator[Finding]:
    """Every relative markdown link resolves from the page carrying it.

    A moved page leaves the label reading correctly while the target points at
    nothing — live text over dead links.
    """
    for doc in _docs():
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for _, target in _MD_LINK.findall(line):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                path = target.split("#", 1)[0]
                if not path or (doc.parent / path).exists():
                    continue
                yield Finding(
                    "link-target",
                    doc,
                    f"links {target}, which does not resolve from this page",
                    lineno,
                )


def check_dangling_references() -> Iterator[Finding]:
    """Every path and symbol a page cites resolves in the tree.

    Catches a page describing code that was renamed or never existed — what prose
    review misses, because a plausible symbol reads exactly like a real one.
    """
    words = _source_words()
    cited_foreign = set(FOREIGN_CITATIONS)
    seen_foreign: set[tuple[str, str]] = set()
    for doc in _docs():
        rel_doc = doc.relative_to(REPO).as_posix()
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8").splitlines(), start=1
        ):
            # A link label is prose about its target; the target itself is
            # checked as a link, so only free-standing citations land here.
            for raw in _BACKTICKED.findall(_MD_LINK.sub(r"\2", line)):
                token = _candidate(raw)
                if token is None:
                    continue
                if (rel_doc, token) in cited_foreign:
                    seen_foreign.add((rel_doc, token))
                    continue
                if token.endswith("/") or _PATH_LIKE.match(token):
                    if _resolves_as_path(doc, token):
                        continue
                    yield Finding(
                        "dangling-reference",
                        doc,
                        f"cites `{raw}`, which is not in the tree",
                        lineno,
                    )
                elif token not in words:
                    yield Finding(
                        "dangling-reference",
                        doc,
                        f"cites `{raw}`, which this tree names nowhere",
                        lineno,
                    )
    for key in sorted(cited_foreign - seen_foreign):
        yield Finding(
            "dangling-reference",
            REPO / "tools" / "lint_docs.py",
            f"FOREIGN_CITATIONS excuses `{key[1]}` in {key[0]}, which no "
            "longer cites it — drop the entry",
        )


CHECKS = (check_dangling_references, check_link_targets)


def main() -> int:
    findings = [f for check in CHECKS for f in check()]
    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print(f"\n{len(findings)} documentation finding(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
