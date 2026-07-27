# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Mechanical documentation checks, run as a pre-commit hook.

A rule written as prose drifts silently; a rule that runs fails the commit that
breaks it. Everything decidable from the tree lives here, and the tree is read
rather than described, so a new block is covered the moment its directory lands.
`.claude/skills/lint-zorch-docs` keeps only what needs judgment.
"""

from __future__ import annotations

import ast
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


def _module_has_docstring(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    return ast.get_docstring(tree) is not None


def _is_design_module(path: Path) -> bool:
    """A module shipping a block, as opposed to scaffolding or a benchmark."""
    name = path.name
    if name == "__init__.py" or name.endswith("_test.py") or name.startswith("bench_"):
        return False
    return _module_has_docstring(path)


# ---------------------------------------------------------------------------
# subsystem-doc-parity
# ---------------------------------------------------------------------------


def _subsystems() -> list[Path]:
    root = REPO / "zorch"
    return [
        d
        for d in sorted(root.iterdir())
        if d.is_dir()
        and d.name not in SUPPORT_DIRS
        and any(_is_design_module(p) for p in d.glob("*.py"))
    ]


def check_subsystem_doc_parity() -> Iterator[Finding]:
    """Every subsystem has a page the hub links.

    The name is derived, not listed (`logup_gkr` -> `logup-gkr.md`) and resolved
    anywhere under `docs/`, so the layout moves without touching this file.
    """
    hub = REPO / "docs" / "README.md"
    hub_text = hub.read_text(encoding="utf-8")
    for subsystem in _subsystems():
        stem = subsystem.name.replace("_", "-")
        pages = sorted(REPO.glob(f"docs/**/{stem}.md"))
        if not pages:
            yield Finding(
                "subsystem-doc-parity",
                subsystem,
                f"ships a design block but has no docs/**/{stem}.md",
            )
            continue
        for page in pages:
            link = page.relative_to(REPO / "docs").as_posix()
            if link not in hub_text:
                yield Finding(
                    "subsystem-doc-parity",
                    hub,
                    f"hub table has no row linking {link}",
                )


# ---------------------------------------------------------------------------
# module-doc-reachability
# ---------------------------------------------------------------------------

# Modules whose concept has no page yet. The reason states what a reader loses,
# making this a debt list rather than a mute suppression; an entry that stops
# applying fails, so the list cannot outlive what it excuses.
UNREACHED_MODULES = {
    "zorch/constraint_eval.py": (
        "constraint evaluation has no page; its concepts (constraint folding, "
        "the symbolic expression tree) are described only in the module docstring"
    ),
    "zorch/grind.py": (
        "proof-of-work grinding is named in stage-composition.md as a shared "
        "protocol operation, but has no page describing its security accounting"
    ),
}


def check_module_doc_reachability() -> Iterator[Finding]:
    """Every top-level module is cited by path from some page.

    A module no page names is one a reader cannot navigate to. Citing the path
    rather than a symbol is what makes a rename break the pointer loudly.
    """
    prose = "\n".join(p.read_text(encoding="utf-8") for p in _docs())
    live = {
        p.relative_to(REPO).as_posix()
        for p in sorted((REPO / "zorch").glob("*.py"))
        # A leading underscore marks a module the package does not export, so
        # no page owes the reader a route to it.
        if not p.name.startswith("_") and _is_design_module(p)
    }
    for rel in sorted(live):
        if rel in prose or rel in UNREACHED_MODULES:
            continue
        yield Finding(
            "module-doc-reachability",
            REPO / rel,
            "no page cites this path; add the citation or list it in "
            "UNREACHED_MODULES with what a reader loses",
        )
    for rel in sorted(UNREACHED_MODULES):
        if rel not in live:
            yield Finding(
                "module-doc-reachability",
                REPO / "tools" / "lint_docs.py",
                f"UNREACHED_MODULES lists {rel}, which no longer exists",
            )
        elif rel in prose:
            yield Finding(
                "module-doc-reachability",
                REPO / "tools" / "lint_docs.py",
                f"UNREACHED_MODULES lists {rel}, but a page now cites it — "
                "drop the entry",
            )


# ---------------------------------------------------------------------------
# canonical-claim
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalClaim:
    """A claim stated in exactly one place and linked from everywhere else."""

    name: str
    pattern: re.Pattern[str]
    home: str
    anchor: str


# Restating rather than linking is how copies drift into contradiction: the
# fusion unit claim read "one fused kernel" in CLAUDE.md and README.md while the
# measured page recorded two, and the prose lint pinned the wrong pair together.
CANONICAL_CLAIMS = (
    CanonicalClaim(
        name="field-dtype-gotchas",
        # A restated *limit* is the duplicate; naming the same op for another
        # reason (the fusion op scan bans `jnp.tile` as a gather) is not.
        pattern=re.compile(
            r"\b(field|extension)\b[^.]{0,60}\b(unsupported|unimplemented|"
            r"aborts|trips)\b"
            r"|\b(unsupported|unimplemented|aborts|trips)\b[^.]{0,60}"
            r"\b(field|extension)\b"
            r"|\bno iota over an extension\b",
            re.IGNORECASE,
        ),
        home="docs/blocks/poly.md",
        anchor="poly.md#field-dtype-gotchas",
    ),
    CanonicalClaim(
        name="fusion-unit",
        pattern=re.compile(r"(single|one) fused kernel", re.IGNORECASE),
        home="docs/README.md",
        anchor="README.md#fusion-north-star",
    ),
)


def check_canonical_claims() -> Iterator[Finding]:
    for claim in CANONICAL_CLAIMS:
        home = REPO / claim.home
        if not home.exists():
            yield Finding(
                "canonical-claim",
                REPO / "tools" / "lint_docs.py",
                f"{claim.name} names a home that does not exist: {claim.home}",
            )
            continue
        if not claim.pattern.search(home.read_text(encoding="utf-8")):
            yield Finding(
                "canonical-claim",
                home,
                f"is the home of {claim.name} but no longer states it",
            )
        for doc in _docs():
            if doc == home:
                continue
            text = doc.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if not claim.pattern.search(line):
                    continue
                if claim.anchor in text:
                    continue
                yield Finding(
                    "canonical-claim",
                    doc,
                    f"restates {claim.name}; link {claim.anchor} instead",
                    lineno,
                )


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


CHECKS = (
    check_subsystem_doc_parity,
    check_link_targets,
    check_module_doc_reachability,
    check_canonical_claims,
    check_dangling_references,
)


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
