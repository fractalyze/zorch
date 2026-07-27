# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tests for the mechanical documentation checks.

Each check is exercised against a synthetic tree rather than the repository, so
a test pins the rule instead of the state of today's docs. The repository is
covered by one end-to-end case: the hook runs clean on the tree that ships.
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path
from typing import Any
from unittest import mock

from absl.testing import absltest

from tools import lint_docs


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


class LintDocsTestCase(absltest.TestCase):
    """A synthetic repository with the minimum every check reads."""

    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        _write(self.repo / "docs" / "README.md", "# hub\n")
        _write(self.repo / "README.md", "# project\n")
        _write(self.repo / "CLAUDE.md", "# context\n")
        # The registries name pages of the real repository, so a synthetic tree
        # starts with them empty; a test that exercises one patches it.
        patchers: tuple[Any, ...] = (
            mock.patch.object(lint_docs, "REPO", self.repo),
            mock.patch.object(lint_docs, "FOREIGN_CITATIONS", {}),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)


class DanglingReferenceTest(LintDocsTestCase):
    def test_symbol_named_nowhere_is_flagged(self) -> None:
        _write(self.repo / "docs" / "README.md", "the `JaggedLayout` type\n")
        findings = list(lint_docs.check_dangling_references())
        self.assertLen(findings, 1)
        self.assertIn("JaggedLayout", findings[0].message)

    def test_symbol_in_a_string_literal_resolves(self) -> None:
        _write(self.repo / "zorch" / "m.py", 'NAME = "zorch.jagged_bp"\n')
        _write(self.repo / "docs" / "README.md", "the `zorch.jagged_bp` marker\n")
        self.assertEmpty(list(lint_docs.check_dangling_references()))

    def test_path_resolves_from_the_page_that_cites_it(self) -> None:
        _write(self.repo / "docs" / "blocks" / "poly.md", "# poly\n")
        _write(self.repo / "docs" / "blocks" / "pcs.md", "see `poly.md`\n")
        self.assertEmpty(list(lint_docs.check_dangling_references()))

    def test_missing_path_is_flagged(self) -> None:
        _write(self.repo / "docs" / "README.md", "see `zorch/commit/jagged/`\n")
        findings = list(lint_docs.check_dangling_references())
        self.assertLen(findings, 1)
        self.assertIn("not in the tree", findings[0].message)

    def test_glob_and_placeholder_are_not_citations(self) -> None:
        _write(
            self.repo / "docs" / "README.md",
            "walk `zorch/*/` under `$DEVENV_ENVS_DIR/<workspace>/`\n",
        )
        self.assertEmpty(list(lint_docs.check_dangling_references()))

    def test_foreign_symbol_prefix_is_not_checked(self) -> None:
        _write(self.repo / "docs" / "README.md", "`jnp.arange` raises\n")
        self.assertEmpty(list(lint_docs.check_dangling_references()))

    def test_link_label_is_checked_as_a_target_not_a_symbol(self) -> None:
        _write(self.repo / "docs" / "blocks" / "poly.md", "# poly\n")
        _write(self.repo / "docs" / "README.md", "[`poly.md`](blocks/poly.md)\n")
        self.assertEmpty(list(lint_docs.check_dangling_references()))

    def test_stale_excuse_is_flagged(self) -> None:
        with mock.patch.object(
            lint_docs,
            "FOREIGN_CITATIONS",
            {("docs/README.md", "PolynomialSpace"): "a reason"},
        ):
            findings = list(lint_docs.check_dangling_references())
        self.assertLen(findings, 1)
        self.assertIn("no longer cites it", findings[0].message)


class LinkTargetTest(LintDocsTestCase):
    def test_broken_relative_link_is_flagged(self) -> None:
        _write(self.repo / "docs" / "blocks" / "pcs.md", "[c](../conventions.md)\n")
        findings = list(lint_docs.check_link_targets())
        self.assertLen(findings, 1)
        self.assertIn("does not resolve", findings[0].message)

    def test_anchor_and_absolute_links_are_left_alone(self) -> None:
        _write(
            self.repo / "docs" / "README.md",
            "[a](#section) and [b](https://example.invalid/x)\n",
        )
        self.assertEmpty(list(lint_docs.check_link_targets()))


class RepositoryTest(absltest.TestCase):
    def test_the_shipped_tree_passes(self) -> None:
        if not (lint_docs.REPO / "docs").is_dir():
            # Under a build sandbox the docs are not in the runfiles, so this
            # case runs from a source checkout; the pre-commit hook is what
            # enforces it everywhere else.
            self.skipTest("no docs/ beside the module — not a source checkout")
        self.assertEqual(lint_docs.main(), 0)


if __name__ == "__main__":
    absltest.main()
