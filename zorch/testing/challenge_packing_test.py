# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Reading a squeeze as a field element happens in exactly one module.

A challenge has two halves: how many transcript words to squeeze, and what field
to read them as. `ChallengePolicy` derives both from one dtype. A call site that
squeezes raw words and then reinterprets them picks the two independently, so
nothing stops a policy sampling a narrower field than the claim it folds -- the
challenge is then spread across a field it does not span, and the prover and its
verifier can disagree about the packing without either being obviously wrong.

Enforced as an import rule rather than per-call-site checks: a contract that has
to be restated at each new site is one a new site can omit.
"""

from __future__ import annotations

import ast
import pathlib

from absl.testing import absltest

import zorch

_PACKING = "reinterpret_challenge"
# `sample_challenge` is transcript.py's own typed one-squeeze helper and packs
# through the same routine, so it is an entry point rather than a bypass.
_OWNERS = {"challenge.py", "transcript.py"}


class ChallengePackingTest(absltest.TestCase):
    def test_only_the_owning_modules_import_the_packing(self) -> None:
        root = pathlib.Path(zorch.__file__).parent
        offenders, scanned = [], 0
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root)
            if rel.name in _OWNERS or "testing" in rel.parts:
                continue
            scanned += 1
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                if _PACKING in names:
                    offenders.append(rel.as_posix())

        self.assertGreater(scanned, 50, "the sweep found almost nothing to scan")
        self.assertEqual(
            offenders,
            [],
            f"{_PACKING} must reach call sites through ChallengePolicy, which "
            f"derives the limb count and the field together; imported by: "
            f"{sorted(set(offenders))}",
        )


if __name__ == "__main__":
    absltest.main()
