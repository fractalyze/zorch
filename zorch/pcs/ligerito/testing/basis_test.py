# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The `CommitBasis` module invariant, for every singleton:

    initial_matrix(w, log_interleave) == pre(witness_pre(w).reshape(kappa, rho))

`commit` takes its witness in the basis's own order and builds the level-0
encode input with `initial_matrix` alone, so this equality is the only thing
tying that matrix to the `pre` the paired `expand` reads back. The right-hand
side is spelled out here independently of the method, which pins both halves of
the seam: that `initial_matrix`'s default really is the declared composite
(`EVAL_BASIS`, which supplies no override), and that an override reaches the
same matrix by its cheaper route (`MONOMIAL_BASIS`, whose plain transpose is
three bit-reversals that cancel). Run over the interleave extremes as well as
the ordinary splits, since a transposed reshape is easiest to get backwards
there.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from absl.testing import absltest, parameterized
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.ligerito.basis import EVAL_BASIS, MONOMIAL_BASIS, CommitBasis
from zorch.testkit.random_field import rand_ext_field

# `(num_vars, log_interleave)` splits: ordinary ones, plus both extremes — a
# single lane (`kappa = 1`) and a two-column message (`rho = 2`).
_SPLITS = ((1, 0), (5, 1), (6, 2), (8, 2), (9, 8), (10, 3), (12, 4))


def _cases() -> Iterator[dict[str, Any]]:
    for name, basis in (("eval", EVAL_BASIS), ("monomial", MONOMIAL_BASIS)):
        for num_vars, log_interleave in _SPLITS:
            yield dict(
                testcase_name=f"{name}_n{num_vars}_k{log_interleave}",
                basis=basis,
                num_vars=num_vars,
                log_interleave=log_interleave,
            )


class CommitBasisInvariantTest(parameterized.TestCase):
    @parameterized.named_parameters(_cases())
    def test_initial_matrix_is_the_pre_composite(
        self, basis: CommitBasis, num_vars: int, log_interleave: int
    ) -> None:
        witness = rand_ext_field(num_vars, (1 << num_vars,), F, EF)
        kappa = 1 << log_interleave
        rho = (1 << num_vars) >> log_interleave
        want = basis.pre(basis.witness_pre(witness).reshape(kappa, rho))
        got = basis.initial_matrix(witness, log_interleave)
        self.assertEqual(got.shape, (kappa, rho))
        self.assertEqual(got.tolist(), want.tolist())


if __name__ == "__main__":
    absltest.main()
