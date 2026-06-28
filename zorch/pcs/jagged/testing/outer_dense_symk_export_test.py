# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Symbolic dense-length export of the outer Hadamard sumcheck byte-matches the
concrete halving loop.

``outer_sumcheck_scan`` is the fixed-width-mask form of ``outer_sumcheck`` — a
``lax.scan`` (the carry stays full width, dead tail masked) so the round count
``n = log2(dense)`` can be a symbolic ``jax.export`` dim. It trades the halving's
real shrink for full-width work each round (a ``lax.scan`` can't carry a shrinking
buffer); the live ``JaggedEvalRound`` keeps the real-halving ``outer_sumcheck``.
This locks the scan form: (1) byte-identical to the concrete halving loop, and
(2) one symbolic binary (dense length ``m`` and round count ``n`` independent
dims) byte-matches the concrete for two ``(m=2^n, n)`` pairs. Mont-u32.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import Array, export
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.jagged.prover import outer_sumcheck, outer_sumcheck_scan
from zorch.testkit.transcript import cheap_transcript

_PRIME = 2013265921


def _u32(x: Array) -> list[int]:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint32)).reshape(-1).tolist()


def _rand_ef(seed: int, shape: tuple[int, ...]) -> Array:
    ints = np.random.default_rng(seed).integers(
        1, 1 << 30, size=(*shape, 4), dtype=np.int64
    )
    return jax.lax.bitcast_convert_type(jnp.array(ints, dtype=BF), EF)


def _inputs(m: int) -> tuple[Array, Array, Array]:
    dense = jnp.asarray(
        np.random.default_rng(m).integers(0, _PRIME, (m,), np.uint32)
    ).view(BF)
    indicator = _rand_ef(m, (m,))
    claim = _rand_ef(m + 1, (1,))[0]
    return dense, indicator, claim


class OuterDenseSumcheckScanTest(absltest.TestCase):
    def test_scan_byte_matches_halving_loop(self) -> None:
        for m in (8, 16, 32):
            dense, indicator, claim = _inputs(m)
            n = (m - 1).bit_length()
            ref = outer_sumcheck(dense, indicator, claim, cheap_transcript(BF))
            got = outer_sumcheck_scan(dense, indicator, claim, cheap_transcript(BF), n)
            for i, (a, b) in enumerate(
                zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got))
            ):
                self.assertEqual(_u32(a), _u32(b), f"leaf {i} diverged at m={m}")

    def test_one_binary_byte_matches_concrete_for_every_dense(self) -> None:
        def fn(
            dense: Array,
            indicator: Array,
            claim: Array,
            rounds: Array,
            transcript: Any,
        ) -> Any:
            return outer_sumcheck_scan(
                dense, indicator, claim, transcript, rounds.shape[0]
            )

        # Dense length declared as 2*h so the (full -> (half, 2)) reshape is
        # provable; n (round count = log2 len) is its own dim (log is not poly).
        h, n = export.symbolic_shape(
            "h, n", constraints=["h >= 4", "h <= 16", "n >= 3", "n <= 5"]
        )
        exported = export.export(jax.jit(fn))(
            jax.ShapeDtypeStruct((2 * h,), BF),
            jax.ShapeDtypeStruct((2 * h,), EF),
            jax.ShapeDtypeStruct((), EF),
            jax.ShapeDtypeStruct((n,), EF),
            jax.tree_util.tree_map(
                lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), cheap_transcript(BF)
            ),
        )
        for m_c in (8, 32):
            n_c = (m_c - 1).bit_length()
            dense, indicator, claim = _inputs(m_c)
            rounds = jnp.zeros((n_c,), EF)
            ref = outer_sumcheck_scan(
                dense, indicator, claim, cheap_transcript(BF), n_c
            )
            got = exported.call(dense, indicator, claim, rounds, cheap_transcript(BF))
            for i, (a, b) in enumerate(
                zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got))
            ):
                self.assertEqual(_u32(a), _u32(b), f"leaf {i} diverged at m={m_c}")


if __name__ == "__main__":
    absltest.main()
