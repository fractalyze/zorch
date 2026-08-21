# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The fused fold+message region is emitted, and only for the wire it describes.

`_fold_and_message` marks `zorch.sumcheck.round` with `variant="product"` and
`poly_form="coefficient"`. That pair names the COMPRESSED wire — `[c_0, c_2]`,
with `c_1` rebuilt by the verifier — so the marker must reach a recognizing
emitter only when the round really sends it. `StandardRound` sends the three
natural-domain evaluations instead; a marker over that round would be recognized
and emit a wire the decomposition never produces, silently, because the
recognizer rejects an unknown *variant* but not an unknown attr VALUE.

Both directions are asserted here: the compressed round emits with its routing
attrs, the standard round emits nothing and takes the plain decomposition.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.ligerito.prover import (
    _compressed_round,
    _fold_and_message,
    _round,
)
from zorch.sumcheck.prover import (
    SUMCHECK_ROUND_MARKER,
    CompressedProductRound,
    StandardRound,
)

_Round = StandardRound | CompressedProductRound

_NUM_VARS = 3


def _operands() -> tuple[frx.Array, frx.Array, frx.Array]:
    n = 1 << _NUM_VARS
    W = fnp.arange(n, dtype=fnp.uint32).astype(EF) + 1
    B = fnp.arange(n, dtype=fnp.uint32).astype(EF) + 5
    return W, B, fnp.asarray(3, EF)


def _trace(round_: _Round) -> str:
    W, B, r = _operands()
    return frx.make_jaxpr(lambda w, b, c: _fold_and_message(w, b, c, round_))(
        W, B, r
    ).pretty_print()


class FoldAndMessageMarkerTest(absltest.TestCase):
    def test_compressed_round_emits_marker_with_abi(self) -> None:
        text = _trace(_compressed_round(EF))
        self.assertIn(SUMCHECK_ROUND_MARKER, text)
        # phase/variant are the recognizer's routing key; poly_form and degree
        # are what it builds the message width from.
        self.assertIn("mid", text)
        self.assertIn("product", text)
        self.assertIn("coefficient", text)

    def test_standard_round_is_not_marked(self) -> None:
        # Not a style choice: `poly_form="coefficient"` over this round would be
        # RECOGNIZED and emit the compressed pair where the decomposition yields
        # three evaluations.
        self.assertNotIn(SUMCHECK_ROUND_MARKER, _trace(_round(EF)))

    def test_both_rounds_agree_with_their_round_poly(self) -> None:
        # The marked path and the plain path must each still be the fold's own
        # round poly -- the region is a fusion boundary, not a different sum.
        W, B, r = _operands()
        for round_ in (_compressed_round(EF), _round(EF)):
            msg, fw, fb = _fold_and_message(W, B, r, round_)
            folded = fnp.stack([fw, fb])
            self.assertEqual(
                fnp.asarray(msg).tobytes(),
                fnp.asarray(round_._round_poly(folded)).tobytes(),
                f"{type(round_).__name__} message differs from its round poly",
            )


if __name__ == "__main__":
    absltest.main()
