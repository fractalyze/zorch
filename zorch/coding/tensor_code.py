# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The TensorCode seam: a LinearCode whose codeword coordinates are *point
evaluations* of the committed message's multilinear extension.

Ligerito recurses by reading each proximity right-hand side
`<G[s], w> = encode(w)[s]` as an eval-claim `ŵ(p_s)` of the *committed* `w`, then
batching the Q claims into one sumcheck. That reinterpretation needs the
generator row `G[s]` to factor as a tensor `eq(p_s, ·)` — i.e. the code must
expose the point `p_s` each codeword coordinate evaluates. This capability is
orthogonal to `FoldableCode.fold`: Ligerito never folds one codeword to the end
(it re-commits per level), so a plain `FoldableCode` does not supply it and a
`TensorCode` need not be foldable. Keeping it its own seam lets the ligero/ligerito
provers require exactly what they use — Ligero only `LinearCode.encode`, Ligerito
`TensorCode` — without dragging in the fold seam.

The seam contract (independent-oracle form, pinned in `tensor_code_test`):

    encode(w)[positions] == eval_mle(mle_coeffs_to_evals(w), eval_point(positions))

Reed-Solomon implements it — its Vandermonde generator row `(1, d_s, d_s², …)`
factors as a geometric tensor. The additive-NTT code (flock's LCH novel basis)
implements it for the binary-field instantiation (fractalyze/flock-zorch#11).

Like every code seam, an implementation MUST carry value-based `__eq__`/`__hash__`
(the `LinearCode` static-jit-zone-key contract, #214).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array

from zorch.coding.linear_code import LinearCode


@runtime_checkable
class TensorCode(LinearCode, Protocol):
    def eval_point(self, positions: Array) -> Array:
        """The multilinear point `p_s` that codeword coordinate `positions`
        evaluates. With `k = log2(message_len)`, returns shape
        `(*positions.shape, k)` such that

            encode(w)[s] == eval_mle(mle_coeffs_to_evals(w), eval_point(s)).

        The `s ↦ p_s` map is the generator row's tensor factorization — part of
        the code's identity, so it lives behind this seam.
        `mle_coeffs_to_evals` (the message-coefficient → hypercube-evaluation
        basis change) is the caller's, keeping the seam a pure point map."""
        ...
