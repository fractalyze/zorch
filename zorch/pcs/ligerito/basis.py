# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The basis a Ligerito level commits its matrix in — the `(pre, expand)` pair
tied by one invariant, bundled so the two halves cannot drift.

A row-leaf commit encodes `pre(matrix)` and reads a codeword coordinate back as a
point-eval of the committed row:

    encode(pre(matrix))[.., s] == <matrix_row, expand(eval_point(s), one)>

`pre` (commit side) and `expand` (open/verify side) are two halves of that single
identity — pick `pre` and `expand` is forced, and vice versa. They live in
separate call sites (commit vs. the sumcheck induce / terminal check), so a
`CommitBasis` value ties them together: a mismatched pairing is unrepresentable,
and the prover, the verifier's induce, and the verifier's terminal check all read
the same `expand`.

Two conventions exist today; use the module singletons (`EVAL_BASIS` /
`MONOMIAL_BASIS`) rather than fresh instances — they are identity-stable, so they
are clean `frx.jit` static keys and commit-cache keys (a fresh instance per call
would defeat the trace cache, #214).

The concept is matrix-commit-level, not Ligerito-specific; it lives here because
Ligerito's induce is its only consumer today (single-shot `pcs/ligero`
re-encodes `w` directly and needs no basis vector). Promote to `pcs/matrix_commit`
when a second scheme consumes the `expand` half.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import frx
from frx import Array

from zorch.poly.eq import expand_eq_to_hypercube, expand_monomial_to_hypercube
from zorch.poly.multilinear import mle_evals_to_coeffs


@dataclass(frozen=True)
class CommitBasis:
    """A commit basis: the pre-transform a level's matrix is encoded through and
    the hypercube expansion its opened coordinates evaluate against, tied by the
    module invariant. `pre` is read at commit time, `expand` at open/verify time;
    keep them in one value so the pairing is chosen once. Frozen with
    function-valued fields, so it is hashable by identity — pass a singleton, not
    a fresh instance, to keep it a stable jit static key."""

    pre: Callable[[Array], Array]
    expand: Callable[[Array, Array], Array]

    def proximity_basis(self, points_s: Array, weights: Array) -> Array:
        """`(Q, num_vars) -> (Q, 2^num_vars)`: the basis vector each opened
        coordinate evaluates the folded witness against, scaled by that row's
        weight —

            <row, proximity_basis(points, w)[s]> == w_s · codeword[.., s]

        which is the module invariant exactly when `w` is one. One definition
        for the prover's induce, the verifier's induce, and the verifier's
        terminal residual check — they would desynchronize the glued sumcheck if
        they drifted.

        `weights` is `(Q,)` to scale each row, or 0-d to share one value. It
        SEEDS the expansion rather than scaling its output, which is why it is a
        parameter rather than something the caller applies afterwards: both
        `expand` conventions thread the weight into the tensor product they
        build, where it rides multiplies that already exist, whereas scaling the
        result costs a GF multiply per output element — `Q·2^num_vars` of them.
        A caller batching rows (`Σ_s w_s·basis(p_s)`) should therefore pass its
        coefficients here and reduce, rather than reduce and scale. GF
        multiplication is associative and exact, so the two spellings are
        byte-equal.
        """
        return frx.vmap(self.expand, in_axes=(0, None if weights.ndim == 0 else 0))(
            points_s, weights
        )


def _bit_reverse_matrix(matrix: Array) -> Array:
    # Two single-dimension passes: the CPU lowering rejects a multi-dim reverse.
    reversed_rows = frx.lax.bit_reverse(matrix, dimensions=(0,))
    return frx.lax.bit_reverse(reversed_rows, dimensions=(1,))


def _eval_expand(point: Array, weight: Array) -> Array:
    return expand_eq_to_hypercube(point, weight)


def _monomial_expand(point: Array, weight: Array) -> Array:
    return expand_monomial_to_hypercube(point[::-1], weight)


# zorch native: commit encodes `mle_evals_to_coeffs`, so a codeword coordinate is
# a clean `eval_mle` of the eval-basis row (`<row, expand_eq(eval_point(s))>`).
EVAL_BASIS = CommitBasis(pre=mle_evals_to_coeffs, expand=_eval_expand)

# Coefficient basis: commit the bit-reversed raw matrix, so the coordinate reads
# `<row, expand_monomial(reversed(eval_point(s)))>` — the raw-lane convention of
# wire formats that commit the witness directly (flock's `ligero_commit`). The
# identity holds for both multiplicative and additive-NTT Reed-Solomon (both
# codes' basis polynomials are multiplicative over index bits).
MONOMIAL_BASIS = CommitBasis(pre=_bit_reverse_matrix, expand=_monomial_expand)


def select_commit_basis(monomial_commit: bool) -> CommitBasis:
    """Map the `LigeritoConfig.monomial_commit` selector to its basis singleton —
    the single point that turns the config's declarative bool into the
    `(pre, expand)` pairing, so both sides derive the same one."""
    return MONOMIAL_BASIS if monomial_commit else EVAL_BASIS
