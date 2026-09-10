# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The basis a Ligerito level commits its matrix in — the
`(witness_pre, pre, expand)` triple tied by one invariant, bundled so the halves
cannot drift.

A row-leaf commit encodes `pre(matrix)` and reads a codeword coordinate back as a
point-eval of the committed row:

    encode(pre(matrix))[.., s] == <matrix_row, expand(eval_point(s), one)>

`pre` (commit side) and `expand` (open/verify side) are two halves of that single
identity — pick `pre` and `expand` is forced, and vice versa. They live in
separate call sites (commit vs. the sumcheck induce / terminal check), so a
`CommitBasis` value ties them together: a mismatched pairing is unrepresentable,
and the prover, the verifier's induce, and the verifier's terminal check all read
the same `expand`.

The basis also fixes the *witness order* the initial commit is handed, which is
why `witness_pre` rides the same value. A convention whose `pre` permutes the
matrix wants the flat witness in the order it permutes *from*, not in the order
the open folds: handed the folded order, the caller has already paid the inverse
permutation and `pre` pays it again. So `commit` takes the witness in the
basis's own commit order and derives both halves from it — `witness_pre` the
flat multilinear the open recursion folds, `initial_matrix` the level-0 encode
input, which is by definition

    initial_matrix(w, log_interleave) == pre(witness_pre(w).reshape(kappa, rho))

with `kappa = 2^log_interleave` and `rho = len(w) / kappa`. `initial_matrix`
computes that composite itself, so a convention with nothing to gain says
nothing and is correct by construction. One that reaches the same matrix by a
cheaper route overrides it with `initial_pre` (the monomial basis does; see
`_monomial_initial_pre`) — never a *different* matrix: that equality is what
keeps the commit order un-forgeable, and `testing/basis_test.py` checks it for
every singleton. The seam is initial-commit-only — a level's re-commit takes the
folded witness straight off the sumcheck, already in the open's order, and goes
through `pre` as before.

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
    """A commit basis: the witness order the initial commit is handed, the
    pre-transform a level's matrix is encoded through, and the hypercube
    expansion its opened coordinates evaluate against, tied by the module
    invariants. `witness_pre` is read at the initial commit, `pre` at every
    level's commit, `expand` at open/verify time; keep them in one value so the
    pairing is chosen once. Frozen with function-valued fields, so it is hashable
    by identity — pass a singleton, not a fresh instance, to keep it a stable jit
    static key.

    `witness_pre` maps the committed witness to the flat multilinear the open
    recursion folds. The level-0 encode input is `initial_matrix`, which composes
    the two by default; `initial_pre` overrides it and exists only for a
    convention that can spell that composite cheaper.
    """

    witness_pre: Callable[[Array], Array]
    pre: Callable[[Array], Array]
    expand: Callable[[Array, Array], Array]
    initial_pre: Callable[[Array, int], Array] | None = None

    def initial_matrix(self, witness: Array, log_interleave: int) -> Array:
        """The level-0 encode input, from a `witness` in this basis's commit
        order: `pre` of the `witness_pre` output reshaped into interleave lanes.

        `initial_pre` overrides this with a cheaper spelling of the same matrix;
        omitting it is what makes a convention correct by construction."""
        if self.initial_pre is not None:
            return self.initial_pre(witness, log_interleave)
        kappa = 1 << log_interleave
        lanes = self.witness_pre(witness).reshape(kappa, witness.shape[0] // kappa)
        return self.pre(lanes)

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


def _identity(witness: Array) -> Array:
    return witness


def _bit_reverse(witness: Array) -> Array:
    return frx.lax.bit_reverse(witness, dimensions=(0,))


def _bit_reverse_matrix(matrix: Array) -> Array:
    # Two single-dimension passes: the CPU lowering rejects a multi-dim reverse.
    reversed_rows = frx.lax.bit_reverse(matrix, dimensions=(0,))
    return frx.lax.bit_reverse(reversed_rows, dimensions=(1,))


def _monomial_initial_pre(witness: Array, log_interleave: int) -> Array:
    """The module invariant's composite, `_bit_reverse_matrix` of the lanes of
    `_bit_reverse(w)`, as the plain transpose it collapses to.

    Three bit-reversals cancel. Splitting the flat `n`-bit index as
    `w[r * rho + c]` with `r` over `log2(kappa)` bits and `c` over `log2(rho)`,
    the flat reversal is `rev_n(r * rho + c) = rev_rho(c) * kappa + rev_kappa(r)`
    — it swaps the two halves and reverses each. The matrix reversal then
    reverses each axis again, undoing both halves and leaving only the swap,
    which is the transpose:

        bit_reverse_matrix(bit_reverse(w).reshape(kappa, rho))
            == w.reshape(rho, kappa).T
    """
    kappa = 1 << log_interleave
    return witness.reshape(witness.shape[0] // kappa, kappa).T


def _eval_expand(point: Array, weight: Array) -> Array:
    return expand_eq_to_hypercube(point, weight)


def _monomial_expand(point: Array, weight: Array) -> Array:
    return expand_monomial_to_hypercube(point[::-1], weight)


# zorch native: commit encodes `mle_evals_to_coeffs`, so a codeword coordinate is
# a clean `eval_mle` of the eval-basis row (`<row, expand_eq(eval_point(s))>`).
# The open folds what was committed, so the commit order is the open's.
EVAL_BASIS = CommitBasis(
    witness_pre=_identity,
    pre=mle_evals_to_coeffs,
    expand=_eval_expand,
)

# Coefficient basis: commit the bit-reversed raw matrix, so the coordinate reads
# `<row, expand_monomial(reversed(eval_point(s)))>` — the raw-lane convention of
# wire formats that commit the witness directly (flock's `ligero_commit`). The
# identity holds for both multiplicative and additive-NTT Reed-Solomon (both
# codes' basis polynomials are multiplicative over index bits). The commit order
# is the raw coefficient order the convention reverses from, and the open folds
# `_bit_reverse` of it — the read side already reverses (`_monomial_expand`
# reverses the point), so taking coefficient order on the write side is what
# makes the convention self-consistent.
MONOMIAL_BASIS = CommitBasis(
    witness_pre=_bit_reverse,
    pre=_bit_reverse_matrix,
    expand=_monomial_expand,
    initial_pre=_monomial_initial_pre,
)


def select_commit_basis(monomial_commit: bool) -> CommitBasis:
    """Map the `LigeritoConfig.monomial_commit` selector to its basis singleton —
    the single point that turns the config's declarative bool into the
    `(witness_pre, pre, expand)` pairing, so both sides derive the same one."""
    return MONOMIAL_BASIS if monomial_commit else EVAL_BASIS
