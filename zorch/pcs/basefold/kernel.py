# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold per-round sumcheck kernel — the seam that fixes the round ALGEBRA,
decoupled from the Fiat-Shamir framing (`BasefoldChoreography`) and the fold
schedule (`BasefoldConfig`).

The interleaved sumcheck's per-round math diverges across consumers more than a
thin core + choreography can absorb: zorch-native is a degree-1 single-MLE check
(message `(s(0), s(1))`, MLE fold), whereas a product consumer runs a degree-2
check over two vectors (message `(Σaₑbₑ, Σ(aₑ+aₒ)(bₑ+bₒ))`, folding both). That is
neither FS framing nor fold schedule — it is the round algebra itself.

`SumcheckKernel` owns it over an OPAQUE, consumer-defined round state that the
shared core just threads:

- `initial_state(mle, basis, claim)` — build the round state from the open's
  inputs (native: the MLE, running claim, and unbound point suffix).
- `message(state)` — the raw round-message components (native: `(s(0), s(1))`);
  the choreography frames+emits them onto the wire and the proof stores them.
- `fold(state, message, r)` — fold the state by the shared challenge `r` (the
  SAME `r` the core folds the codeword by).
- `final(state)` — the prover's terminal sumcheck value(s), bound alongside the
  codeword (native: none — the folded codeword is the terminal).
- `reduce_claim(claim, message, r)` / `round_check(claim, message, coord)` — the
  verifier-side per-round recurrence and consistency check.

The default is zorch's native single-MLE algebra, byte-identical to the wire
`BasefoldProver`/`BasefoldVerifier` drove before the seam existed; a byte-fixed
consumer subclasses and overrides only its deltas. Prover and verifier share ONE
instance (every method is a pure function of its arguments).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as jnp
from frx import Array

from zorch.poly.multilinear import eval_mle, mle_fold


@dataclass(frozen=True)
class SumcheckKernel:
    """zorch's native single-MLE degree-1 sumcheck algebra as an overridable
    kernel. Stateless — a consumer subclasses and overrides only its deltas.
    The native round state is `(mle, claim, zs)`: the running MLE the sumcheck
    folds, the running claim, and the unbound opening-point suffix (`zs[-1]` is
    the variable bound this round)."""

    def initial_state(self, mle: Array, basis: Array, claim: Array) -> tuple:
        """Build the round state from the open's inputs. Native drives from an
        opening POINT, so `basis` is the point `zs` and the state is
        `(mle, claim, zs)`."""
        return (mle, claim, basis)

    def message(self, state: tuple) -> tuple[Array, Array]:
        """The degree-1 sumcheck message `(s(0), s(1))` for the variable bound
        this round (`zs[-1]`), from the running MLE and claim. `mle_fold(., 0)`
        fixes the bound variable to 0 (the additive fold coincides with the
        multilinear partial-eval at beta=0), so `zero_val` is `s(0)`; `one_val`
        is recovered from the running claim."""
        mle, claim, zs = state
        zero_mle = mle_fold(mle, jnp.zeros((), zs.dtype))
        rest = zs[:-1]
        zero_val = eval_mle(zero_mle, rest) if rest.shape[0] > 0 else zero_mle[0]
        one_val = (claim - zero_val) / zs[-1] + zero_val
        return zero_val, one_val

    def fold(self, state: tuple, message: tuple[Array, Array], r: Array) -> tuple:
        """Fold the state by the shared challenge `r`: fold the MLE, advance the
        running claim (`s(0) + r·s(1)`), and shrink the point suffix."""
        mle, _claim, zs = state
        zero_val, one_val = message
        return (mle_fold(mle, r), zero_val + r * one_val, zs[:-1])

    def final(self, state: tuple) -> Any:
        """The prover's terminal sumcheck value(s), bound alongside the codeword.
        Native binds only the folded codeword, so there is nothing extra here."""
        del state
        return None

    def reduce_claim(
        self, claim: Array, message: tuple[Array, Array], r: Array
    ) -> Array:
        """Verifier per-round running-claim recurrence: `s(0) + r·s(1)` — the
        additive BaseFold/FRI combine (`mle_fold`'s `e0 + r·e1`, NOT the affine
        partial-eval bind), by construction the same `r` that folds the codeword.
        `claim` rides unused in the native reduction — a consumer whose
        recurrence also depends on the prior claim overrides."""
        del claim
        zero_val, one_val = message
        return zero_val + r * one_val

    def round_check(
        self, claim: Array, message: tuple[Array, Array], coord: Array
    ) -> Array:
        """Verifier per-round point-consistency: the running claim equals the
        sumcheck message evaluated at the bound point coordinate `coord`
        (`(1-coord)·s(0) + coord·s(1)`). Returns a boolean array. A consumer
        without an opening point (a raw-basis product check) overrides."""
        zero_val, one_val = message
        one = jnp.ones((), coord.dtype)
        return claim == (one - coord) * zero_val + coord * one_val

    def verify_final(self, claim: Array, final_state: Any) -> tuple[Array, Array]:
        """Verifier-side terminal for the raw-basis / cadence replay: check the
        prover's terminal sumcheck value(s) (`final`) against the reduced running
        claim, and return `(ok, codeword_value)` — `codeword_value` being the
        constant the fully folded codeword must equal, the tie between the
        sumcheck's final value and the FRI terminal.

        The point path enforces per-round consistency through `round_check` and
        ties the terminal with `code.check_final(final_poly, claim)`, so the
        native default carries no extra terminal state: it passes and ties the
        codeword to the running claim. A basis consumer whose terminal is a
        product of folded vectors (`final = (a[0], b[0])`, claim `Σ a·b`)
        overrides to check `a[0]·b[0] == claim` and return `a[0]`."""
        del final_state
        return jnp.bool_(True), claim
