# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` rebuilds the queried codeword leaves from the committed roots and checks
the fold consistency of the batch open: the staggered RLC of the committed
matrices' opened rows must agree with the batched codeword's first pair-leaf, and
each fold layer's opened pair must fold to the next layer's, down to the constant
final poly. It holds only the public params (`code` for the block geometry and
fold, `tree` for the Merkle config) — never the prover's retained codeword.

The replay is driven by a `(BasefoldConfig, BasefoldChoreography)` pair — the
verify dual of `BasefoldProver`: the config fixes the fold schedule (commit
cadence), the choreography fixes the Fiat-Shamir framing (running-claim
recurrence via `reduce_claim`, message/root/terminal observes, query sampling,
grind checks). `BasefoldVerifier`'s defaults are zorch's native wire — so the
plain `BasefoldVerifier(code, tree, num_queries=…)` construction replays
byte-for-byte today's implementation and accepts/rejects identically. Prover and
verifier must share ONE choreography instance so their Fiat-Shamir streams stay
equal by construction. A byte-fixed consumer supplies its own config +
choreography and drives `verify_with_basis` (raw basis, `bind_statement`'s
`point=None`) — the dual of `BasefoldProver.open_with_basis`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array, lax

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import batch_staggered, sample_staggered_coeffs
from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import BasefoldCommitment, BasefoldConfig, BasefoldProof
from zorch.pcs.fold import from_base_field, verify_fold_chain, verify_openings
from zorch.transcript import Transcript
from zorch.utils.bits import log2_strict_usize

if TYPE_CHECKING:
    from zorch.pcs.protocol import PcsVerifier


@dataclass(frozen=True)
class BasefoldVerifier:
    """BaseFold PCS verifier (`PcsVerifier`). `code` fixes the block geometry +
    fold; `tree` the Merkle config; `choreography` the Fiat-Shamir wire (share
    the instance with the prover); `config` the fold schedule. The defaults are
    zorch's native wire — the plain `BasefoldVerifier(code, tree, num_queries=…)`
    construction replays byte-for-byte today's implementation. `config=None`
    derives the native per-verify config (`commits_per_round`, `num_queries`
    from the verifier)."""

    code: FoldableCode
    tree: MerkleTree
    # Must match the prover's; placeholder count, not soundness-calibrated.
    num_queries: int = 4
    choreography: BasefoldChoreography = BasefoldChoreography()
    config: BasefoldConfig | None = None

    def _resolved_config(self, num_vars: int) -> BasefoldConfig:
        """The config driving a verify over `num_vars` variables: the explicit
        one (checked against the point dimension) or the native default
        (`commits_per_round`, `num_queries` from the verifier). Mirrors
        `BasefoldProver._resolved_config` so both sides resolve identically."""
        if self.config is None:
            return BasefoldConfig(num_vars=num_vars, num_queries=self.num_queries)
        if self.config.num_vars != num_vars:
            raise ValueError(
                f"config.num_vars={self.config.num_vars} doesn't match the "
                f"verify's variable count {num_vars}"
            )
        return self.config

    def verify(
        self,
        commitment: BasefoldCommitment,
        points: Sequence[Array],
        values: Array,
        proof: BasefoldProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Verify a single-matrix open — the degenerate one-round batch, the
        `PcsVerifier` seam shape."""
        return self.verify_batch([commitment], points, [values], proof, transcript)

    def verify_batch(
        self,
        commitments: Sequence[BasefoldCommitment],
        points: Sequence[Array],
        values: Sequence[Array],
        proof: BasefoldProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        if len(points) != 1:
            raise ValueError(
                f"BaseFold opens the matrices at one shared point, got {len(points)}"
            )
        if not (len(commitments) == len(values) == len(proof.component_openings)):
            raise ValueError(
                f"batch mismatch: {len(commitments)} commitments, {len(values)} "
                f"value vectors, {len(proof.component_openings)} component openings"
            )
        z = points[0]
        num_vars = z.shape[0]
        # Eager shape guards, ahead of the jit zone (mirrors the prover): with
        # zero variables the fold replay below would index an empty layer list.
        if num_vars < 1:
            raise ValueError("BaseFold opens over at least one variable, got none")
        if self.code.message_len != (1 << num_vars):
            raise ValueError(
                f"point dimension {num_vars} doesn't match message_len "
                f"{self.code.message_len} (expected 2^{num_vars})"
            )
        self._check_proof_shape(proof, num_vars)
        # Fail loud on a non-native cadence / scheduled grind BEFORE any verdict,
        # symmetric to the prover's deferrals (P2 deferred, fail-loud).
        _require_native_cadence(self._resolved_config(num_vars))
        _require_no_grind(self.choreography, num_vars)
        return _verify_batch_body(
            self, list(commitments), z, list(values), proof, transcript
        )

    def verify_with_basis(
        self,
        commitment: BasefoldCommitment,
        basis: Array,
        value: Array,
        proof: BasefoldProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Verify a RAW-basis open — the dual of `BasefoldProver.open_with_basis`
        (`bind_statement` receives `point=None`). The native per-round check
        evaluates the sumcheck message at the opening point's coordinates, which
        a raw basis lacks, so the native verifier structure has no basis-path
        replay: symmetric to the prover's deferred basis-path message, the replay
        is a fail-loud consumer delta. The native binding still refuses
        `point=None` up front (a basis consumer overrides `bind_statement`)."""
        if basis.shape[0] < 2:
            raise ValueError("BaseFold opens over at least one variable, got none")
        num_vars = log2_strict_usize(basis.shape[0])
        if self.code.message_len != (1 << num_vars):
            raise ValueError(
                f"basis length {basis.shape[0]} doesn't match message_len "
                f"{self.code.message_len} (expected 2^num_vars)"
            )
        self._check_proof_shape(proof, num_vars)
        _require_native_cadence(self._resolved_config(num_vars))
        _require_no_grind(self.choreography, num_vars)
        # Bind the statement via the choreography with point=None (native refuses;
        # a basis consumer binds via the basis). Even a basis consumer then hits
        # the deferred basis-path replay below.
        self.choreography.bind_statement(transcript, commitment, None, value)
        raise NotImplementedError(
            "verify_with_basis's raw-basis replay is not wired here: the native "
            "per-round check evaluates the sumcheck message at the opening point, "
            "which a raw basis lacks; a basis consumer supplies its own per-round "
            "check from (message, basis), symmetric to open_with_basis"
        )

    def _check_proof_shape(self, proof: BasefoldProof, num_vars: int) -> None:
        """Fail loud on a structurally malformed proof — a short message/layer
        list would otherwise let the round loop silently skip checks."""
        if (
            len(proof.univariate_messages) != num_vars
            or len(proof.fri_roots) != num_vars
            or len(proof.query_openings) != num_vars
        ):
            raise ValueError(
                f"malformed proof: expected {num_vars} sumcheck messages / fold "
                f"layers, got {len(proof.univariate_messages)} / "
                f"{len(proof.fri_roots)} / {len(proof.query_openings)}"
            )


def _require_native_cadence(config: BasefoldConfig) -> None:
    """Fail loud on a non-native fold schedule: this driver replays only
    `commits_per_round` (pre-fold arity-2 pair commit every round), the verify
    dual of `BasefoldProver._require_native_cadence`. The row-batch prefix +
    multi-arity epoch cadence is deferred (design §"Core driver"), wired +
    byte-gated with its first consumer; only the config STRUCTURE exists here."""
    if not config.commits_per_round:
        raise NotImplementedError(
            "non-native fold cadence (row_batch_prefix / fold_arities) is not "
            "replayed yet; only commits_per_round (zorch-native) is wired. The "
            "deferred row-batch-prefix + multi-arity epoch cadence lands with "
            "its first byte-fixed consumer"
        )


def _require_no_grind(chor: BasefoldChoreography, num_vars: int) -> None:
    """Fail loud on a scheduled grind: `BasefoldProof` carries no pow-witness
    field (the native wire grinds nothing), so there is nothing for the verifier
    to `check_grind` against. The grind-check wire is a deferred consumer delta,
    symmetric to the prover's pow-witness NotImplementedError."""
    scheduled = (
        any(chor.fold_grind_bits(r, 0) is not None for r in range(num_vars))
        or chor.query_grind_bits(0) is not None
    )
    if scheduled:
        raise NotImplementedError(
            "the choreography schedules a grind, but BasefoldProof carries no "
            "pow-witness field for the verifier to check_grind against; the "
            "grind-check wire is a deferred consumer delta"
        )


# Jitted verify body: an eager replay interprets each composite op-by-op in
# Python (issue #140). The verifier is the static key (by value, #214), so its
# config and choreography (both frozen, value-compared) fix the compiled zone.
@partial(jax.jit, static_argnames=("verifier",))
def _verify_batch_body(
    verifier: BasefoldVerifier,
    commitments: list[Array],
    z: Array,
    values: list[Array],
    proof: BasefoldProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    chor = verifier.choreography
    dtype = z.dtype
    n = verifier.code.block_len
    num_vars = z.shape[0]
    config = verifier._resolved_config(num_vars)
    one = jnp.ones((), dtype)
    t = transcript

    # Re-derive the batch weights + initial claim (mirror open's FS order):
    # bind every commitment root, observe every matrix's claims, sample the
    # staggered coeffs, then bind the fold-round count. This multi-matrix
    # statement binding is the native consumer's staggered-RLC batching — kept
    # verifier-side (its prover dual is in `_open_batch_body`, not a choreography
    # hook, because "combine separate matrices" is where consumers diverge).
    for root in commitments:
        t = t.observe(root)
    for vals in values:
        t = t.observe(vals)
    total_width = sum(int(v.shape[0]) for v in values)
    t, coeffs = sample_staggered_coeffs(t, total_width, dtype)
    current_claim = batch_staggered(list(values), coeffs)
    t = t.observe(jnp.asarray(num_vars, dtype))

    # Replay the interleaved sumcheck + fold challenges through the choreography.
    # Every round observes its message (`round_message`), then its pre-fold
    # pair-leaf commitment root (`observe_root`), then samples β, then reduces
    # the running claim (`reduce_claim`, native `s(0)+β·s(1)`). All num_vars
    # rounds are homogeneous (no peeled final round) and ride one lax.scan — the
    # poseidon2 permute markers stop scaling with num_vars (#185). z_rev[r] binds
    # the variable folded in round r; the point-consistency check is native
    # (a non-native message reframes it — deferred). The native choreography's
    # message/root observes are pass-throughs, so the wire is byte-identical.
    z_rev = z[::-1]
    zero_vals = jnp.stack([m[0] for m in proof.univariate_messages])
    one_vals = jnp.stack([m[1] for m in proof.univariate_messages])
    fri_roots = jnp.stack(proof.fri_roots)

    def fold_round(
        carry: tuple[Transcript, Array, Array],
        xs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Transcript, Array, Array], Array]:
        t, claim, ok = carry
        zero_val, one_val, last, root = xs
        # Native point-consistency check: the running claim equals the sumcheck
        # message evaluated at the point coordinate `last`. Not a choreography
        # hook — the message form is native (a non-native message reframes this,
        # deferred); the ADDITIVE running-claim reduction is the hook.
        expected = (one - last) * zero_val + last * one_val
        ok = ok & (claim == expected)
        msg = chor.round_message(zero_val, one_val)
        t = chor.observe_message(t, msg)
        t = chor.observe_root(t, root)
        t, beta = t.sample()
        beta = beta.reshape(())
        return (t, chor.reduce_claim(claim, msg, beta), ok), beta

    (t, current_claim, ok), betas_stacked = lax.scan(
        fold_round,
        (t, current_claim, jnp.bool_(True)),
        (zero_vals, one_vals, z_rev, fri_roots),
    )
    # Index, don't iterate: list(field_array) dispatches lax.sign under CUDA.
    betas = [betas_stacked[r] for r in range(num_vars)]

    # IOPP terminal membership: the fully folded codeword is the base-code
    # encoding of the final claim (a constant on the order-blowup domain).
    ok = ok & verifier.code.check_final(proof.final_poly, current_claim)

    # Bind the cleartext final codeword before sampling queries (mirror open).
    t = chor.observe_final(t, proof.final_poly)

    # Query phase: shared positions; per-layer leaf index off the code seam.
    t, positions = chor.sample_queries(t, n, config.num_queries)
    a = verifier.code.layer_positions(positions, num_vars)

    # Merkle: every component matrix rebuilds its commitment at the query
    # positions, every fold layer's pair-leaf rebuilds its fri root at the
    # halved index. One batched pass per leaf-row width (#163).
    legs = [
        (commitments[r], positions, proof.component_openings[r])
        for r in range(len(commitments))
    ]
    for i in range(num_vars):
        legs.append((proof.fri_roots[i], a[i], proof.query_openings[i]))
    ok = ok & verify_openings(verifier.tree, legs)

    # Fold consistency. The staggered RLC of the opened component rows is the
    # batched codeword at `positions`; it must be the right leg of fold layer
    # 0's opened pair. Then each layer's pair folds to the next layer's
    # opened value, down to the final poly.
    # The committed leaves store base-field limbs; reinterpret each opened row
    # back to the code's value dtype before the EF RLC / fold comparison.
    comp_rows = [
        from_base_field(co.row, dtype, int(v.shape[0]))
        for co, v in zip(proof.component_openings, values, strict=True)
    ]
    comp_val = batch_staggered(comp_rows, coeffs)  # (Q,) batched value at `positions`
    lo0, _ = verifier.code.pair_indices(a[0], 0)
    leaf0 = from_base_field(proof.query_openings[0].row, dtype, 2)  # (Q, 2)
    in_leaf0 = jnp.where(positions == lo0, leaf0[:, 0], leaf0[:, 1])
    ok = ok & jnp.all(comp_val == in_leaf0)

    # Each layer's opened pair folds to the next layer's / the final poly.
    ok = ok & verify_fold_chain(
        verifier.code, proof.query_openings, betas, a, proof.final_poly
    )
    return ok, t


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[PcsVerifier[BasefoldCommitment, BasefoldProof]] = BasefoldVerifier
