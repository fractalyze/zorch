# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""BaseFold verifier — the `PcsVerifier` half of the multilinear PCS.

`verify` rebuilds the queried codeword leaves from the committed roots and checks
the fold consistency of the batch open: the staggered RLC of the committed
matrices' opened rows must agree with the batched codeword's first pair-leaf, and
each fold layer's opened pair must fold to the next layer's, down to the constant
final poly. It holds only the public params (`code` for the block geometry and
fold, `tree` for the Merkle config) — never the prover's retained codeword.

The replay is driven by a `(BasefoldConfig, BasefoldChoreography, SumcheckKernel)`
triple — the verify dual of `BasefoldProver`: the config fixes the fold schedule
(commit cadence), the choreography fixes the Fiat-Shamir framing (message/root/
terminal observes, query sampling, grind checks), and the kernel owns the round
algebra (the per-round `round_check` consistency + `reduce_claim` recurrence).
`BasefoldVerifier`'s defaults are zorch's native wire — so the plain
`BasefoldVerifier(code, tree, num_queries=…)` construction replays byte-for-byte
today's implementation and accepts/rejects identically. Prover and verifier must
share ONE choreography + kernel so their Fiat-Shamir streams stay equal by
construction. A byte-fixed consumer supplies its own config +
choreography and drives `verify_with_basis` (raw basis, `bind_statement`'s
`point=None`) — the dual of `BasefoldProver.open_with_basis`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import frx
import frx.numpy as jnp
from frx import Array, lax

from zorch.coding.foldable_code import FoldableCode
from zorch.commit.merkle import MerkleTree
from zorch.pcs.basefold.batching import batch_staggered, sample_staggered_coeffs
from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import (
    BasefoldCommitment,
    BasefoldConfig,
    BasefoldProof,
    CadenceProof,
)
from zorch.pcs.basefold.kernel import SumcheckKernel
from zorch.pcs.fold import (
    from_base_field,
    lane_combine,
    verify_fold_chain,
    verify_openings,
)
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
    kernel: SumcheckKernel = SumcheckKernel()
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
        config = self._resolved_config(num_vars)
        config.require_native("verify")
        _require_no_grind(self.choreography, config)
        return _verify_batch_body(
            self, list(commitments), z, list(values), proof, transcript
        )

    def verify_with_basis(
        self,
        commitment: BasefoldCommitment,
        basis: Array,
        value: Array,
        proof: BasefoldProof | CadenceProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        """Verify a RAW-basis open — the dual of `BasefoldProver.open_with_basis`
        (`bind_statement` receives `point=None`), dispatching on the fold schedule
        exactly as the prover entry does.

        Under a non-native schedule (`row_batch_prefix` / `fold_arities`) this
        replays the generic cadence (row-batch prefix + multi-arity FRI epochs)
        against a `CadenceProof`, the symmetric dual of `_open_with_basis_cadence`:
        `commitment` is the prover's initial codeword root (bound, not observed —
        the outer protocol committed it), `value` the claimed target the kernel's
        `reduce_claim`/`verify_final` fold against.

        Under the native uniform schedule the per-round check evaluates the
        sumcheck message at the opening point's coordinates, which a raw basis
        lacks, so that path has no basis replay yet — a fail-loud consumer delta
        (the native binding also refuses `point=None`)."""
        if basis.shape[0] < 2:
            raise ValueError("BaseFold opens over at least one variable, got none")
        num_vars = log2_strict_usize(basis.shape[0])
        config = self._resolved_config(num_vars)
        if not config.commits_per_round:
            if config.row_batch_prefix == 0:
                raise NotImplementedError(
                    "non-native cadence verify is wired for a row-batch prefix "
                    "(row_batch_prefix > 0, the row-batch-prefix shape); the "
                    "prefix-free multi-arity sub-case commits no post-prefix "
                    "layer and needs its own bridge — not replayed here"
                )
            # The schedule fixes the proof type (dispatch mirrors the prover):
            # a non-native config carries a `CadenceProof`.
            return _verify_with_basis_cadence(
                self,
                commitment,
                basis,
                value,
                config,
                cast(CadenceProof, proof),
                transcript,
            )
        # Native single-MLE basis path (deferred, as before): a raw basis lacks
        # the opening point the native per-round check needs.
        if self.code.message_len != (1 << num_vars):
            raise ValueError(
                f"basis length {basis.shape[0]} doesn't match message_len "
                f"{self.code.message_len} (expected 2^num_vars)"
            )
        self._check_proof_shape(cast(BasefoldProof, proof), num_vars)
        _require_no_grind(self.choreography, config)
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


def _require_no_grind(chor: BasefoldChoreography, config: BasefoldConfig) -> None:
    """Fail loud on a scheduled grind: `BasefoldProof` / `CadenceProof` now carry a
    `pow_witnesses` wire slot, but the prover does not populate it and the verifier
    does not `check_grind` against it yet — the grind production + check are a
    deferred delta. The schedule count comes off the choreography's bits methods
    (`num_pow_witnesses`), the one source of truth shared with the prover's guards."""
    if chor.num_pow_witnesses(config) > 0:
        raise NotImplementedError(
            "the choreography schedules a grind, but the verifier does not "
            "check_grind against proof.pow_witnesses yet (the field is the wire "
            "slot; the grind check is a deferred delta)"
        )


@partial(frx.jit, static_argnames=("code", "base_level"))
def _fold_coset(
    code: FoldableCode,
    coset: Array,
    betas: list[Array],
    base_level: int,
    leaf_index: Array,
) -> Array:
    """Fold a `[Q, 2^len(betas)]` opened coset down to `[Q]` by `len(betas)`
    successive binary code folds, from code fold level `base_level` — the
    verifier's per-epoch refold, the dual of the prover's `code.fold` chain within
    one epoch. A jit island (`code`/`base_level` static, `coset`/`betas`/
    `leaf_index` traced): the cadence verify keeps its Fiat-Shamir orchestration
    eager, but the refold is pure compute that must lower to one fused kernel. The
    coset is contiguous (adjacent entries are the code's conjugate
    pair, the row-batch/multi-arity layout the epoch commits group), so each level
    reshapes to `[Q, half, 2]` and folds the pair; `leaf_index` [Q] is the coset's
    index in each layer (unchanged as the width halves), and the pair's landing
    index in the next layer is `leaf_index*half + j` — the per-epoch coset refold
    assembled from the `FoldableCode` seam."""
    buf = coset
    for k, beta in enumerate(betas):
        q, width = buf.shape
        half = width // 2
        pairs = buf.reshape(q, half, 2)
        lo, hi = pairs[:, :, 0], pairs[:, :, 1]
        pos = (
            leaf_index[:, None] * half + jnp.arange(half, dtype=leaf_index.dtype)
        ).reshape(-1)
        folded = code.fold_values(
            lo.reshape(-1), hi.reshape(-1), beta, pos, base_level + k
        )
        buf = folded.reshape(q, half)
    return buf[:, 0]


def _verify_with_basis_cadence(
    verifier: BasefoldVerifier,
    commitment: BasefoldCommitment,
    basis: Array,
    value: Array,
    config: BasefoldConfig,
    proof: CadenceProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    """Replay a non-native cadence open (row-batch prefix + multi-arity FRI
    epochs), the structural dual of `prover._open_with_basis_cadence`. Eager, like
    the prover's driver — a host-sequential byte-wire replay cannot ride one jit
    zone. Reuses `pcs.fold`: `lane_combine` for the row-batch, `verify_openings`
    for the Merkle legs, `code.fold_values` (via `_fold_coset`) for the per-epoch
    refold; the sumcheck rides the kernel (`reduce_claim` + `verify_final`)."""
    del basis  # the target rides `value`; a basis consumer that ties the terminal
    # to the basis overrides `verify_final` to consume it.
    chor = verifier.choreography
    kernel = verifier.kernel
    code = verifier.code
    tree = verifier.tree
    num_vars = config.num_vars
    prefix = config.row_batch_prefix
    arities = config.fold_arities
    num_epochs = len(arities)
    n_pos = code.block_len

    # Fail loud on a scheduled grind BEFORE any verdict: `CadenceProof` carries no
    # pow-witness field, symmetric to the prover's grind guard on
    # `_open_with_basis_cadence`.
    _require_no_grind(chor, config)

    # Shape guard on the CadenceProof, symmetric to `_check_proof_shape` — a short
    # message / root / layer list would let the replay skip checks silently.
    expected_roots = num_epochs  # 1 post-prefix commit + (num_epochs - 1) epochs
    expected_layers = 1 + expected_roots
    if (
        len(proof.round_messages) != num_vars
        or len(proof.commit_roots) != expected_roots
        or len(proof.layer_openings) != expected_layers
    ):
        raise ValueError(
            f"malformed cadence proof: expected {num_vars} round messages, "
            f"{expected_roots} commit roots, {expected_layers} layer openings; "
            f"got {len(proof.round_messages)} / {len(proof.commit_roots)} / "
            f"{len(proof.layer_openings)}"
        )

    # Statement bind (mirror the prover: initial root, no point, value). The outer
    # protocol committed the root, so the cadence does not observe it again here.
    t = chor.bind_statement(transcript, commitment, None, value)

    # Replay the interleaved sumcheck + observe the commit roots in lockstep with
    # the prover: absorb each round message, sample the shared challenge, reduce
    # the running claim, and observe a commit root at the prefix end / each epoch
    # boundary (all but the last). The kernel owns the claim recurrence.
    claim = value
    betas: list[Array] = []
    root_idx = 0
    epoch = in_epoch = 0
    for rnd in range(num_vars):
        components = proof.round_messages[rnd]
        msg = chor.round_message(*components)
        t = chor.observe_message(t, msg)
        t, r = chor.fold_challenge(t, None, rnd, 0)
        claim = kernel.reduce_claim(claim, components, r)
        betas.append(r)
        if rnd < prefix:
            if rnd + 1 == prefix and arities:
                t = chor.observe_root(t, proof.commit_roots[root_idx])
                root_idx += 1
        else:
            in_epoch += 1
            if in_epoch == arities[epoch]:
                if epoch + 1 < num_epochs:
                    t = chor.observe_root(t, proof.commit_roots[root_idx])
                    root_idx += 1
                in_epoch = 0
                epoch += 1

    # Terminal: the kernel checks the prover's final sumcheck value(s) against the
    # reduced claim and yields the constant the folded codeword must equal (the
    # sumcheck<->FRI tie). The final codeword is constant, so `all == cw_const`
    # covers both constancy and the tie.
    ok, cw_const = kernel.verify_final(claim, proof.final_state)
    ok = ok & jnp.all(proof.final_codeword == cw_const)

    # Bind the terminal, then sample the shared query positions (mirror prover).
    t = chor.observe_final(t, proof.final_codeword)
    t, positions = chor.sample_queries(t, n_pos, config.num_queries)
    ok = ok & jnp.all(positions == proof.positions)

    # Per-layer folded query indices: layer 0 the initial commit at the full
    # index, then the post-prefix layer, then one per committed epoch, each
    # addressed by `positions >> shift`. The shift sequence is a pure function of
    # the fold arities — the prover commits against the identical `layer_shifts()`.
    layer_shifts = config.layer_shifts()

    # Merkle: layer 0 rebuilds the initial commitment, each fold layer its commit
    # root, at the shifted query index — one batched pass per leaf-row width.
    roots = [commitment, *proof.commit_roots]
    legs = [
        (roots[i], positions >> shift, proof.layer_openings[i])
        for i, shift in enumerate(layer_shifts)
    ]
    ok = ok & verify_openings(tree, legs)

    # Fold consistency: the row-batch of the opened initial lanes must sit at the
    # queried leg of the post-prefix coset; each epoch's coset then folds to the
    # next epoch's leg, down to the final codeword.
    q = jnp.arange(positions.shape[0])
    rb = betas[:prefix]
    fri = betas[prefix:]
    prbv = lane_combine(proof.layer_openings[0].row, rb)  # [Q]
    if not arities:
        # log_dim == 0: the row-batched value is the terminal at the query index.
        ok = ok & jnp.all(proof.final_codeword[positions] == prbv)
        return ok, t

    arity0 = arities[0]
    post_rb_row = proof.layer_openings[1].row  # [Q, 2^arity0]
    inner = positions & ((1 << arity0) - 1)
    ok = ok & jnp.all(post_rb_row[q, inner] == prbv)
    expected = _fold_coset(code, post_rb_row, fri[:arity0], 0, positions >> arity0)

    cum = arity0
    for i in range(num_epochs - 1):
        next_arity = arities[i + 1]
        layer_row = proof.layer_openings[2 + i].row  # [Q, 2^next_arity]
        p_at = positions >> cum
        offset = p_at & ((1 << next_arity) - 1)
        ok = ok & jnp.all(layer_row[q, offset] == expected)
        expected = _fold_coset(
            code, layer_row, fri[cum : cum + next_arity], cum, p_at >> next_arity
        )
        cum += next_arity

    ok = ok & jnp.all(proof.final_codeword[positions >> cum] == expected)
    return ok, t


# Jitted verify body: an eager replay interprets each composite op-by-op in
# Python (issue #140). The verifier is the static key (by value, #214), so its
# config and choreography (both frozen, value-compared) fix the compiled zone.
@partial(frx.jit, static_argnames=("verifier",))
def _verify_batch_body(
    verifier: BasefoldVerifier,
    commitments: list[Array],
    z: Array,
    values: list[Array],
    proof: BasefoldProof,
    transcript: Transcript,
) -> tuple[Array, Transcript]:
    chor = verifier.choreography
    kernel = verifier.kernel
    dtype = z.dtype
    n = verifier.code.block_len
    num_vars = z.shape[0]
    config = verifier._resolved_config(num_vars)
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

    # Replay the interleaved sumcheck + fold challenges through the kernel +
    # choreography. Every round checks the running claim against the message
    # (`kernel.round_check`), frames+observes the message (`round_message` +
    # `observe_message`), observes its pre-fold pair-leaf commitment root
    # (`observe_root`), samples β, then reduces the running claim
    # (`kernel.reduce_claim`, native `s(0)+β·s(1)`). All num_vars rounds are
    # homogeneous (no peeled final round) and ride one lax.scan — the poseidon2
    # permute markers stop scaling with num_vars (#185). z_rev[r] binds the
    # variable folded in round r. The native choreography's message/root
    # observes are pass-throughs, so the wire is byte-identical.
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
        components = (zero_val, one_val)
        ok = ok & kernel.round_check(claim, components, last)
        msg = chor.round_message(*components)
        t = chor.observe_message(t, msg)
        t = chor.observe_root(t, root)
        t, beta = t.sample()
        beta = beta.reshape(())
        return (t, kernel.reduce_claim(claim, components, beta), ok), beta

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
    # mypy-enforced seam conformance — docs/reference/conventions.md
    # "Seam conformance pins".
    _: type[PcsVerifier[BasefoldCommitment, BasefoldProof]] = BasefoldVerifier
