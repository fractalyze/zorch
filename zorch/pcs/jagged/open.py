# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1-schedule stacked BaseFold open — the second half of the stage-5 eval proof.

The sumcheck half (``JaggedEvalRound``) reduces the trace opening to a single
claim ``D(z_final)`` over the committed dense buffer. This module opens that
claim: it batch-opens the separately committed regions (preprocessed + main) at
the trailing ``log_stacking_height`` coordinates of ``z_final`` via one shared
FRI, the BaseFold batch open SP1's ``StackedPcsProver::prove_trusted_evaluation``
performs.

The wire is SP1's, not zorch's generic ``BasefoldProver.open_batch``: every fold
layer is committed through the SP1 single-matrix commitment (``smcs``), so its
**separator-bound** root is what the transcript observes (zorch's open observes
the raw Merkle root); the per-round component roots were already bound upstream
at commit time, so they are not re-observed here; the scalar ``D(z_final)`` is
observed before the open (SP1's ``prove_untrusted_evaluation``); the FRI query
phase runs a proof-of-work grind before sampling query positions; and the final
poly the transcript binds is the folded codeword's first element, not the whole
residual codeword. Extension challenges use zorch's ``sample_challenge`` (degree
base squeezes reinterpreted as one extension element — SP1's ``sample_ext_element``
convention), and query positions take the canonical low bits of a base squeeze
(SP1's ``sample_bits``).

The fold geometry is zorch's bit-reversed Reed-Solomon foldable code, which pairs
conjugates adjacently — the order ``trace_commit`` already writes its codeword in.

References (the consumer's BaseFold FRI commit phase is the byte-match
reference pipeline):
- batch RLC weights — ``partial_lagrange`` over ``log2_ceil(total_width)`` EF
  challenges, allocated staggered across rounds.
- per-round message ``(s(0), s(1))`` — derived from the running MLE under the
  running claim, identical to the interleaved BaseFold sumcheck.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import frx
import frx.numpy as jnp
from frx import Array
from zk_dtypes import efinfo

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.basefold.batching import batch_staggered, partial_lagrange
from zorch.poly.multilinear import eval_mle, mle_fold
from zorch.transcript import (
    GrindingTranscript,
    TranscriptT,
    sample_challenge,
)
from zorch.utils.bits import log2_ceil_usize, log2_strict_usize


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["block", "digest_layers"],
    meta_fields=[],
)
@dataclass(frozen=True)
class StackedRound:
    """One committed region's retained witness for the stacked open.

    ``block`` is the ``[K, S]`` message-domain matrix in the commit's own
    layout (row ``k`` is stacked column ``k`` at
    ``S = 2^log_stacking_height``) — for a packed region it is a free reshape
    view of the dense buffer (``JaggedRegion.block``), no transpose. The open
    evaluates each block at ``stack_point``, folds their RLC, and re-encodes
    to the committed ``[S * blowup, K]`` codeword
    (bit-reversed rows = the leaves); any transposes happen inside the jitted
    zones, where XLA fuses them. ``digest_layers`` is that SMCS commit's
    layered digest tree.
    """

    block: Array
    digest_layers: list[Array]


# (rows, sibling_paths) — one SMCS batch opening, the shape the SP1 heap proof
# serializes. ``rows`` is (Q, width); ``paths`` is one (Q, digest) per tree level.
Opening = tuple[Array, list[Array]]


def sample_rlc_coeffs_bits(
    transcript: TranscriptT, nbv: int, dtype: Any
) -> tuple[TranscriptT, Array]:
    """The staggered partial-Lagrange RLC weights from ``nbv`` extension
    challenges expanded to the eq basis (``2^nbv`` weights). Split from
    ``sample_rlc_coeffs`` so the symbolic-K open can pass a static bit count
    directly (``total_width`` is then a symbolic dim and ``log2_ceil`` of it is
    unavailable); both entries share this body so the weights cannot drift."""
    if nbv == 0:
        return transcript, jnp.ones(1, dtype)
    limbs = efinfo(dtype).degree
    samples = []
    for _ in range(nbv):
        transcript, challenge = sample_challenge(transcript, dtype, limbs)
        samples.append(challenge)
    return transcript, partial_lagrange(jnp.stack(samples))


def sample_rlc_coeffs(
    transcript: TranscriptT, total_width: int, dtype: Any
) -> tuple[TranscriptT, Array]:
    """The staggered partial-Lagrange RLC weights over the batch's total
    column width: ``log2_ceil(total_width)`` extension challenges expanded to
    the eq basis. One definition driven by the open and its verifier dual,
    so the batching weights cannot drift between their Fiat-Shamir streams."""
    return sample_rlc_coeffs_bits(transcript, log2_ceil_usize(total_width), dtype)


def sample_query_positions(
    transcript: TranscriptT, block_len: int, num_queries: int
) -> tuple[TranscriptT, Array]:
    """SP1's ``sample_bits`` rule: one base squeeze per query, masked to the
    canonical low ``log2(block_len)`` bits. One definition driven by the open
    and its verifier dual — zorch's ``sample_positions`` reduces the Mont
    bitpattern mod the block length instead, a different wire."""
    transcript, raw = transcript.sample(num_queries)
    mask = jnp.uint32((1 << log2_strict_usize(block_len)) - 1)
    return transcript, (raw.astype(jnp.uint32) & mask).astype(jnp.int32)


@partial(
    frx.tree_util.register_dataclass,
    data_fields=[
        "component_commitments",
        "fri_raw_roots",
        "fri_commitments",
        "univariate_messages",
        "final_poly",
        "pow_witness",
        "batch_evals",
        "component_openings",
        "query_openings",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class StackedOpenProof:
    """The stacked BaseFold open proof, byte-matched field-for-field against the
    SP1 reference dump.

    A registered pytree so the proof crosses the open's `@frx.jit` boundary.

    component_commitments: per round, the shape-bound SMCS root of the
        committed codeword — SP1's ``merkle_tree_commitments``, the roots the
        verifier checks the component openings against; the structure rebind
        ties each to the statement's (preamble-observed) commitment.
    fri_raw_roots / fri_commitments: per fold layer, the raw Merkle root and the
        SP1 separator-bound root (the transcript observes the bound one).
    univariate_messages: per fold round, the ``(s(0), s(1))`` sumcheck message
        pair, ``(num_vars, 2)``. The transcript observes them and the shard
        wire serializes them, so the open retains them rather than making the
        serializer replay the fold.
    final_poly: the folded codeword's first element (the residual constant).
    pow_witness: the FRI query-phase proof-of-work witness.
    batch_evals: per round, the ``(K,)`` column evaluations at ``stack_point``.
    component_openings: per round, the codeword rows + paths at the query
        positions.
    query_openings: per fold layer, the pair-leaf rows + paths at the halved
        query positions.
    """

    component_commitments: list[Array]
    fri_raw_roots: Array
    fri_commitments: Array
    univariate_messages: Array
    final_poly: Array
    pow_witness: Array
    batch_evals: list[Array]
    component_openings: list[Opening]
    query_openings: list[Opening]


# K-shaped zone: everything whose shapes carry a round's column count K,
# collapsed to the single [S] RLC'd mle + scalar claim the fold zone consumes.
# Recompiles per K tuple, but its graph is a few simple large-data kernels, so
# that compile is cheap.
@partial(frx.jit, static_argnames=("code", "rlc_bits", "total_width"))
def _open_prologue(
    code: BitReversedReedSolomon,
    blocks: list[Array],
    stack_point: Array,
    dense_eval: Array,
    transcript: GrindingTranscript,
    *,
    rlc_bits: int | None,
    total_width: int | None,
) -> tuple[list[Array], list[Array], Array, Array, GrindingTranscript]:
    ef_dtype = dense_eval.dtype
    # [S, K] views; inside the jit XLA fuses the transposes into the consumers,
    # so no [S, K] copy materializes.
    mles = [block.T for block in blocks]
    # Each round's per-column evaluation at the stacking point (SP1's batch
    # evaluations, observed into the transcript). vmap over the column axis serves
    # concrete and symbolic K alike, byte-identical to a per-column unroll.
    batch_evals = [
        frx.vmap(eval_mle, in_axes=(1, None))(mle, stack_point) for mle in mles
    ]

    t = transcript
    # SP1's prove_untrusted_evaluation observes the scalar D(z_final) first.
    t = t.observe(dense_eval)
    for evals in batch_evals:
        t = t.observe(evals)

    # Staggered RLC weights over the total column width, allocated staggered
    # across the rounds (round r consumes its K_r weights). Under symbolic K the
    # bit count is the static bracket arg (``total_width`` is symbolic, so its
    # ``log2_ceil`` is unavailable); ``batch_staggered`` consumes the leading
    # ``total_width`` of the ``2^rlc_bits`` weights.
    if rlc_bits is not None:
        t, coeffs = sample_rlc_coeffs_bits(t, rlc_bits, ef_dtype)
    else:
        assert total_width is not None  # the wrapper passes exactly one of the two
        t, coeffs = sample_rlc_coeffs(t, total_width, ef_dtype)
    # mle.T is the [K, S] message the commit encoded; the trailing .T lands the
    # [S*blowup, K] leaf-major layout whose bit-reversed rows are the committed
    # leaves, so these paths authenticate against the same digest tree. Kept
    # per round only for the query-phase openings.
    round_codewords = [code.encode(mle.T).T for mle in mles]
    mle = batch_staggered(mles, coeffs)
    claim = batch_staggered(batch_evals, coeffs)
    return batch_evals, round_codewords, mle, claim, t


# K-independent zone: the interleaved sumcheck + FRI fold, grind, query
# positions, and fold-layer openings over the single RLC'd [S] mle. No shape
# here carries K, so this — the open's dominant codegen — is shared by every
# shard of one (S, blowup, num_queries, pow_bits) configuration.
@partial(frx.jit, static_argnames=("smcs", "code", "num_queries", "pow_bits"))
def _open_fold(
    smcs: SingleMatrixCommitmentScheme,
    code: BitReversedReedSolomon,
    mle: Array,
    claim: Array,
    stack_point: Array,
    transcript: GrindingTranscript,
    *,
    num_queries: int,
    pow_bits: int,
) -> tuple[Array, Array, Array, Array, Array, Array, list[Opening], GrindingTranscript]:
    ef_dtype = claim.dtype
    bf_dtype = code.dtype
    ef_degree = efinfo(ef_dtype).degree
    num_vars = stack_point.shape[0]
    t = transcript

    # By code linearity the batched codeword is encode(mle), not
    # Σ coeffs·encode(columns) — the latter materializes the [S*blowup, K] EF
    # product (~25 GiB on wide shards) the reduce can't fuse (NTT lays the leaf
    # axis contiguous, not K).
    codeword = code.encode(mle)

    # Domain separation: bind the fold-round count (mirrors the reference).
    t = t.observe(jnp.asarray(num_vars, bf_dtype))

    # Interleaved sumcheck + pre-fold pair-leaf FRI fold, one variable per round.
    raw_roots: list[Array] = []
    bound_roots: list[Array] = []
    messages: list[Array] = []
    fold_layers: list[Opening] = []
    zero = jnp.zeros((), ef_dtype)
    for i in range(num_vars):
        # The sumcheck message (s(0), s(1)) for the variable bound this round
        # (stack_point[-(i+1)]), from the running MLE under the running claim.
        unbound = stack_point[: num_vars - i]
        zero_mle = mle_fold(mle, zero)
        rest = unbound[:-1]
        zero_val = eval_mle(zero_mle, rest) if rest.shape[0] > 0 else zero_mle[0]
        one_val = (claim - zero_val) / unbound[-1] + zero_val

        # Commit the codeword's conjugate-pair leaves through the SP1 SMCS so the
        # transcript binds the separator-bound root. Bit-reversed order pairs
        # conjugates adjacently, so reshaping the base-field view into [n//2, *]
        # rows lands each pair in one leaf.
        n = codeword.shape[0]
        leaves = frx.lax.bitcast_convert_type(codeword, bf_dtype).reshape(n // 2, -1)
        bound_root, digest_layers = smcs.commit(leaves)
        raw_roots.append(digest_layers[-1][0])
        bound_roots.append(bound_root)
        fold_layers.append((leaves, digest_layers))

        message = jnp.stack([zero_val, one_val])
        messages.append(message)
        t = t.observe(message)
        t = t.observe(bound_root)
        t, beta = sample_challenge(t, ef_dtype, ef_degree)

        codeword = code.fold(codeword, beta)
        mle = mle_fold(mle, beta)
        claim = zero_val + beta * one_val

    # The residual codeword is the base-code encoding of the final claim (a
    # constant); SP1 binds only its first element as the cleartext final poly.
    final_poly = codeword[0]
    t = t.observe(jnp.atleast_1d(final_poly))

    # FRI query-phase proof-of-work grind (zorch#170). Even at pow_bits == 0 the
    # canonical-zero witness advances the transcript (observe + squeeze), so the
    # query positions depend on it.
    t, pow_witness = t.grind(pow_bits)

    # Query positions: SP1's sample_bits — the canonical value, not the Mont
    # bitpattern.
    t, positions = sample_query_positions(t, code.block_len, num_queries)

    # Each fold layer's pair-leaf opened at the cumulatively halved positions.
    layer_positions = code.layer_positions(positions, num_vars)
    query_openings = [
        smcs.open_batch(layer_positions[i], leaves, digest_layers)
        for i, (leaves, digest_layers) in enumerate(fold_layers)
    ]

    return (
        jnp.stack(raw_roots),
        jnp.stack(bound_roots),
        jnp.stack(messages),
        final_poly,
        pow_witness,
        positions,
        query_openings,
        t,
    )


# K-shaped query zone: per-round component row gathers and shape-bound roots.
# Transcript-free, so it sits after the fold zone without touching the
# Fiat-Shamir stream.
@partial(frx.jit, static_argnames=("smcs", "bf_dtype"))
def _open_queries(
    smcs: SingleMatrixCommitmentScheme,
    positions: Array,
    round_codewords: list[Array],
    digest_layers: list[list[Array]],
    *,
    bf_dtype: Any,
) -> tuple[list[Opening], list[Array]]:
    # Each component matrix opened at the full positions.
    component_openings = [
        smcs.open_batch(positions, cw, layers)
        for layers, cw in zip(digest_layers, round_codewords, strict=True)
    ]
    # The per-round shape-bound roots (SP1's merkle_tree_commitments): the
    # verifier checks the component openings against these, then ties each to
    # the statement's commitment via the structure rebind.
    component_commitments = [
        smcs.bind_root(
            layers[-1][0],
            log2_strict_usize(cw.shape[0]),
            cw.shape[1],
            bf_dtype,
        )
        for layers, cw in zip(digest_layers, round_codewords, strict=True)
    ]
    return component_openings, component_commitments


def stacked_basefold_open(
    smcs: SingleMatrixCommitmentScheme,
    code: BitReversedReedSolomon,
    rounds: Sequence[StackedRound],
    z_final: Array,
    dense_eval: Array,
    log_stacking_height: int,
    *,
    num_queries: int,
    pow_bits: int,
    transcript: GrindingTranscript,
    rlc_bits: int | None = None,
) -> tuple[StackedOpenProof, GrindingTranscript]:
    """Open the stacked dense at ``z_final`` via one batched FRI over ``rounds``.

    ``transcript`` must be a real grinding transcript at the state SP1 enters the
    open with (the component roots already bound upstream) — the open observes
    ``dense_eval`` and the batch evals, samples the RLC + fold challenges, grinds,
    then samples query positions, so a scripted replay cannot drive it. Returns
    ``(proof, transcript)``.

    Runs as three ``@jit`` zones split on the K (column count) compile key —
    K-shaped prologue, K-independent fold core (the dominant codegen),
    K-shaped component queries — so shards differing only in K share the fold
    executable and re-pay only the two cheap zones. Under a consumer's outer
    ``@jit`` the zones inline. Byte-identical either way: the cuts sit on the
    value flow, not in it.

    ``rlc_bits`` switches the open into the **symbolic-K** mode used for a
    ``frx.export``: when set, each round's column count ``K`` may be a symbolic
    dim. The two K-dependent steps are made shape-polymorphic — the per-column
    evals run under ``vmap`` over the column axis (not a static ``range(K)``),
    and the RLC samples exactly ``rlc_bits`` challenges instead of deriving the
    count from ``log2_ceil(total_width)``. ``rlc_bits`` must equal ``log2_ceil``
    of the bracket's total column width (one binary per power-of-2 column
    bracket); the resulting ``2^rlc_bits`` weights cover any ``K`` in the
    bracket, and ``batch_staggered`` consumes the leading ``total_width`` of
    them. Left ``None`` (the default) the open is byte-identical to the
    concrete path.
    """
    if not rounds:
        raise ValueError("the stacked open needs at least one committed round")

    if log_stacking_height < 1:
        # z_final[-0:] is the whole vector, not an empty suffix; reject the
        # degenerate zero-stacking open up front (the verifier rejects it too).
        raise ValueError(
            f"need at least one stacking variable, got "
            f"log_stacking_height={log_stacking_height}"
        )
    # Fail loud at the code/mle seam: a stacking-height mismatch would otherwise
    # surface as a cryptic RS-encode shape error inside the re-encode.
    if z_final.shape[0] < log_stacking_height:
        raise ValueError(
            f"z_final has {z_final.shape[0]} coordinates, fewer than "
            f"log_stacking_height={log_stacking_height}"
        )
    stacking = 1 << log_stacking_height
    if code.message_len != stacking:
        raise ValueError(
            f"code message_len {code.message_len} != stacking height {stacking}"
        )
    if any(rd.block.shape[1] != stacking for rd in rounds):
        raise ValueError(f"every round block must stack to height {stacking}")

    stack_point = z_final[-log_stacking_height:]
    blocks = [rd.block for rd in rounds]
    total_width = None if rlc_bits is not None else sum(int(b.shape[0]) for b in blocks)
    batch_evals, round_codewords, mle, claim, t = _open_prologue(
        code,
        blocks,
        stack_point,
        dense_eval,
        transcript,
        rlc_bits=rlc_bits,
        total_width=total_width,
    )
    (
        fri_raw_roots,
        fri_commitments,
        univariate_messages,
        final_poly,
        pow_witness,
        positions,
        query_openings,
        t,
    ) = _open_fold(
        smcs,
        code,
        mle,
        claim,
        stack_point,
        t,
        num_queries=num_queries,
        pow_bits=pow_bits,
    )
    component_openings, component_commitments = _open_queries(
        smcs,
        positions,
        round_codewords,
        [rd.digest_layers for rd in rounds],
        bf_dtype=code.dtype,
    )

    proof = StackedOpenProof(
        component_commitments=component_commitments,
        fri_raw_roots=fri_raw_roots,
        fri_commitments=fri_commitments,
        univariate_messages=univariate_messages,
        final_poly=final_poly,
        pow_witness=pow_witness,
        batch_evals=batch_evals,
        component_openings=component_openings,
        query_openings=query_openings,
    )
    return proof, t


__all__ = [
    "StackedRound",
    "StackedOpenProof",
    "sample_query_positions",
    "sample_rlc_coeffs",
    "sample_rlc_coeffs_bits",
    "stacked_basefold_open",
]
