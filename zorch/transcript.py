# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fiat-Shamir transcript: the `Transcript` interface and a real duplex-sponge
implementation.

`DuplexTranscript` is the device-side duplex sponge (fixed-size buffers + position
scalars) — a JAX pytree whose state threads functionally under `@jit`, with no
host callback or zkVM FFI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache, partial
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar

import jax
import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jax.tree_util import register_dataclass, tree_map
from zk_dtypes import pfinfo

from zorch.hash.permutation import Permutation

# Candidate window for the grind search: each `lax.while_loop` step tests this
# many witnesses at once (static shape), trading device memory for fewer
# host-visible iterations.
_GRIND_CHUNK = 1 << 16


class GrindError(RuntimeError):
    """Raised when a proof-of-work grind cannot return a valid witness -- either
    the field is too wide for the uint32 search (needs x64) or the searched
    candidate range was exhausted without a hit. The loud-failure contract that
    stops an unverified witness from ever being returned."""


def _validate_pow_bits(pow_bits: int) -> None:
    if not 0 <= pow_bits < 32:
        raise ValueError(f"pow_bits must be in [0, 32), got {pow_bits}")


def _pow_satisfied(sample: Array, pow_bits: int) -> Array:
    """The proof-of-work predicate: the challenge's low `pow_bits` canonical bits
    are all zero. Shared by `check_witness` and the grind search so the prover's
    search and the verifier's check can never drift."""
    mask = jnp.uint32((1 << pow_bits) - 1)
    return (sample.astype(jnp.uint32) & mask) == jnp.uint32(0)


def _require_uint32_field(field_dtype: Any) -> int:
    """The witness counter and the canonical bit-check are both uint32 (jax x64
    is off), so the field's order must fit 32 bits. Return the modulus, or raise
    loudly for a wider field -- otherwise the narrowing canonical convert fails
    with an opaque backend error instead of a clear one."""
    modulus = pfinfo(field_dtype).modulus
    if modulus > 2**32:
        raise GrindError(
            f"field order {modulus} needs more than 32 bits; the uint32 grind "
            "(jax x64 off) cannot represent its canonical witnesses"
        )
    return modulus


class Transcript(Protocol):
    @property
    def has_dedicated_fusion(self) -> bool: ...
    def observe(self, values: Array) -> Self: ...
    def sample(self, n: int = 1) -> tuple[Self, Array]: ...
    def observe_and_sample(self, values: Array, n: int = 1) -> tuple[Self, Array]: ...


class GrindingTranscript(Transcript, Protocol):
    """A `Transcript` that also supports a proof-of-work grind. Split from the
    base seam because grinding is meaningful only for a transcript that squeezes
    a field element to check leading-zero bits against -- a consumer that needs a
    PoW witness type-narrows to this, and a transcript that cannot grind never
    has to pretend it can."""

    def check_witness(self, pow_bits: int, witness: Array) -> tuple[Self, Array]: ...
    def grind(self, pow_bits: int) -> tuple[Self, Array]: ...


# Generic over the transcript flavor so the free Fiat-Shamir helpers
# (`sample_challenge`, the open/verifier's `sample_*`) preserve a
# `GrindingTranscript` rather than widening it to the base `Transcript`.
TranscriptT = TypeVar("TranscriptT", bound=Transcript)


def reinterpret_challenge(raw: Array, dtype: Any) -> Array:
    """Reinterpret consecutive transcript squeezes `raw` as one `dtype` challenge:
    the identity when `dtype` is the transcript's own field, else the extension
    element whose coefficients are the squeezes. The single definition of the
    limbs/dtype packing -- shared by `sample_challenge` and the sumcheck scan
    driver so a prover and its verifier dual cannot drift.

    Fails loud on a packing mismatch: the squeezes are already consumed, so
    silently truncating to the first element would leave the stream advanced past
    a challenge nobody received.
    """
    viewed = raw.view(dtype)
    if viewed.shape != (1,):
        raise ValueError(
            f"{raw.shape[0]} squeezes reinterpret to {viewed.shape} elements of "
            f"{dtype}; a challenge needs exactly one"
        )
    return viewed[0]


def sample_challenge(
    transcript: TranscriptT, dtype: Any, limbs: int = 1
) -> tuple[TranscriptT, Array]:
    """Squeeze one challenge of `dtype` as `limbs` transcript samples.

    A transcript squeezes elements of its own field; a challenge field that
    extends it takes `limbs` consecutive squeezes reinterpreted as the
    extension element's coefficients (`limbs == 1` with the transcript's own
    field is the identity reinterpret, via `reinterpret_challenge`). Module-level
    so a prover, its verifier dual, and any binding glue derive challenges from
    one definition -- a drift would desynchronize their Fiat-Shamir streams.
    """
    if limbs < 1:
        raise ValueError(f"limbs must be >= 1, got {limbs}")
    transcript, raw = transcript.sample(limbs)
    return transcript, reinterpret_challenge(raw, dtype)


@register_dataclass
@dataclass(frozen=True)
class DuplexState:
    """Duplex-sponge state. Fixed-size buffers + position scalars: the buffers
    keep `observe`'s absorb a single `lax.scan` (compile size independent of input
    length), and the constant shape makes the whole state a valid `lax.scan` carry
    (issue #58)."""

    input_buffer: Array  # (rate,) — valid prefix is [0:in_pos]
    output_buffer: Array  # (rate,) — valid prefix is [0:out_pos]
    sponge_state: Array  # (width,)
    in_pos: Array  # int32 0-D, 0 <= in_pos < rate
    out_pos: Array  # int32 0-D, 0 <= out_pos <= rate


def _absorb_permute(
    permutation: Permutation, sponge: Array, in_buf: Array, in_pos: Array, rate: int
) -> Array:
    """Overwrite `sponge[:in_pos]` with `in_buf` — preserving the suffix
    `sponge[in_pos:rate]`, since zeroing it would clobber prior state — then
    permute. The one absorb step shared by `observe`'s scan body (a full block,
    `in_pos == rate`) and `_duplexing` (the partial flush at sample time)."""
    idx = jnp.arange(rate, dtype=jnp.int32)
    merged = jnp.where(idx < in_pos, in_buf, sponge[:rate])
    return permutation.permute(sponge.at[:rate].set(merged))


class _FsBackend(Protocol):
    """The Fiat-Shamir sponge backend a `DuplexTranscript` dispatches every absorb /
    squeeze through — device (graph ops) or host (CPU callback, state-resident). A
    transcript meta-field, so the device/host choice is structural: every method
    routes through it and none can silently forget to honor the host (the trap the
    old per-method `if self.fs_on_host` invited -- `check_witness` once did)."""

    @property
    def on_host(self) -> bool: ...

    def observe(self, t: DuplexTranscript, values: Array) -> DuplexTranscript: ...

    def sample(self, t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, Array]: ...

    def observe_and_sample(
        self, t: DuplexTranscript, values: Array, n: int
    ) -> tuple[DuplexTranscript, Array]: ...

    def check_witness(
        self, t: DuplexTranscript, witness: Array, pow_bits: int
    ) -> tuple[DuplexTranscript, Array]: ...


@dataclass(frozen=True)
class _DeviceFs:
    """The default backend: every duplex step is a device op (the compiled graph)."""

    on_host: bool = False

    def observe(self, t: DuplexTranscript, values: Array) -> DuplexTranscript:
        return _observe_body(t, values)

    def sample(self, t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, Array]:
        return _sample_body(t, n)

    def observe_and_sample(
        self, t: DuplexTranscript, values: Array, n: int
    ) -> tuple[DuplexTranscript, Array]:
        return _observe_and_sample_body(t, values, n)

    def check_witness(
        self, t: DuplexTranscript, witness: Array, pow_bits: int
    ) -> tuple[DuplexTranscript, Array]:
        return _check_witness_body(t, witness, pow_bits)


@dataclass(frozen=True)
class _HostFs:
    """The host-FS backend: the serial sponge runs on the CPU and its state stays
    host-resident across the stream (see the host-FS backend section below)."""

    on_host: bool = True

    def observe(self, t: DuplexTranscript, values: Array) -> DuplexTranscript:
        return _observe_host(t, values)

    def sample(self, t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, Array]:
        return _sample_host(t, n)

    def observe_and_sample(
        self, t: DuplexTranscript, values: Array, n: int
    ) -> tuple[DuplexTranscript, Array]:
        return _observe_and_sample_host(t, values, n)

    def check_witness(
        self, t: DuplexTranscript, witness: Array, pow_bits: int
    ) -> tuple[DuplexTranscript, Array]:
        return _check_witness_host(t, witness, pow_bits)


_DEVICE_FS = _DeviceFs()
_HOST_FS = _HostFs()


@partial(
    register_dataclass,
    data_fields=["state"],
    meta_fields=["permutation", "rate", "fs"],
)
@dataclass(frozen=True)
class DuplexTranscript:
    """Overwrite-mode duplex sponge implementing `Transcript`. A JAX pytree whose
    `state` buffers are the leaves and whose `permutation`/`rate` are static, so
    the whole transcript threads through `@jit` (and, later, a `lax.scan` carry).
    Every step is a device op — no host callback, no zkVM FFI.

    The `fs` backend (`_DeviceFs` / `_HostFs`) chooses where every absorb / squeeze
    runs. `_HostFs` runs the serial Poseidon chain on the host CPU (a latency-bound
    hash chain idles accelerator cores, so a host-driven, per-round-kernel consumer
    offloads it), with the sponge state kept CPU-resident across the stream -- an
    eager primitive (see the host-FS backend section below). `_DeviceFs` (the
    default) leaves the device path and its compiled graph untouched. Pick one via
    `new(..., fs_on_host=)`."""

    permutation: Permutation
    rate: int
    state: DuplexState
    fs: _FsBackend = _DEVICE_FS

    @property
    def fs_on_host(self) -> bool:
        """Whether Fiat-Shamir runs on the host CPU — read off the `fs` backend.
        Kept as a bool for callers (the `sumcheck.prover` gate, the consumer's
        `.new(fs_on_host=...)`)."""
        return self.fs.on_host

    @property
    def has_dedicated_fusion(self) -> bool:
        """Whether the Fiat-Shamir permutation lowers to a dedicated fusion marker
        a vendor can expand — the LogUp-GKR jagged prover's gate
        (`zorch.logup_gkr.jagged_prover`) reads it to mark its sumcheck scan as one
        register-resident kernel (mirrors `Sponge`/`Compression`). False for a test
        `CheapPermutation`, so unit tests keep the plain scan."""
        return self.permutation.has_dedicated_fusion

    @classmethod
    def new(
        cls, permutation: Permutation, rate: int, fs_on_host: bool = False
    ) -> DuplexTranscript:
        if not 1 <= rate < permutation.width:
            raise ValueError(
                f"rate ({rate}) must satisfy 1 <= rate < width ({permutation.width})"
            )
        dtype: Any = permutation.dtype
        state = DuplexState(
            input_buffer=jnp.zeros(rate, dtype=dtype),
            output_buffer=jnp.zeros(rate, dtype=dtype),
            sponge_state=jnp.zeros(permutation.width, dtype=dtype),
            in_pos=jnp.int32(0),
            out_pos=jnp.int32(0),
        )
        return cls(permutation, rate, state, _HOST_FS if fs_on_host else _DEVICE_FS)

    def _with_state(self, state: DuplexState) -> DuplexTranscript:
        return DuplexTranscript(self.permutation, self.rate, state, self.fs)

    def _duplexing(self) -> DuplexTranscript:
        """Flush the pending input prefix and refill the output buffer."""
        st = self.state
        sponge = _absorb_permute(
            self.permutation, st.sponge_state, st.input_buffer, st.in_pos, self.rate
        )
        return self._with_state(
            DuplexState(
                input_buffer=jnp.zeros(self.rate, dtype=sponge.dtype),
                output_buffer=sponge[: self.rate],
                sponge_state=sponge,
                in_pos=jnp.int32(0),
                out_pos=jnp.int32(self.rate),
            )
        )

    def observe(self, values: Array) -> DuplexTranscript:
        """Absorb `values` (any field, flattened to the base field) into the
        transcript. The absorb is one `lax.scan` over the flat input, so the
        compiled graph size is independent of `len(values)`."""
        return self.fs.observe(self, values)

    def _sample_one(self) -> tuple[DuplexTranscript, Array]:
        # Permute when input is pending or the output buffer is drained.
        need_perm = (self.state.in_pos > 0) | (self.state.out_pos == 0)
        # `select`, not `lax.cond`: a traced-predicate `cond` reads `need_perm`
        # back to the host to choose a branch -- one device->host sync per
        # sample. The two branches are shape-equal, so selecting the
        # unconditionally-computed `_duplexing()` is byte-identical to the cond;
        # the only cost is running the permute on the no-perm path too, a net win
        # because it removes the host round-trip.
        permuted = self._duplexing()
        t = self._with_state(
            tree_map(
                lambda p, c: jnp.where(need_perm, p, c), permuted.state, self.state
            )
        )
        out_pos = t.state.out_pos - 1
        item = t.state.output_buffer[out_pos]
        return t._with_state(replace(t.state, out_pos=out_pos)), item

    def sample(self, n: int = 1) -> tuple[DuplexTranscript, Array]:
        return self.fs.sample(self, n)

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[DuplexTranscript, Array]:
        """Absorb `values`, then squeeze `n` challenges — the per-round
        Fiat-Shamir primitive (commit -> challenge). One method so the absorb and
        squeeze fuse into a single kernel under `@jit` by construction, never by a
        per-primitive pattern-match (the repo's fusion contract)."""
        return self.fs.observe_and_sample(self, values, n)

    def check_witness(
        self, pow_bits: int, witness: Array
    ) -> tuple[DuplexTranscript, Array]:
        """Observe `witness`, squeeze one challenge, and report whether its low
        `pow_bits` canonical bits are zero -- the verifier-side proof-of-work
        check, and the predicate `grind` searches against. `witness` must be a
        scalar element of the transcript's field -- the domain `grind`
        enumerates -- so the verifier accepts exactly the witness space the
        prover searched (`observe` itself would bitcast-flatten any array).
        Fully jit-traceable, so a verifier runs it inside its own `@jit` zone.
        Returns the advanced transcript (observe + one sample applied), so prover
        and verifier reach the same state from the same witness."""
        _validate_pow_bits(pow_bits)
        field_dtype = self.state.sponge_state.dtype
        _require_uint32_field(field_dtype)
        if witness.shape != () or witness.dtype != field_dtype:
            raise ValueError(
                f"witness must be a scalar {field_dtype} field element (the grind "
                f"search's domain), got shape {witness.shape} dtype {witness.dtype}"
            )
        return self.fs.check_witness(self, witness, pow_bits)

    @partial(jit, static_argnames=("pow_bits", "chunk"))
    def _grind_search(self, pow_bits: int, chunk: int) -> Array:
        """Search canonical witnesses `0, 1, 2, ...` for the lowest one whose
        challenge has `pow_bits` zero low bits. Each `lax.while_loop` step tests
        a whole `chunk`-wide window IN PARALLEL -- `vmap` over the window, not a
        sequential `lax.map` -- and keeps the lowest-index hit; the loop only
        tiles windows because the full field cannot be vmapped at once (memory),
        and it early-exits at the first window that hits. For a typical
        `pow_bits` the hit is in the first window, so the loop runs once. Returns
        the winning witness (or the trailing fallback on exhaustion -- `grind`
        re-checks it before returning). Fields wider than 32 bits raise (the
        uint32 counter/bit-check would need x64); koalabear-class fields are
        searched in full."""
        field_dtype = self.state.sponge_state.dtype
        modulus = _require_uint32_field(field_dtype)
        # Search the whole field, but cap `base` below the uint32 wrap point so
        # `base + chunk` stays in range. For a koalabear-class field this is the
        # field order; the cap only bites a field whose order nears 2**32.
        bound = jnp.uint32(min(modulus, 2**32 - chunk))
        offsets = jnp.arange(chunk, dtype=jnp.uint32)

        def satisfies(witness: Array) -> Array:
            _, sample = self.observe(witness).sample(1)
            return _pow_satisfied(sample[0], pow_bits)

        def cond(carry: tuple[Array, Array, Array]) -> Array:
            found, base, _ = carry
            return jnp.logical_and(jnp.logical_not(found), base < bound)

        def body(carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            found, base, best = carry
            candidates = (base + offsets).astype(field_dtype)
            hits = vmap(satisfies)(candidates)
            any_hit = jnp.any(hits)
            first = jnp.min(jnp.where(hits, offsets, jnp.uint32(chunk)))
            index = jnp.where(any_hit, first, jnp.uint32(0)).astype(jnp.int32)
            return (
                jnp.logical_or(found, any_hit),
                base + jnp.uint32(chunk),
                jnp.where(any_hit, candidates[index], best),
            )

        init = (jnp.bool_(False), jnp.uint32(0), jnp.zeros((), field_dtype))
        _found, _base, witness = lax.while_loop(cond, body, init)
        return witness

    def grind(
        self, pow_bits: int, *, chunk: int = _GRIND_CHUNK
    ) -> tuple[DuplexTranscript, Array]:
        """Find a proof-of-work witness and return the transcript advanced past
        it. Searches canonical witnesses `0, 1, 2, ...` for the lowest one whose
        observation squeezes a challenge with `pow_bits` zero low bits, then
        advances the transcript by `check_witness(pow_bits, witness)` -- so a
        verifier replaying `check_witness` on the witness reaches the identical
        state.

        A single-window search covers only its first `chunk` candidates and
        silently returns an invalid witness once `pow_bits` outgrows it; the
        windowed search (`_grind_search`) instead keeps advancing, and grind
        **validates the result before returning** -- raising `GrindError` rather
        than handing back an unverified witness. The host-side validation makes
        `grind` an eager (prover-side) call; the verifier's `check_witness` stays
        jit-traceable.

        Returning the lowest-index witness is soundness-neutral: a PoW witness is
        a work-proof, not a secret or nonce, so the security is the ~2**pow_bits
        work to find *any* satisfying witness -- enforced by the verifier's
        `check_witness`, independent of which witness is returned or how much of
        the field is scanned (the range only has to contain one). Selection and
        the search bound are completeness/efficiency choices, not soundness ones;
        a range too small to hold a witness raises rather than degrading
        silently."""
        _validate_pow_bits(pow_bits)
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        field_dtype = self.state.sponge_state.dtype
        if pow_bits == 0:
            # No work required: the canonical zero witness always passes.
            witness = jnp.zeros((), field_dtype)
            return self.check_witness(pow_bits, witness)[0], witness
        witness = self._grind_search(pow_bits, chunk)
        advanced, ok = self.check_witness(pow_bits, witness)
        if not bool(ok):
            raise GrindError(
                f"no proof-of-work witness with {pow_bits} zero bits found "
                "within the searched candidate range"
            )
        return advanced, witness


# Module-level cached zones behind DuplexTranscript's public ops. Outside jit,
# the Python-loop `sample` re-traces its permutation graph on EVERY call, and
# `observe`'s eager `lax.scan` pays the same. Routing through module-level jit
# makes every eager call site hit one process-wide cache: `permutation`/`rate`
# are static meta_fields with value-equality keys (#214), so fresh same-config
# transcripts reuse the trace.
# `inline=True` keeps call sites already inside a jit zone byte-identical:
# without it the zone stays a nested pjit call in the outer jaxpr, which stops
# the permutation's round constants from auto-lifting into the
# `zorch.sumcheck` composite envelope (the operand layout zkx expands).


@partial(jit, static_argnames=("n",), inline=True)
def _sample_body(t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, Array]:
    if n == 1:
        # One squeeze runs at most one permute already; the unrolled loop would
        # build a chain of one, so route straight through `_sample_one` (also the
        # exact path `_grind_search` and `check_witness` replay).
        t, x = t._sample_one()
        return t, jnp.stack([x.reshape(())])

    rate = t.rate
    st = t.state

    # Squeeze a rate-block of outputs per permutation rather than one permutation
    # per limb: the obvious per-limb form runs a `_duplexing` (permutation) on
    # every limb and selects it away while the output buffer still has limbs,
    # doing ~n permutes when ~ceil(n/rate) suffice. Build the chain of permuted
    # states ONCE -- `chain[0]` is the entry state, `chain[1]` flushes pending
    # input, `chain[i+1]` is a plain permute -- then read the n limbs out of the
    # right chain entry. Byte-identical to the per-limb form: the per-limb
    # `need_perm` selects exactly the same `_duplexing` result, so reading from
    # the chosen chain entry returns that value.
    depth = 1 + (n + rate - 1) // rate
    chain = [t]
    for _ in range(depth):
        chain.append(chain[-1]._duplexing())
    output_buffers = [c.state.output_buffer for c in chain]  # depth+1 x (rate,)

    # Select the chain entry for `perm_count` with a one-hot select over the
    # STATIC chain, NOT a traced-index gather into a stacked array
    # (`output_buffers[perm_count, ...]` / `leaves[perm_count]`): that gather
    # miscompiles on the ZKX CPU backend (fractalyze/zkx#500 class), the same
    # reason `_observe_body` unrolls its block loop. `depth` is static, so the
    # one-hot is a fixed chain of selects; `out_pos` stays a 1-D buffer gather,
    # which `_sample_one` already uses CPU-safely.
    def _pick(stacked: list, idx: Array) -> Array:
        acc = stacked[0]
        for i in range(1, len(stacked)):
            acc = jnp.where(idx == i, stacked[i], acc)
        return acc

    # Replay the per-limb schedule with traced scalars only (no field ops): track
    # how many permutes have fired (`perm_count`, the chain index) and the running
    # `out_pos`; a permute fires iff input is pending or the buffer is drained --
    # the same `need_perm` the per-limb loop tested.
    perm_count = jnp.int32(0)
    in_pos = st.in_pos
    out_pos = st.out_pos
    outs = []
    for _ in range(n):
        need_perm = (in_pos > 0) | (out_pos == 0)
        perm_count = jnp.where(need_perm, perm_count + 1, perm_count)
        in_pos = jnp.where(need_perm, jnp.int32(0), in_pos)
        out_pos = jnp.where(need_perm, jnp.int32(rate), out_pos)
        out_pos = out_pos - 1
        outs.append(_pick(output_buffers, perm_count)[out_pos].reshape(()))

    chain_state_leaves = [c.state for c in chain]
    final_state = tree_map(
        lambda *leaves: _pick(list(leaves), perm_count), *chain_state_leaves
    )
    final_state = replace(final_state, out_pos=out_pos)
    return t._with_state(final_state), jnp.stack(outs)


@partial(jit, inline=True)
def _observe_body(t: DuplexTranscript, values: Array) -> DuplexTranscript:
    base_dtype = t.state.sponge_state.dtype
    flat = lax.bitcast_convert_type(values, base_dtype).reshape(-1)
    m = flat.shape[0]
    if m == 0:
        return t

    rate = t.rate
    permutation = t.permutation
    st = t.state

    # Absorb a rate-block per permutation rather than a base element per
    # permutation: the obvious per-element form runs a full `_absorb_permute` on
    # every input and keeps only the rate-boundary one (`jnp.where(full, ...)`),
    # doing ~M permutes to absorb M elements when ~ceil(M/rate) suffice. This
    # scans over the rate-sized BLOCKS of the combined stream instead, permuting
    # once per block. Byte-identical to the per-element form: a full block in that
    # form overwrites the whole rate lane with those `rate` consecutive stream
    # elements (`new_in_pos == rate`), which is exactly
    # `permutation.permute(sponge.at[:rate].set(block))`.
    #
    # The combined stream is `input_buffer[0:in_pos] ++ flat`, runtime length
    # `length = in_pos + M`. `in_pos < rate` is static-bounded, so at most
    # `num_blocks = (rate - 1 + M) // rate` full blocks can ever form; the live
    # count `length // rate` is masked against that static bound. The trailing
    # `length % rate` elements go back into `input_buffer` for the next absorb.
    in_pos = st.in_pos
    length = in_pos + jnp.int32(m)
    active_blocks = length // rate  # runtime count of full rate-blocks
    num_blocks = (rate - 1 + m) // rate  # static upper bound on full blocks

    # Drop the unused gap `input_buffer[in_pos:rate]` from the stream: for stream
    # position `j`, the source index is `j` while `j < in_pos`, else shifted by
    # `rate - in_pos` to skip past the buffer's invalid suffix.
    combined_src = jnp.concatenate([st.input_buffer, flat])  # (rate + M,)
    total = (num_blocks + 1) * rate  # >= length, with a rate-block of tail slack
    pos = jnp.arange(total, dtype=jnp.int32)
    src_idx = pos + jnp.where(pos < in_pos, jnp.int32(0), rate - in_pos)
    src_idx = jnp.clip(src_idx, 0, combined_src.shape[0] - 1)
    combined = combined_src[src_idx]  # (total,) — valid prefix is [0:length]

    # Unroll the rate-block absorb in Python (num_blocks is STATIC) rather than
    # `lax.scan`: a scan whose array carry evolves under a per-step scatter
    # (`sponge.at[:rate].set`) and a `dynamic_slice` of the closed-over `combined`
    # is the fractalyze/zkx#500 CPU-backend miscompile, which silently corrupts
    # the CPU Fiat-Shamir transcript (the byte-identity tests are GPU-pinned, so
    # it slipped through). num_blocks = ceil((rate-1+M)/rate) is small and
    # CONSTANT for the fixed-size messages the rolled prove observes per round, so
    # the per-call graph stays O(1) across rounds; a one-time large observe pays a
    # small static unroll. `concatenate` (not `sponge.at[:rate].set`) overwrites
    # the rate lanes, byte-identical to the per-element absorb, and the static
    # `combined[k*rate:(k+1)*rate]` avoids the traced-index dynamic_slice.
    sponge = st.sponge_state
    if isinstance(num_blocks, int):
        for k in range(num_blocks):
            block = combined[k * rate : (k + 1) * rate]  # static slice
            permuted = permutation.permute(jnp.concatenate([block, sponge[rate:]]))
            # Blocks past the live count are padding-only: leave the sponge alone.
            sponge = jnp.where(jnp.int32(k) < active_blocks, permuted, sponge)
    else:
        # Symbolic M (shape-poly export): num_blocks is a symbolic dim, so the
        # Python unroll is unavailable — loop the rate-blocks with a `fori_loop`.
        # Export targets the GPU sponge plugin, not the CPU backend whose zkx#500
        # scan miscompile the unroll protects against; the body is byte-identical
        # (overwrite the rate lanes via concatenate, permute, mask blocks past the
        # live count), so a concrete refinement reproduces the unrolled bytes.
        def absorb_block(k: Array, sponge: Array) -> Array:
            block = lax.dynamic_slice_in_dim(combined, k * rate, rate)
            permuted = permutation.permute(jnp.concatenate([block, sponge[rate:]]))
            return jnp.where(k < active_blocks, permuted, sponge)

        sponge = lax.fori_loop(0, num_blocks, absorb_block, sponge)

    # The `length % rate` tail of the combined stream stays pending in the input
    # buffer (positions [0:in_pos_out]); higher slots are zero (overwrite mode
    # reads only [0:in_pos]). `tail_start` is the live tail's stream offset.
    tail_len = length - active_blocks * rate
    tail_start = active_blocks * rate
    tail = lax.dynamic_slice_in_dim(combined, tail_start, rate)
    slot = jnp.arange(rate, dtype=jnp.int32)
    in_buf = jnp.where(slot < tail_len, tail, jnp.zeros(rate, dtype=base_dtype))
    in_pos_out = tail_len

    # If the last full block permuted and no tail remains (in_pos_out == 0), the
    # post-permute sponge prefix is the fresh output; otherwise the next sample
    # permutes. Matches the per-element form's `last_was_perm` exactly.
    last_was_perm = in_pos_out == 0
    out_pos = jnp.where(last_was_perm, jnp.int32(rate), jnp.int32(0))
    output_buffer = jnp.where(
        last_was_perm, sponge[:rate], jnp.zeros(rate, dtype=base_dtype)
    )
    return t._with_state(
        DuplexState(in_buf, output_buffer, sponge, in_pos_out, out_pos)
    )


@partial(jit, static_argnames=("n",), inline=True)
def _observe_and_sample_body(
    t: DuplexTranscript, values: Array, n: int
) -> tuple[DuplexTranscript, Array]:
    return _sample_body(_observe_body(t, values), n)


@partial(jit, static_argnames=("pow_bits",), inline=True)
def _check_witness_body(
    t: DuplexTranscript, witness: Array, pow_bits: int
) -> tuple[DuplexTranscript, Array]:
    advanced, sample = _sample_body(_observe_body(t, witness), 1)
    return advanced, _pow_satisfied(sample[0], pow_bits)


# ============================================================================
# Host-FS backend — the duplex sponge run on the host CPU. A serial hash chain
# idles accelerator cores, so a host-driven, per-round-kernel consumer offloads it;
# The `_HostFs` backend routes observe/sample/observe_and_sample/check_witness here
# and the device path above is untouched.
#
# The sponge state lives on the CPU for the whole Fiat-Shamir stream: the host op
# calls the CPU sponge jit DIRECTLY on host-resident leaves and moves only `values`
# in / the squeezed challenge back to the compute device. Crossing the host
# boundary costs per array-leaf (a `jax.pure_callback` round-trip of the 5 state
# leaves every hop was ~6x that and dominated a warm prove once the sponge math got
# fast); keeping the state resident drops each hop to one `values` in + one
# challenge out. This is an eager primitive -- it is the production jit=False
# relaunch's per-round FS, not a graph op.
#
# Byte-identical to the device sponge: the jit reconstructs a `DuplexTranscript`
# (fs_on_host defaults False -> no recursion) and runs the SAME `_observe_body`/
# `_sample_body`. The `permutation` must lower its `permute` to the host (a raw
# permute, not one pinning an inner accelerator jit). State leaves cross as their
# own field dtype -- jax_fork#45 (FFI ABI carries ZK field types) retired the
# uint32 bitcast workaround for the #44 abort.
# ============================================================================


@cache
def _host_cpu() -> jax.Device:
    """The host device, resolved lazily so a host-FS transcript -- not merely
    importing -- is what requires a CPU backend."""
    return jax.devices("cpu")[0]


def _state_leaves(
    state: DuplexState,
) -> tuple[Array, Array, Array, Array, Array]:
    """The five `DuplexState` arrays in field order — the jagged LogUp-GKR prover's
    `zorch.sumcheck` marker threads them as its `lax.composite` operands and reads
    them back, and a downstream consumer reads them in this order, so every
    producer/consumer shares this one ordering. (`_state_on_host` also walks them to
    commit each leaf to the CPU.)"""
    return (
        state.input_buffer,
        state.output_buffer,
        state.sponge_state,
        state.in_pos,
        state.out_pos,
    )


def _host_raw(permutation: Permutation) -> Permutation:
    """The CPU sponge needs a permutation whose `permute` lowers to the host. A
    wrapper pinning `permute` to an accelerator exposes the underlying one as
    `_inner`; unwrap to it -- the accelerator jit is what host-FS bypasses."""
    return getattr(permutation, "_inner", permutation)


# The CPU sponge jit per config -- the host sponge math, memoized so a fixed
# closure compiles once. Runs on the CPU because the caller feeds it a host-
# resident state (`_state_on_host`) and CPU-committed `values`.
@cache
def _host_observe_jit(perm: Permutation, rate: int) -> Any:
    @jit
    def f(s: DuplexState, x: Array) -> DuplexState:
        return DuplexTranscript(perm, rate, s).observe(x).state

    return f


@cache
def _host_sample_jit(perm: Permutation, rate: int, n: int) -> Any:
    @jit
    def f(s: DuplexState) -> tuple[DuplexState, Array]:
        t, out = DuplexTranscript(perm, rate, s).sample(n)
        return t.state, out

    return f


@cache
def _host_obs_sample_jit(perm: Permutation, rate: int, n: int) -> Any:
    @jit
    def f(s: DuplexState, x: Array) -> tuple[DuplexState, Array]:
        t, out = DuplexTranscript(perm, rate, s).observe(x).sample(n)
        return t.state, out

    return f


@cache
def _host_compute_device() -> jax.Device:
    """Where a squeezed challenge returns to -- the device the surrounding eager
    prove computes on (its kernels consume the challenge)."""
    return jax.devices()[0]


def _on_host(x: Array) -> bool:
    return next(iter(x.devices())).platform == "cpu"


def _state_on_host(state: DuplexState) -> DuplexState:
    """Commit the sponge state to the CPU. The first hop pays it once; the host
    sponge returns host leaves, so later hops find it already resident and the
    per-hop device<->host round-trip of the 5 state leaves disappears."""
    if _on_host(state.sponge_state):
        return state
    c = _host_cpu()
    return DuplexState(*(jax.device_put(leaf, c) for leaf in _state_leaves(state)))


def _observe_host(transcript: DuplexTranscript, values: Array) -> DuplexTranscript:
    """`observe` on the host sponge; the state stays host-resident."""
    s = _state_on_host(transcript.state)
    f = _host_observe_jit(_host_raw(transcript.permutation), transcript.rate)  # type: ignore[arg-type]
    return transcript._with_state(f(s, jax.device_put(values, _host_cpu())))


def _sample_host(
    transcript: DuplexTranscript, n: int = 1
) -> tuple[DuplexTranscript, Array]:
    """`sample` n raw squeezes on the host sponge; the state stays host-resident,
    the challenge returns to the compute device."""
    s = _state_on_host(transcript.state)
    f = _host_sample_jit(_host_raw(transcript.permutation), transcript.rate, n)  # type: ignore[arg-type]
    state, out = f(s)
    return (
        transcript._with_state(state),
        jax.device_put(out, _host_compute_device()),
    )


def _observe_and_sample_host(
    transcript: DuplexTranscript, values: Array, n: int = 1
) -> tuple[DuplexTranscript, Array]:
    """`observe_and_sample`: absorb then squeeze n raw in one host hop -- the
    per-round Fiat-Shamir primitive. The challenge returns to the device `values`
    came from -- the accelerator the round polys are produced on -- so a
    multi-device prove gets it back where its kernels consume it (the sponge state
    is CPU-resident in steady state, so its device can't name the compute one)."""
    compute_device = next(iter(values.devices()))
    s = _state_on_host(transcript.state)
    f = _host_obs_sample_jit(_host_raw(transcript.permutation), transcript.rate, n)  # type: ignore[arg-type]
    state, out = f(s, jax.device_put(values, _host_cpu()))
    return (
        transcript._with_state(state),
        jax.device_put(out, compute_device),
    )


def _check_witness_host(
    transcript: DuplexTranscript, witness: Array, pow_bits: int
) -> tuple[DuplexTranscript, Array]:
    """`check_witness` on the host sponge — observe + one squeeze + the pow check,
    the host counterpart of `_check_witness_body`. Routing it through the backend
    (not the device body unconditionally) keeps a host-FS grind and its re-check on
    the same sponge, so the two can't disagree."""
    advanced, sample = _observe_and_sample_host(transcript, witness, 1)
    return advanced, _pow_satisfied(sample[0], pow_bits)


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[Transcript] = DuplexTranscript
    _grinding: type[GrindingTranscript] = DuplexTranscript
