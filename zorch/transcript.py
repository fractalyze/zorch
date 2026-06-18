# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fiat-Shamir transcript: the `Transcript` interface and a real duplex-sponge
implementation.

`DuplexTranscript` is the device-side duplex sponge (fixed-size buffers + position
scalars) — a JAX pytree whose state threads functionally under `@jit`, with no
host callback or zkVM FFI.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, Self

import jax.numpy as jnp
from jax import Array, jit, lax, vmap
from jax.tree_util import register_dataclass
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
    transcript: Transcript, dtype: Any, limbs: int = 1
) -> tuple[Transcript, Array]:
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


@partial(register_dataclass, data_fields=["state"], meta_fields=["permutation", "rate"])
@dataclass(frozen=True)
class DuplexTranscript:
    """Overwrite-mode duplex sponge implementing `Transcript`. A JAX pytree whose
    `state` buffers are the leaves and whose `permutation`/`rate` are static, so
    the whole transcript threads through `@jit` (and, later, a `lax.scan` carry).
    Every step is a device op — no host callback, no zkVM FFI."""

    permutation: Permutation
    rate: int
    state: DuplexState

    @property
    def has_dedicated_fusion(self) -> bool:
        """Whether the Fiat-Shamir permutation lowers to a dedicated fusion marker
        a vendor can expand — the gate `zorch.sumcheck.prover` reads to mark its
        scan as one register-resident sumcheck kernel (mirrors `Sponge`/
        `Compression`). False for a test `CheapPermutation`, so unit tests keep the
        plain scan."""
        return self.permutation.has_dedicated_fusion

    @classmethod
    def new(cls, permutation: Permutation, rate: int) -> DuplexTranscript:
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
        return cls(permutation, rate, state)

    def _with_state(self, state: DuplexState) -> DuplexTranscript:
        return DuplexTranscript(self.permutation, self.rate, state)

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
        return _observe_body(self, values)

    def _sample_one(self) -> tuple[DuplexTranscript, Array]:
        # Permute when input is pending or the output buffer is drained.
        need_perm = (self.state.in_pos > 0) | (self.state.out_pos == 0)
        t = lax.cond(need_perm, lambda c: c._duplexing(), lambda c: c, self)
        out_pos = t.state.out_pos - 1
        item = t.state.output_buffer[out_pos]
        return t._with_state(replace(t.state, out_pos=out_pos)), item

    def sample(self, n: int = 1) -> tuple[DuplexTranscript, Array]:
        return _sample_body(self, n)

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[DuplexTranscript, Array]:
        """Absorb `values`, then squeeze `n` challenges — the per-round
        Fiat-Shamir primitive (commit -> challenge). One method so the absorb and
        squeeze fuse into a single kernel under `@jit` by construction, never by a
        per-primitive pattern-match (the repo's fusion contract)."""
        return _observe_and_sample_body(self, values, n)

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
        return _check_witness_body(self, witness, pow_bits)

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
# the Python-loop `sample` re-traces its `lax.cond` branches — the full
# permutation graph included — on EVERY call, and `observe`'s eager `lax.scan`
# pays the same. Routing through module-level jit makes every eager call site
# hit one process-wide cache: `permutation`/`rate` are static meta_fields with
# value-equality keys (#214), so fresh same-config transcripts reuse the trace.
# `inline=True` keeps call sites already inside a jit zone byte-identical:
# without it the zone stays a nested pjit call in the outer jaxpr, which stops
# the permutation's round constants from auto-lifting into the
# `zorch.sumcheck` composite envelope (the operand layout zkx expands).


@partial(jit, static_argnames=("n",), inline=True)
def _sample_body(t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, Array]:
    outs = []
    for _ in range(n):
        t, x = t._sample_one()
        outs.append(x.reshape(()))
    return t, jnp.stack(outs)


@partial(jit, inline=True)
def _observe_body(t: DuplexTranscript, values: Array) -> DuplexTranscript:
    base_dtype = t.state.sponge_state.dtype
    flat = lax.bitcast_convert_type(values, base_dtype).reshape(-1)
    if flat.shape[0] == 0:
        return t

    rate = t.rate
    permutation = t.permutation

    def step(
        carry: tuple[Array, Array, Array], x: Array
    ) -> tuple[tuple[Array, Array, Array], None]:
        in_buf, in_pos, sponge = carry
        in_buf = in_buf.at[in_pos].set(x)
        new_in_pos = in_pos + 1
        full = new_in_pos == rate

        def perm(args: tuple[Array, Array]) -> tuple[Array, Array]:
            sp, ib = args
            # Full block: new_in_pos == rate, so the whole rate lane is `ib`.
            new_sponge = _absorb_permute(permutation, sp, ib, new_in_pos, rate)
            return new_sponge, jnp.zeros_like(ib)

        sponge, in_buf = lax.cond(full, perm, lambda a: a, (sponge, in_buf))
        in_pos_out = jnp.where(full, jnp.int32(0), new_in_pos)
        return (in_buf, in_pos_out, sponge), None

    init = (t.state.input_buffer, t.state.in_pos, t.state.sponge_state)
    (in_buf, in_pos, sponge), _ = lax.scan(step, init, flat)

    # If the final scan step permuted (in_pos == 0 at exit), the post-permute
    # sponge prefix is the fresh output; otherwise the next sample permutes.
    last_was_perm = in_pos == 0
    out_pos = jnp.where(last_was_perm, jnp.int32(rate), jnp.int32(0))
    output_buffer = jnp.where(
        last_was_perm, sponge[:rate], jnp.zeros(rate, dtype=base_dtype)
    )
    return t._with_state(DuplexState(in_buf, output_buffer, sponge, in_pos, out_pos))


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


if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[Transcript] = DuplexTranscript
    _grinding: type[GrindingTranscript] = DuplexTranscript
