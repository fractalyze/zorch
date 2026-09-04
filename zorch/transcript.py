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
from typing import TYPE_CHECKING, Any, Protocol, Self

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array, jit, lax, vmap
from frx.tree_util import register_dataclass, tree_map
from hash_frx.fusion import fused_region
from hash_frx.permutation import Permutation
from typing_extensions import TypeVar as TypeVarWithDefault
from zk_dtypes import pfinfo

from zorch.grind import grind_search

# Candidate window for the grind search: each `lax.while_loop` step tests this
# many witnesses at once (static shape), trading device memory for fewer
# host-visible iterations.
_GRIND_CHUNK = 1 << 16


class GrindError(RuntimeError):
    """Raised when a proof-of-work grind cannot run: the field is too wide for
    the uint32 search (needs x64)."""


def _validate_pow_bits(pow_bits: int) -> None:
    if not 0 <= pow_bits < 32:
        raise ValueError(f"pow_bits must be in [0, 32), got {pow_bits}")


def _pow_satisfied(sample: Array, pow_bits: int) -> Array:
    """The proof-of-work predicate: the challenge's low `pow_bits` canonical bits
    are all zero. Shared by `check_witness` and the grind search so the prover's
    search and the verifier's check can never drift."""
    mask = fnp.uint32((1 << pow_bits) - 1)
    return (sample.astype(fnp.uint32) & mask) == fnp.uint32(0)


def _require_uint32_field(field_dtype: Any) -> int:
    """The witness counter and the canonical bit-check are both uint32 (frx x64
    is off), so the field's order must fit 32 bits. Return the modulus, or raise
    loudly for a wider field -- otherwise the narrowing canonical convert fails
    with an opaque backend error instead of a clear one."""
    modulus = pfinfo(field_dtype).modulus
    if modulus > 2**32:
        raise GrindError(
            f"field order {modulus} needs more than 32 bits; the uint32 grind "
            "(frx x64 off) cannot represent its canonical witnesses"
        )
    return modulus


class Transcript(Protocol):
    @property
    def field(self) -> Any:
        """The field one `sample` word is drawn from.

        A challenge in another field is packed from consecutive words of this
        one, so how many words a challenge costs is a fact about the pair, not
        about the challenge field alone -- an extension-native sponge already
        yields an extension element per word.
        """

    @property
    def has_dedicated_fusion(self) -> bool: ...
    def observe(self, values: Array) -> Self: ...
    def sample(self, n: int = 1) -> tuple[Self, Array]: ...
    def observe_and_sample(self, values: Array, n: int = 1) -> tuple[Self, Array]: ...
    def grind(self, pow_bits: int) -> tuple[Self, Array]: ...
    def check_witness(self, witness: Array, *, pow_bits: int) -> tuple[Self, Array]: ...


# Carries the caller's concrete transcript through the FS helpers and the
# composition roles, so a consumer that hands in a `DuplexTranscript` gets one
# back rather than the widened seam. Capability plays no part — every transcript
# grinds — so this is plumbing, not a narrowing.
#
# The default keeps the parameter optional; it comes from `typing_extensions`
# because these are subscripted in base-class position, so it must exist at
# runtime (`typing` gains it in 3.13).
TranscriptT = TypeVarWithDefault("TranscriptT", bound=Transcript, default=Transcript)


@partial(jit, static_argnums=(1,))
def reinterpret_challenge(raw: Array, dtype: Any) -> Array:
    """Reinterpret consecutive transcript squeezes `raw` as one `dtype` challenge:
    the identity when `dtype` is the transcript's own field, else the extension
    element whose coefficients are the squeezes. The single definition of the
    limbs/dtype packing -- shared by `sample_challenge` and the sumcheck scan
    driver so a prover and its verifier dual cannot drift.

    Fails loud on a packing mismatch: the squeezes are already consumed, so
    silently truncating to the first element would leave the stream advanced past
    a challenge nobody received. The check runs at trace time (shapes are static),
    so jitting does not cost the loud failure.

    Jitted because these three ops are called eagerly once per FS squeeze, and
    unjitted each one launches its own kernel -- loose single-op dispatch that
    dominates a warm prove."""
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
    length), and the constant shape makes the whole state a valid `lax.scan` carry."""

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
    idx = fnp.arange(rate, dtype=fnp.int32)
    merged = fnp.where(idx < in_pos, in_buf, sponge[:rate])
    return permutation.permute(sponge.at[:rate].set(merged))


class _FsBackend(Protocol):
    """The Fiat-Shamir sponge backend a `DuplexTranscript` dispatches every absorb /
    squeeze through — device (graph ops) or host (CPU callback, state-resident). A
    transcript meta-field, so the device/host choice is structural: every method
    routes through the backend and so cannot silently ignore the host placement.
    Re-testing `fs_on_host` per method instead would let one method be missed."""

    @property
    def on_host(self) -> bool: ...

    def observe(self, t: DuplexTranscript, values: Array) -> DuplexTranscript: ...

    def sample(self, t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, Array]: ...

    def observe_and_sample(
        self, t: DuplexTranscript, values: Array, n: int
    ) -> tuple[DuplexTranscript, Array]: ...

    def check_witness(
        self, t: DuplexTranscript, witness: Array, *, pow_bits: int
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
        return _observe_and_sample_marked(t, values, n)

    def check_witness(
        self, t: DuplexTranscript, witness: Array, *, pow_bits: int
    ) -> tuple[DuplexTranscript, Array]:
        return _check_witness_body(t, witness, pow_bits=pow_bits)


@dataclass(frozen=True)
class _HostFs:
    """The host-FS backend — see the host-FS backend section below."""

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
        self, t: DuplexTranscript, witness: Array, *, pow_bits: int
    ) -> tuple[DuplexTranscript, Array]:
        return _check_witness_host(t, witness, pow_bits=pow_bits)


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
    No step crosses a zkVM FFI on any backend.

    The `fs` backend (`_DeviceFs` / `_HostFs`) chooses where every absorb / squeeze
    runs; pick one via `new(..., fs_on_host=)`. Under `_DeviceFs` (the default)
    every step is a device op and nothing calls back to the host. `_HostFs` is an
    eager host primitive -- see the host-FS backend section below for what it
    trades and why."""

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
        return self.permutation.fusion_path.is_one_kernel

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
            input_buffer=fnp.zeros(rate, dtype=dtype),
            output_buffer=fnp.zeros(rate, dtype=dtype),
            sponge_state=fnp.zeros(permutation.width, dtype=dtype),
            in_pos=fnp.int32(0),
            out_pos=fnp.int32(0),
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
                input_buffer=fnp.zeros(self.rate, dtype=sponge.dtype),
                output_buffer=sponge[: self.rate],
                sponge_state=sponge,
                in_pos=fnp.int32(0),
                out_pos=fnp.int32(self.rate),
            )
        )

    def observe(self, values: Array) -> DuplexTranscript:
        """Absorb `values` (any field, flattened to the base field) into the
        transcript. The absorb is one `lax.scan` over the flat input, so the
        compiled graph size is independent of `len(values)`."""
        return self.fs.observe(self, values)

    def absorb_on_host(self, *messages: Array) -> DuplexTranscript:
        """Absorb `messages` in order through the CPU sponge, leaving the state
        on the device it came from. Byte-identical to the same sequence of
        `observe` calls -- the same `_observe_body`, only relocated.

        The win is placement, not algorithm. A sponge on an accelerator runs one
        warp-cooperative permute per rate-block, where ~21 rounds of full-warp
        shuffle latency dominate a few hundred field ops; a CPU pays none of that.
        Measured on koalabear16 at rate 8: 5.4 us per permutation on an RTX 5090
        against 1.0 us on the host. Two host crossings are not worth a short
        message, so the caller applies its own length threshold -- the call site
        is where the message length that justifies relocating is known.

        Variadic because the crossing, not the absorb, is the repeated cost: a
        Fiat-Shamir step is usually several messages in a fixed order (a length
        prefix, then its payload), and absorbing them one call at a time drags the
        state back to the device between each. The whole sequence costs one round
        trip.

        Only THIS absorb relocates; `fs` still decides where the stream runs.

        Eager only -- it moves buffers across the host boundary, so it cannot
        run inside a traced region; `observe` is the in-graph form.
        """
        if not messages:
            return self
        if self.fs.on_host:
            # The stream already runs there; relocating is the backend's job.
            t = self
            for m in messages:
                t = t.observe(m)
            return t
        if isinstance(self.state.sponge_state, frx.core.Tracer) or any(
            isinstance(m, frx.core.Tracer) for m in messages
        ):
            raise ValueError(
                "absorb_on_host is eager (it moves the sponge state across the "
                "host boundary); call observe() inside a traced region"
            )
        device = next(iter(self.state.sponge_state.devices()))
        t = self
        for m in messages:
            # `_observe_host` commits the state to the CPU on the first hop and
            # returns host leaves, so the rest of the sequence finds it resident.
            t = _observe_host(t, m)
        return t._with_state(_state_on_device(t.state, device))

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
                lambda p, c: fnp.where(need_perm, p, c), permuted.state, self.state
            )
        )
        out_pos = t.state.out_pos - 1
        item = t.state.output_buffer[out_pos]
        return t._with_state(replace(t.state, out_pos=out_pos)), item

    def sample(self, n: int = 1) -> tuple[DuplexTranscript, Array]:
        return self.fs.sample(self, n)

    @property
    def field(self) -> Any:
        return self.state.sponge_state.dtype

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[DuplexTranscript, Array]:
        """Absorb `values`, then squeeze `n` raw challenge words — the per-round
        Fiat-Shamir primitive (commit -> challenge). One method so the absorb and
        squeeze fuse into a single kernel under `@jit` by construction, never by a
        per-primitive pattern-match (the repo's fusion contract).

        This method is an FS entry point, and there are exactly two:

        - a **typed field challenge** -> `ChallengePolicy.observe_and_sample`,
          which owns the limbs<->dtype packing (and calls this underneath).
          Spelling `observe_and_sample(v, limbs)` + `reinterpret_challenge(raw,
          dtype)` by hand re-implements that policy's body -- reach for the policy.
        - **raw squeezes** in the transcript's own field -> this method.

        Never a backend body (`_observe_and_sample_marked` and its siblings): a
        body is one placement's implementation, and only going through the
        transcript honours `fs_on_host`."""
        return self.fs.observe_and_sample(self, values, n)

    def check_witness(
        self, witness: Array, *, pow_bits: int
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
        return self.fs.check_witness(self, witness, pow_bits=pow_bits)

    @partial(jit, static_argnames=("pow_bits", "chunk"))
    def _grind_search(self, pow_bits: int, chunk: int) -> Array:
        """Search canonical witnesses `0, 1, 2, ...` for the lowest one whose
        challenge has `pow_bits` zero low bits — `grind.grind_search` over the
        challenge predicate (`vmap` over the window, not a sequential
        `lax.map`). Returns the winning witness (or the trailing fallback on
        exhaustion -- `grind` re-checks it before returning). Fields wider than
        32 bits raise (the uint32 counter/bit-check would need x64);
        koalabear-class fields are searched in full (`bound` = the field
        order)."""
        field_dtype = self.state.sponge_state.dtype
        modulus = _require_uint32_field(field_dtype)

        def satisfies(witness: Array) -> Array:
            _, sample = self.observe(witness).sample(1)
            return _pow_satisfied(sample[0], pow_bits)

        def check_batch(counters: Array) -> Array:
            return vmap(satisfies)(counters.astype(field_dtype))

        return grind_search(check_batch, modulus, chunk).astype(field_dtype)

    def grind(
        self, pow_bits: int, *, chunk: int = _GRIND_CHUNK
    ) -> tuple[DuplexTranscript, Array]:
        """Find a proof-of-work witness and return the transcript advanced past
        it via `check_witness`, so a verifier replaying it reaches the same state.
        Searches canonical witnesses for the lowest whose squeezed challenge has
        `pow_bits` zero low bits. Jit-traceable and does not raise on an exhausted
        search: `check_witness` is the soundness gate, so which witness the search
        returns is soundness-neutral."""
        _validate_pow_bits(pow_bits)
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        field_dtype = self.state.sponge_state.dtype
        if pow_bits == 0:
            # No work required: the canonical zero witness always passes.
            witness = fnp.zeros((), field_dtype)
            return self.check_witness(witness, pow_bits=pow_bits)[0], witness
        witness = self._grind_search(pow_bits, chunk)
        advanced, _ = self.check_witness(witness, pow_bits=pow_bits)
        return advanced, witness


# Module-level cached zones behind DuplexTranscript's public ops. Outside jit,
# the Python-loop `sample` re-traces its permutation graph on EVERY call, and
# `observe`'s eager `lax.scan` pays the same. Routing through module-level jit
# makes every eager call site hit one process-wide cache: `permutation`/`rate`
# are static meta_fields with value-equality keys (fractalyze/zorch#214), so
# fresh same-config transcripts reuse the trace.
# `inline=True` keeps call sites already inside a jit zone byte-identical:
# without it the zone stays a nested pjit call in the outer jaxpr, changing
# the emitted module around the `zorch.sumcheck` composite envelope (the
# operand layout XLA expands).


@partial(jit, static_argnames=("n",), inline=True)
def _sample_body(t: DuplexTranscript, n: int) -> tuple[DuplexTranscript, Array]:
    if n == 1:
        # One squeeze runs at most one permute already; the unrolled loop would
        # build a chain of one, so route straight through `_sample_one` (also the
        # exact path `_grind_search` and `check_witness` replay).
        t, x = t._sample_one()
        return t, fnp.stack([x.reshape(())])

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
    # miscompiles on the XLA CPU backend, the same reason `_observe_body`
    # unrolls its block loop. `depth` is static, so the
    # one-hot is a fixed chain of selects; `out_pos` stays a 1-D buffer gather,
    # which `_sample_one` already uses CPU-safely.
    def _pick(stacked: list, idx: Array) -> Array:
        acc = stacked[0]
        for i in range(1, len(stacked)):
            acc = fnp.where(idx == i, stacked[i], acc)
        return acc

    # Replay the per-limb schedule with traced scalars only (no field ops): track
    # how many permutes have fired (`perm_count`, the chain index) and the running
    # `out_pos`; a permute fires iff input is pending or the buffer is drained --
    # the same `need_perm` the per-limb loop tested.
    perm_count = fnp.int32(0)
    in_pos = st.in_pos
    out_pos = st.out_pos
    outs = []
    for _ in range(n):
        need_perm = (in_pos > 0) | (out_pos == 0)
        perm_count = fnp.where(need_perm, perm_count + 1, perm_count)
        in_pos = fnp.where(need_perm, fnp.int32(0), in_pos)
        out_pos = fnp.where(need_perm, fnp.int32(rate), out_pos)
        out_pos = out_pos - 1
        outs.append(_pick(output_buffers, perm_count)[out_pos].reshape(()))

    chain_state_leaves = [c.state for c in chain]
    final_state = tree_map(
        lambda *leaves: _pick(list(leaves), perm_count), *chain_state_leaves
    )
    final_state = replace(final_state, out_pos=out_pos)
    return t._with_state(final_state), fnp.stack(outs)


# A whole duplex-absorb chain (sequential rate-block permutes with a runtime
# live-count mask) as one marked region, so a vendor with a chain emitter runs
# it as a single kernel — the per-permute thunk/launch machinery of a scanned
# chain is the dominant cost of a long host absorb, not the permute math.
ABSORB_CHAIN_MARKER = "zorch.absorb_chain"
ABSORB_CHAIN_MARKER_VERSION = 1


# Module-level jit zone for the same reason as poseidon2's `_permute_body`:
# `lax.composite` re-traces its decomposition per emission, and one prove
# absorbs many identical-aval messages. `permutation` (value-hashable) and
# `rate` ride static.
@partial(jit, static_argnames=("permutation", "rate"), inline=True)
def _absorb_chain(
    permutation: Permutation,
    sponge: Array,
    blocks: Array,
    active_blocks: Array,
    rate: int,
) -> Array:
    """Absorb `blocks` (`(num_blocks, rate)`) into `sponge` (`(width,)`) as one
    `zorch.absorb_chain` region, generic over the permutation via its
    `fused_region_spec` (the `hash_frx.sponge_hash` pattern): the ABI operands and
    the `permutation`-discriminated attrs come from the spec, and the
    decomposition rebuilds the permute from those operands — the emitter's
    operand contract names the constants there. Blocks at
    index >= `active_blocks` (int32 scalar) are padding and leave the sponge
    unchanged. Caller gates on `fusion_path.is_one_kernel` and a concrete
    `num_blocks > 1` -- a chain of one is not a chain."""
    operands, permute_from_operands, perm_attrs = permutation.fused_region_spec(sponge)
    constants = operands[1:]
    perm_name, perm_version = permutation.fused_region_marker

    def chain(
        spg: Array, blocks: Array, active: Array, *consts: Array, **_attrs: object
    ) -> Array:
        # Each block permute is RE-MARKED with the permutation's own dedicated
        # marker, fed the chain composite's operand parameters — the per-permute
        # emitter ABI names the constants as operands, so they must thread
        # through rather than close over. So the chain's own fallback still runs
        # dedicated per-permute kernels, exactly as an unmarked absorb does. Keep
        # it that way: the dedicated kernel is the byte-authority the goldens pin, and a
        # raw permute body is not guaranteed to match it, so letting the raw body
        # serve as the fallback would silently change what a fallback absorbs.
        def inner(s: Array, *c: Array, **_a: object) -> Array:
            return permute_from_operands(s, *c)

        def _absorb(
            carry: tuple[Array, Array], block: Array
        ) -> tuple[tuple[Array, Array], None]:
            s, k = carry
            permuted = fused_region(
                inner,
                fnp.concatenate([block, s[rate:]]),
                *consts,
                name=perm_name,
                version=perm_version,
                **perm_attrs,
            )
            # Blocks past the live count are padding-only: sponge unchanged.
            return (fnp.where(k < active, permuted, s), k + fnp.int32(1)), None

        (s, _), _ = lax.scan(_absorb, (spg, fnp.int32(0)), blocks)
        return s

    return fused_region(
        chain,
        sponge,
        blocks,
        active_blocks,
        *constants,
        name=ABSORB_CHAIN_MARKER,
        version=ABSORB_CHAIN_MARKER_VERSION,
        rate=rate,
        **perm_attrs,
    )


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
    # every input and keeps only the rate-boundary one (`fnp.where(full, ...)`),
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
    length = in_pos + fnp.int32(m)
    active_blocks = length // rate  # runtime count of full rate-blocks
    num_blocks = (rate - 1 + m) // rate  # static upper bound on full blocks

    # Drop the unused gap `input_buffer[in_pos:rate]` from the stream: for stream
    # position `j`, the source index is `j` while `j < in_pos`, else shifted by
    # `rate - in_pos` to skip past the buffer's invalid suffix.
    combined_src = fnp.concatenate([st.input_buffer, flat])  # (rate + M,)
    total = (num_blocks + 1) * rate  # >= length, with a rate-block of tail slack
    pos = fnp.arange(total, dtype=fnp.int32)
    src_idx = pos + fnp.where(pos < in_pos, fnp.int32(0), rate - in_pos)
    src_idx = fnp.clip(src_idx, 0, combined_src.shape[0] - 1)
    combined = combined_src[src_idx]  # (total,) — valid prefix is [0:length]

    # Absorb the rate-blocks with a `lax.scan`: ONE permute body regardless of
    # message length, so the HLO module (and its compile) is O(1) in M. A Python
    # unroll instead emits `num_blocks` block-permute instructions + glue -- fine
    # for the tiny per-round observes, but a ~15 MB module / minutes of XLA passes
    # for a one-shot large observe like the shard preamble (num_blocks ~hundreds).
    # reuse_key dedups the permute cubin either way; it is the O(num_blocks) HLO
    # graph, not the cubin, that the roll collapses.
    #
    # Blocks ride as a leading-axis `xs` and the carry evolves by `concatenate` (no
    # in-place scatter, no traced dynamic_slice), which keeps the absorb
    # byte-identical on GPU and CPU. `k` rides the carry so the live-count mask
    # needs no `arange`, one path for concrete and symbolic `num_blocks` (export).
    blocks = combined[: num_blocks * rate].reshape(num_blocks, rate)

    # A dedicated-fusion permutation routes the whole absorb to the chain
    # marker (one vendor kernel), sidestepping the scanned chain's per-permute
    # thunk/launch machinery. Chain markers need a concrete block count, so the
    # symbolic path (export) keeps the plain scan.
    #
    # `num_blocks == 1` is NOT a chain -- there is no per-permute thunk chain to
    # collapse, so the marker buys nothing and costs: it wraps the single permute
    # in a composite whose whole point is sequencing. The scalar-witness absorb
    # under `grind`'s outer `vmap` is exactly this case (m == 1 -> num_blocks ==
    # 1), and wrapping it makes every witness lane carry a sequential-chain
    # composite. The plain scan below is the same masked permute, already fully
    # parallel across the `vmap`, and lowers to the same dedicated kernel.
    if (
        permutation.fusion_path.is_one_kernel
        and isinstance(num_blocks, int)
        and num_blocks > 1
    ):
        sponge = _absorb_chain(
            permutation, st.sponge_state, blocks, active_blocks, rate
        )
    else:

        def _absorb(
            carry: tuple[Array, Array], block: Array
        ) -> tuple[tuple[Array, Array], None]:
            sponge, k = carry
            permuted = permutation.permute(fnp.concatenate([block, sponge[rate:]]))
            # Blocks past the live count are padding-only: sponge unchanged.
            return (
                fnp.where(k < active_blocks, permuted, sponge),
                k + fnp.int32(1),
            ), None

        (sponge, _), _ = lax.scan(_absorb, (st.sponge_state, fnp.int32(0)), blocks)

    # The `length % rate` tail of the combined stream stays pending in the input
    # buffer (positions [0:in_pos_out]); higher slots are zero (overwrite mode
    # reads only [0:in_pos]). `tail_start` is the live tail's stream offset.
    tail_len = length - active_blocks * rate
    tail_start = active_blocks * rate
    tail = lax.dynamic_slice_in_dim(combined, tail_start, rate)
    slot = fnp.arange(rate, dtype=fnp.int32)
    in_buf = fnp.where(slot < tail_len, tail, fnp.zeros(rate, dtype=base_dtype))
    in_pos_out = tail_len

    # If the last full block permuted and no tail remains (in_pos_out == 0), the
    # post-permute sponge prefix is the fresh output; otherwise the next sample
    # permutes. Matches the per-element form's `last_was_perm` exactly.
    last_was_perm = in_pos_out == 0
    out_pos = fnp.where(last_was_perm, fnp.int32(rate), fnp.int32(0))
    output_buffer = fnp.where(
        last_was_perm, sponge[:rate], fnp.zeros(rate, dtype=base_dtype)
    )
    return t._with_state(
        DuplexState(in_buf, output_buffer, sponge, in_pos_out, out_pos)
    )


@partial(jit, static_argnames=("n",), inline=True)
def _observe_and_sample_body(
    t: DuplexTranscript, values: Array, n: int
) -> tuple[DuplexTranscript, Array]:
    return _sample_body(_observe_body(t, values), n)


# ============================================================================
# Device-FS Fiat-Shamir fusion marker (`zorch.duplex_fs`)
#
# One absorb+squeeze hop otherwise scatters ~9 GPU kernels: two `hash_frx.poseidon2`
# permute composites plus ~7 unfused loop/input fusions for the duplex buffer glue
# (rate-block merge, position select, output extraction). This marker wraps the
# whole hop so a vendor fuses it into one register-resident kernel -- fusion by
# construction, the CLAUDE.md non-negotiable.
#
# The decomposition is the plain `_observe_and_sample_body`. The duplex positions
# `(in_pos, out_pos)` are runtime OPERANDS (inside the threaded state) rather than
# compile-time attrs: they are scalars shared by every thread, so the
# permute-firing test (`in_pos == rate`, `out_pos == 0`) is a uniform branch --
# zero warp divergence -- and one kernel serves every round's phase. Only the
# message length and sample count `n` are static (the absorb/squeeze loop bounds),
# so the binary is recompile-free per shape, like the round-compute kernels.
# ============================================================================
DUPLEX_FS_MARKER = "zorch.duplex_fs"
DUPLEX_FS_MARKER_VERSION = 1


def _duplex_fs_region(
    in_buf: Array,
    out_buf: Array,
    sponge: Array,
    in_pos: Array,
    out_pos: Array,
    values: Array,
    *,
    perm: Permutation,
    rate: int,
    n: int,
    **_attrs: object,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """The `zorch.duplex_fs` decomposition: the plain device hop, entered at the
    runtime duplex positions carried in the state. `rate`/`width`/`n` ride as attrs
    (the emitter's static loop bounds); the positions stay operands so one kernel
    serves every round phase (`_attrs` swallows the pass-through `width`)."""
    state = DuplexState(in_buf, out_buf, sponge, in_pos, out_pos)
    advanced, r = _observe_and_sample_body(
        DuplexTranscript(perm, rate, state), values, n
    )
    return (*_state_leaves(advanced.state), r)


@partial(jit, static_argnames=("n",), inline=True)
def _duplex_fs_zone(
    t: DuplexTranscript, values: Array, n: int
) -> tuple[DuplexTranscript, Array]:
    """The marked hop as one compiled dispatch carrying the `zorch.duplex_fs`
    composite, so the eager host loop fires a single fused FS kernel (mirrors the
    module-level `_observe_body`/`_sample_body` jit zones). `inline=True` keeps a
    call site already inside an outer jit byte-identical."""
    ib, ob, sp, ip, op, r = fused_region(
        partial(_duplex_fs_region, perm=t.permutation),
        *_state_leaves(t.state),
        values,
        name=DUPLEX_FS_MARKER,
        version=DUPLEX_FS_MARKER_VERSION,
        rate=t.rate,
        width=t.permutation.width,
        n=n,
    )
    return t._with_state(DuplexState(ib, ob, sp, ip, op)), r


def _observe_and_sample_marked(
    t: DuplexTranscript, values: Array, n: int
) -> tuple[DuplexTranscript, Array]:
    """`_DeviceFs.observe_and_sample`'s body: the hop under a `zorch.duplex_fs`
    fusion marker, so a vendor fuses the ~9-kernel hop (two permutes + duplex glue)
    into one register-resident kernel. Only a dedicated-fusion permutation is
    marked, so a test `CheapPermutation` keeps the plain path.

    Private, and deliberately so: it is a *backend body*, the peer of
    `_observe_and_sample_body` and `_observe_and_sample_host` -- not an entry point.
    Callers say `transcript.observe_and_sample(...)` and get this one only when the
    transcript's backend is `_DeviceFs`. Calling it directly pins the hop to the
    device and silently ignores `fs_on_host=True`, which is exactly the
    "one method missed" failure `_FsBackend` exists to make impossible."""
    if not t.has_dedicated_fusion:
        return _observe_and_sample_body(t, values, n)
    return _duplex_fs_zone(t, values, n)


@partial(jit, static_argnames=("pow_bits",), inline=True)
def _check_witness_body(
    t: DuplexTranscript, witness: Array, *, pow_bits: int
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
# boundary costs per array-leaf (a `frx.pure_callback` round-trip of the 5 state
# leaves every hop was ~6x that and dominated a warm prove once the sponge math got
# fast); keeping the state resident drops each hop to one `values` in + one
# challenge out. This is an eager primitive -- it is the production jit=False
# relaunch's per-round FS, not a graph op.
#
# Byte-identical to the device sponge: the jit reconstructs a `DuplexTranscript`
# (fs_on_host defaults False -> no recursion) and runs the SAME `_observe_body`/
# `_sample_body`. The `permutation` must lower its `permute` to the host (a raw
# permute, not one pinning an inner accelerator jit). State leaves cross the host
# boundary as their own field dtype, which the FFI ABI carries directly
# (fractalyze/jax#45) -- no uint32 bitcast around them.
# ============================================================================


@cache
def _host_cpu() -> frx.Device:
    """The host device, resolved lazily so a host-FS transcript -- not merely
    importing -- is what requires a CPU backend."""
    return frx.devices("cpu")[0]


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
def _host_compute_device() -> frx.Device:
    """Where a squeezed challenge returns to -- the device the surrounding eager
    prove computes on (its kernels consume the challenge)."""
    return frx.devices()[0]


def _on_host(x: Array) -> bool:
    return next(iter(x.devices())).platform == "cpu"


def _state_on_host(state: DuplexState) -> DuplexState:
    """Commit the sponge state to the CPU. The first hop pays it once; the host
    sponge returns host leaves, so later hops find it already resident and the
    per-hop device<->host round-trip of the 5 state leaves disappears."""
    if _on_host(state.sponge_state):
        return state
    c = _host_cpu()
    return DuplexState(*(frx.device_put(leaf, c) for leaf in _state_leaves(state)))


def _state_on_device(state: DuplexState, device: frx.Device) -> DuplexState:
    """Move the sponge state to `device` — the inverse of `_state_on_host`, for
    a scoped host absorb that has to hand the stream back to the device path."""
    return DuplexState(*(frx.device_put(leaf, device) for leaf in _state_leaves(state)))


# ---------------------------------------------------------------------------
# Host FS from inside a compiled graph.
#
# The resident-state path above is eager by construction: it reads `.devices()`
# and calls `device_put`, and a tracer has neither, so a traced caller raises
# `ConcretizationTypeError`. That rules out every prover that traces its round
# loop — `logup_gkr.jagged_prover` runs its rounds inside `_jagged_round_zone`,
# so `fs_on_host=True` could not be selected there at all.
#
# `pure_callback` crosses to the host from inside the graph. The hop is genuinely
# pure once the sponge state is an explicit argument AND result, so it needs no
# ordering token: the data dependency through the threaded state is what forbids
# reordering, exactly as it does on the device path.
#
# What made the earlier callback attempt cost ~6x (see the section header) was
# crossing all five state leaves each way; the boundary is charged per array, not
# per byte. Packing the state into two arrays is what makes this affordable, so
# the hop crosses three each way (state, positions, values) instead of six.
# ---------------------------------------------------------------------------


def _to_wire(x: Array) -> Array:
    """Bitcast a field array to `uint32` for the callback boundary.

    `pure_callback` materializes its operands as numpy, and the runtime cannot
    build a numpy dtype for a parametric algebraic width ("ALGEBRAIC32 is a
    parametric width class..."), so a field-typed array cannot cross it. The field
    is 32-bit backed, so the bitcast is free and exact; `_from_wire` undoes it on
    the far side. (The FFI ABI carries field dtypes directly -- that is a different
    path from this one.)"""
    return x.view(fnp.uint32)


def _from_wire(x: Array, dtype: Any) -> Array:
    return x.view(dtype)


_host_fs_ffi_target: str | None = None


def set_host_fs_ffi_target(name: str | None) -> None:
    """Point traced host FS at an XLA FFI target instead of the `pure_callback`
    transport, or `None` to go back to the callback.

    A callback crossing costs ~101us/hop of pure transport before the sponge runs
    at all, which is ~18x a fused device hop -- it makes host FS correct but never
    competitive. An FFI custom call stays inside the graph and lands in the same
    range as the device hop, so it is the only transport worth running in
    production.

    zorch ships no such target and names no vendor: the consumer registers its own
    handler (`frx.ffi.register_ffi_target`) and passes the name here. The contract
    is six `uint32` operands -- `in_buf(8)`, `out_buf(8)`, `sponge(16)`,
    `in_pos(1)`, `out_pos(1)`, `values(m)` -- returning those five advanced plus an
    `n`-word challenge. State crosses in, and back out, on every call: the handler
    must hold no sponge of its own, or a replayed or reordered region silently
    forks the transcript.

    Set it once, at startup, before anything traces. It is read while tracing, so
    a zone compiled earlier keeps whichever transport was set then -- change it
    afterwards and the old one stays live with nothing to say so. Process-wide
    rather than per-transcript for the same reason a handler is: one vendor
    registers one handler for the process."""
    global _host_fs_ffi_target
    _host_fs_ffi_target = name


def _wire_leaves(state: DuplexState) -> tuple[Array, ...]:
    """The five state leaves as `uint32`, shaped as the FFI contract wants (the
    positions as 1-element buffers, not scalars).

    Deliberately NOT packed into one buffer. Packing looks cheaper -- 4 operands
    instead of 12 -- but a `concatenate` in and five `slice`s out are real data
    ops the surrounding fusion cannot absorb, and they measured as ~7 extra
    kernels per hop in the layer's compiled module. Views are bitcasts and cost
    nothing; the leaves are already the shapes the handler wants."""
    return (
        _to_wire(state.input_buffer),
        _to_wire(state.output_buffer),
        _to_wire(state.sponge_state),
        state.in_pos.reshape(1).view(fnp.uint32),
        state.out_pos.reshape(1).view(fnp.uint32),
    )


def _host_hop_ffi(
    transcript: DuplexTranscript, values: Array | None, n: int
) -> tuple[DuplexTranscript, Array]:
    """`observe_and_sample` on the host through the registered FFI target.

    Declared PURE (`has_side_effect=False`), which is worth ~0.8-1.5ms a prove
    because the effect token orders every hop against every other one and costs
    more than the crossing it guards. The hop earns it: the sponge state is an
    explicit operand AND result, so the state chain already forbids reordering;
    consecutive hops differ in that state, so CSE cannot merge them; the challenge
    is consumed, so DCE cannot drop a hop anyone uses; and the handler carries its
    whole sponge in and out, so even a rematerialized duplicate call recomputes the
    same answer from the same state. A hop whose results are all discarded IS
    dead, and dropping it is correct -- no Fiat-Shamir was observed.

    This reasoning is specific to the marker-shaped handler. A handler keeping a
    resident sponge would be genuinely effectful and must not be called here."""
    field = transcript.field
    rate, width = transcript.rate, transcript.permutation.width
    u32 = fnp.uint32
    out_types = (
        frx.ShapeDtypeStruct((rate,), u32),
        frx.ShapeDtypeStruct((rate,), u32),
        frx.ShapeDtypeStruct((width,), u32),
        frx.ShapeDtypeStruct((1,), u32),
        frx.ShapeDtypeStruct((1,), u32),
        frx.ShapeDtypeStruct((n,), u32),
    )
    # A bare `sample` is this hop with nothing to absorb, and a bare `observe` is
    # it with nothing to squeeze. Expressing all three through the one target
    # matters more than it looks: leaving either on the `pure_callback` fallback
    # puts ~518us hops back in the stream, and a stage's dozen of them cost more
    # than all 88 fused hops put together.
    wire_values = fnp.zeros(0, fnp.uint32) if values is None else _to_wire(values)
    # Aliasing the state buffers is SAFE here (the handler finishes every read
    # before it writes) but measured slower -- 8.4 -> 9.3 ms min -- so the hop
    # takes the extra buffers rather than constraining XLA's assignment.
    ib, ob, sp, ip, op, chal = frx.ffi.ffi_call(
        _host_fs_ffi_target, out_types, has_side_effect=False
    )(*_wire_leaves(transcript.state), wire_values)
    return (
        transcript._with_state(
            DuplexState(
                input_buffer=_from_wire(ib, field),
                output_buffer=_from_wire(ob, field),
                sponge_state=_from_wire(sp, field),
                in_pos=ip.view(fnp.int32).reshape(()),
                out_pos=op.view(fnp.int32).reshape(()),
            )
        ),
        _from_wire(chal, field),
    )


@cache
def _host_hop_ffi_eager(n: int, absorbs: bool) -> Any:
    """An eager hop that still crosses via the FFI, by wrapping it in `jit`.

    An eager host hop is otherwise the most expensive thing in a host-FS prove:
    `_state_on_host` ships five leaves to the CPU, the sponge runs, and
    `_state_on_device` ships them back, so a hop costs ~713us of `device_put`
    round trips against 0.43us of hashing. Only three of them run per jagged
    prove -- FS at the stage boundaries, outside any zone -- but at that price
    those three WERE the whole host-vs-device gap, while the ~98 hops inside the
    zones are free.

    Wrapping the FFI hop in `jit` makes it traced, so the handler does the
    crossing and the sponge state never leaves the device."""
    if absorbs:
        return jit(lambda t, v: _host_hop_ffi(t, v, n))
    return jit(lambda t: _host_hop_ffi(t, None, n))


def _is_traced(*xs: Any) -> bool:
    """Whether we are inside a compiled graph, which is what decides between the
    eager resident-state hop and the `pure_callback` one. Both run the same CPU
    sponge jit, so only the transport differs -- the math cannot drift."""
    return any(isinstance(x, frx.core.Tracer) for x in xs)


def _pack_state(state: DuplexState) -> tuple[Array, Array]:
    """The five state leaves as two arrays — the field-typed buffers run together,
    the two positions apart because they are `int32`. Halving the array count
    halves the per-hop boundary cost, which is what the five-leaf version paid."""
    return (
        fnp.concatenate((state.input_buffer, state.output_buffer, state.sponge_state)),
        fnp.stack((state.in_pos, state.out_pos)),
    )


def _unpack_state(words: Array, pos: Array, rate: int, width: int) -> DuplexState:
    """Inverse of `_pack_state`; the slice bounds are the one place the packed
    layout is written down, so a producer and consumer cannot disagree about it."""
    return DuplexState(
        input_buffer=words[:rate],
        output_buffer=words[rate : 2 * rate],
        sponge_state=words[2 * rate : 2 * rate + width],
        in_pos=pos[0],
        out_pos=pos[1],
    )


def _host_hop_traced(
    transcript: DuplexTranscript, values: Array, n: int
) -> tuple[DuplexTranscript, Array]:
    """`observe_and_sample` on the host sponge from inside a traced region.

    The callback runs the SAME `_host_obs_sample_jit` the eager path runs, on
    CPU-committed inputs, so the two transports stay byte-identical by sharing the
    sponge rather than by agreeing to. Committing to the CPU inside the callback is
    also what keeps it safe to call: the surrounding graph is waiting on the
    accelerator, and re-entering XLA on the *same* device from a callback is what
    deadlocks."""
    perm = _host_raw(transcript.permutation)
    rate, width = transcript.rate, transcript.permutation.width
    field = transcript.field
    words, pos = _pack_state(transcript.state)
    sponge = _host_obs_sample_jit(perm, rate, n)

    def run(words_h: Any, pos_h: Any, values_h: Any) -> tuple[Any, Any, Any]:
        cpu = _host_cpu()
        state = _unpack_state(
            _from_wire(frx.device_put(words_h, cpu), field),
            frx.device_put(pos_h, cpu),
            rate,
            width,
        )
        advanced, out = sponge(state, _from_wire(frx.device_put(values_h, cpu), field))
        new_words, new_pos = _pack_state(advanced)
        return (
            np.asarray(_to_wire(new_words)),
            np.asarray(new_pos),
            np.asarray(_to_wire(out)),
        )

    shapes = (
        frx.ShapeDtypeStruct(words.shape, fnp.uint32),
        frx.ShapeDtypeStruct(pos.shape, pos.dtype),
        frx.ShapeDtypeStruct((n,), fnp.uint32),
    )
    new_words, new_pos, out = frx.pure_callback(
        run, shapes, _to_wire(words), pos, _to_wire(values)
    )
    return (
        transcript._with_state(
            _unpack_state(_from_wire(new_words, field), new_pos, rate, width)
        ),
        _from_wire(out, field),
    )


def _host_observe_traced(
    transcript: DuplexTranscript, values: Array
) -> DuplexTranscript:
    """`observe` on the host sponge from inside a traced region — `_host_hop_traced`
    without the squeeze, so it returns state alone."""
    perm = _host_raw(transcript.permutation)
    rate, width = transcript.rate, transcript.permutation.width
    field = transcript.field
    words, pos = _pack_state(transcript.state)
    sponge = _host_observe_jit(perm, rate)

    def run(words_h: Any, pos_h: Any, values_h: Any) -> tuple[Any, Any]:
        cpu = _host_cpu()
        state = _unpack_state(
            _from_wire(frx.device_put(words_h, cpu), field),
            frx.device_put(pos_h, cpu),
            rate,
            width,
        )
        new_words, new_pos = _pack_state(
            sponge(state, _from_wire(frx.device_put(values_h, cpu), field))
        )
        return np.asarray(_to_wire(new_words)), np.asarray(new_pos)

    shapes = (
        frx.ShapeDtypeStruct(words.shape, fnp.uint32),
        frx.ShapeDtypeStruct(pos.shape, pos.dtype),
    )
    new_words, new_pos = frx.pure_callback(
        run, shapes, _to_wire(words), pos, _to_wire(values)
    )
    return transcript._with_state(
        _unpack_state(_from_wire(new_words, field), new_pos, rate, width)
    )


def _host_sample_traced(
    transcript: DuplexTranscript, n: int
) -> tuple[DuplexTranscript, Array]:
    """`sample` on the host sponge from inside a traced region — `_host_hop_traced`
    without the absorb, so it takes no `values`."""
    perm = _host_raw(transcript.permutation)
    rate, width = transcript.rate, transcript.permutation.width
    field = transcript.field
    words, pos = _pack_state(transcript.state)
    sponge = _host_sample_jit(perm, rate, n)

    def run(words_h: Any, pos_h: Any) -> tuple[Any, Any, Any]:
        cpu = _host_cpu()
        state = _unpack_state(
            _from_wire(frx.device_put(words_h, cpu), field),
            frx.device_put(pos_h, cpu),
            rate,
            width,
        )
        advanced, out = sponge(state)
        new_words, new_pos = _pack_state(advanced)
        return (
            np.asarray(_to_wire(new_words)),
            np.asarray(new_pos),
            np.asarray(_to_wire(out)),
        )

    shapes = (
        frx.ShapeDtypeStruct(words.shape, fnp.uint32),
        frx.ShapeDtypeStruct(pos.shape, pos.dtype),
        frx.ShapeDtypeStruct((n,), fnp.uint32),
    )
    new_words, new_pos, out = frx.pure_callback(run, shapes, _to_wire(words), pos)
    return (
        transcript._with_state(
            _unpack_state(_from_wire(new_words, field), new_pos, rate, width)
        ),
        _from_wire(out, field),
    )


def _observe_host(transcript: DuplexTranscript, values: Array) -> DuplexTranscript:
    """`observe` on the host sponge; the state stays host-resident."""
    if _is_traced(values, *_state_leaves(transcript.state)):
        if _host_fs_ffi_target is not None:
            return _host_hop_ffi(transcript, values, 0)[0]
        return _host_observe_traced(transcript, values)
    if _host_fs_ffi_target is not None:
        return _host_hop_ffi_eager(0, True)(transcript, values)[0]
    s = _state_on_host(transcript.state)
    f = _host_observe_jit(_host_raw(transcript.permutation), transcript.rate)
    return transcript._with_state(f(s, frx.device_put(values, _host_cpu())))


def _sample_host(
    transcript: DuplexTranscript, n: int = 1
) -> tuple[DuplexTranscript, Array]:
    """`sample` n raw squeezes on the host sponge; the state stays host-resident,
    the challenge returns to the compute device."""
    if _is_traced(*_state_leaves(transcript.state)):
        if _host_fs_ffi_target is not None:
            return _host_hop_ffi(transcript, None, n)
        return _host_sample_traced(transcript, n)
    if _host_fs_ffi_target is not None:
        return _host_hop_ffi_eager(n, False)(transcript)
    s = _state_on_host(transcript.state)
    f = _host_sample_jit(_host_raw(transcript.permutation), transcript.rate, n)
    state, out = f(s)
    return (
        transcript._with_state(state),
        frx.device_put(out, _host_compute_device()),
    )


def _observe_and_sample_host(
    transcript: DuplexTranscript, values: Array, n: int = 1
) -> tuple[DuplexTranscript, Array]:
    """`observe_and_sample`: absorb then squeeze n raw in one host hop -- the
    per-round Fiat-Shamir primitive. The challenge returns to the device `values`
    came from -- the accelerator the round polys are produced on -- so a
    multi-device prove gets it back where its kernels consume it (the sponge state
    is CPU-resident in steady state, so its device can't name the compute one)."""
    if _is_traced(values, *_state_leaves(transcript.state)):
        if _host_fs_ffi_target is not None:
            return _host_hop_ffi(transcript, values, n)
        return _host_hop_traced(transcript, values, n)
    if _host_fs_ffi_target is not None:
        return _host_hop_ffi_eager(n, True)(transcript, values)
    compute_device = next(iter(values.devices()))
    s = _state_on_host(transcript.state)
    f = _host_obs_sample_jit(_host_raw(transcript.permutation), transcript.rate, n)
    state, out = f(s, frx.device_put(values, _host_cpu()))
    return (
        transcript._with_state(state),
        frx.device_put(out, compute_device),
    )


def _check_witness_host(
    transcript: DuplexTranscript, witness: Array, *, pow_bits: int
) -> tuple[DuplexTranscript, Array]:
    """`check_witness` on the host sponge — observe + one squeeze + the pow check,
    the host counterpart of `_check_witness_body`. Routing it through the backend
    (not the device body unconditionally) keeps a host-FS grind and its re-check on
    the same sponge, so the two can't disagree."""
    advanced, sample = _observe_and_sample_host(transcript, witness, 1)
    return advanced, _pow_satisfied(sample[0], pow_bits)


if TYPE_CHECKING:
    _: type[Transcript] = DuplexTranscript
