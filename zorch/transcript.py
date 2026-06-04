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
from typing import Any, Protocol, Self

import jax.numpy as jnp
from jax import Array, lax
from jax.tree_util import register_dataclass

from zorch.hash.permutation import Permutation


class Transcript(Protocol):
    @property
    def has_dedicated_fusion(self) -> bool: ...
    def observe(self, values: Array) -> Self: ...
    def sample(self, n: int = 1) -> tuple[Self, Array]: ...
    def observe_and_sample(self, values: Array, n: int = 1) -> tuple[Self, Array]: ...


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
        base_dtype = self.state.sponge_state.dtype
        flat = lax.bitcast_convert_type(values, base_dtype).reshape(-1)
        if flat.shape[0] == 0:
            return self

        rate = self.rate
        permutation = self.permutation

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

        init = (self.state.input_buffer, self.state.in_pos, self.state.sponge_state)
        (in_buf, in_pos, sponge), _ = lax.scan(step, init, flat)

        # If the final scan step permuted (in_pos == 0 at exit), the post-permute
        # sponge prefix is the fresh output; otherwise the next sample permutes.
        last_was_perm = in_pos == 0
        out_pos = jnp.where(last_was_perm, jnp.int32(rate), jnp.int32(0))
        output_buffer = jnp.where(
            last_was_perm, sponge[:rate], jnp.zeros(rate, dtype=base_dtype)
        )
        return self._with_state(
            DuplexState(in_buf, output_buffer, sponge, in_pos, out_pos)
        )

    def _sample_one(self) -> tuple[DuplexTranscript, Array]:
        # Permute when input is pending or the output buffer is drained.
        need_perm = (self.state.in_pos > 0) | (self.state.out_pos == 0)
        t = lax.cond(need_perm, lambda c: c._duplexing(), lambda c: c, self)
        out_pos = t.state.out_pos - 1
        item = t.state.output_buffer[out_pos]
        return t._with_state(replace(t.state, out_pos=out_pos)), item

    def sample(self, n: int = 1) -> tuple[DuplexTranscript, Array]:
        t = self
        outs = []
        for _ in range(n):
            t, x = t._sample_one()
            outs.append(x.reshape(()))
        return t, jnp.stack(outs)

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[DuplexTranscript, Array]:
        """Absorb `values`, then squeeze `n` challenges — the per-round
        Fiat-Shamir primitive (commit -> challenge). One method so the absorb and
        squeeze fuse into a single kernel under `@jit` by construction, never by a
        per-primitive pattern-match (the repo's fusion contract)."""
        return self.observe(values).sample(n)
