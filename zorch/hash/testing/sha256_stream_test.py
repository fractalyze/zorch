# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The fixed-shape streaming SHA-256 midstate (`Sha256State`) — byte-exact vs the
universal reference `hashlib.sha256`, named by no consumer.

`digest` pads a whole message once on host; this incremental core keeps the
Merkle–Damgård chaining value as a fixed-shape pytree so a byte Fiat-Shamir
transcript threads `@jit` / a `lax.scan` carry (`Sha256FieldTranscript`).
"""
from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from zorch.hash.sha256 import (
    Sha256State,
    sha256_stream_absorb,
    sha256_stream_finalize,
    sha256_stream_init,
)


def _u8(data: bytes) -> jnp.ndarray:
    return jnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _stream_absorb_all(chunks: list[bytes]) -> Sha256State:
    state = sha256_stream_init()
    for c in chunks:
        state = sha256_stream_absorb(state, _u8(c))
    return state


class Sha256StreamTest(absltest.TestCase):
    def test_stream_matches_hashlib_across_lengths(self) -> None:
        # Lengths straddling the 55/56/64 finalization + block boundaries, each
        # finalized with no extra (SHA256 of the buffer) and with an 8-byte extra
        # (the transcript's counter-mode append).
        for n in (0, 1, 31, 32, 55, 56, 63, 64, 65, 100, 127, 128, 191, 200):
            msg = bytes((i * 7 + 3) & 0xFF for i in range(n))
            for extra in (b"", b"\x00\x11\x22\x33\x44\x55\x66\x77"):
                state = _stream_absorb_all([msg])
                got = bytes(
                    np.asarray(
                        sha256_stream_finalize(state, _u8(extra).reshape(1, -1))[0]
                    )
                )
                self.assertEqual(
                    got, hashlib.sha256(msg + extra).digest(), f"n={n} extra={extra!r}"
                )

    def test_stream_matches_hashlib_across_splits(self) -> None:
        # The same message absorbed in different chunk splits must hash identically
        # — pins the pending-block carry across absorb calls.
        msg = bytes((i * 13 + 1) & 0xFF for i in range(150))
        ref = hashlib.sha256(msg).digest()
        for split in ([150], [64, 86], [1, 63, 86], [50, 50, 50], [63, 1, 63, 23]):
            chunks, off = [], 0
            for s in split:
                chunks.append(msg[off : off + s])
                off += s
            state = _stream_absorb_all(chunks)
            got = bytes(
                np.asarray(sha256_stream_finalize(state, _u8(b"").reshape(1, 0))[0])
            )
            self.assertEqual(got, ref, f"split={split}")

    def test_stream_counter_mode_batch(self) -> None:
        # One finalize over a batch of 8-byte counters == per-counter hashlib. This
        # is exactly the transcript's `SHA256(buffer ‖ ctr_le8)` squeeze.
        msg = b"transcript-buffer-bytes"
        state = _stream_absorb_all([msg])
        counters = np.stack(
            [
                np.frombuffer(int(c).to_bytes(8, "little"), dtype=np.uint8)
                for c in range(5)
            ]
        )
        digs = np.asarray(sha256_stream_finalize(state, jnp.asarray(counters)))
        for c in range(5):
            ref = hashlib.sha256(msg + int(c).to_bytes(8, "little")).digest()
            self.assertEqual(bytes(digs[c]), ref, f"ctr={c}")

    def test_stream_threads_under_jit(self) -> None:
        # The whole point: absorb + finalize are pure JAX on a fixed-shape pytree,
        # so they run under @jit unchanged (a `lax.scan` carry is the same contract).
        msg = bytes(range(70))
        extra = b"\x01\x02\x03\x04\x05\x06\x07\x08"

        @jax.jit
        def run(data: jnp.ndarray, ex: jnp.ndarray) -> jnp.ndarray:
            state = sha256_stream_absorb(sha256_stream_init(), data)
            return sha256_stream_finalize(state, ex.reshape(1, -1))

        got = bytes(np.asarray(run(_u8(msg), _u8(extra)))[0])
        self.assertEqual(got, hashlib.sha256(msg + extra).digest())

    def test_stream_threads_through_scan(self) -> None:
        # The design claim `test_stream_threads_under_jit` alludes to: `Sha256State`'s
        # fixed shapes make it a valid `lax.scan` carry. Fold equal-size chunks
        # through a scan and check the finalized digest still matches hashlib.
        msg = bytes(range(96))  # 6 chunks of 16 -> one full block + a 32 B remainder
        chunks = jnp.asarray(np.frombuffer(msg, np.uint8)).reshape(6, 16)

        @jax.jit
        def run(xs: jnp.ndarray) -> jnp.ndarray:
            def step(
                state: Sha256State, chunk: jnp.ndarray
            ) -> tuple[Sha256State, None]:
                return sha256_stream_absorb(state, chunk), None

            state, _ = jax.lax.scan(step, sha256_stream_init(), xs)
            return sha256_stream_finalize(state, jnp.zeros((1, 0), dtype=jnp.uint8))

        got = bytes(np.asarray(run(chunks))[0])
        self.assertEqual(got, hashlib.sha256(msg).digest())


if __name__ == "__main__":
    absltest.main()
