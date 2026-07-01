# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`DeviceSha256Transcript` — byte-identical to the host `Sha256Transcript`.

The host transcript is the established oracle: `byte_transcript_test` pins it to an
independent `hashlib` reference, and flock-zorch's `challenger_test` pins it to
flock-core's `FsChallenger`. This slice proves the DEVICE transcript — the same
Merlin-over-SHA-256 framing, but SHA-256 via the name-routed `zorch.sha256` marker
(`zorch.hash.sha256.digest`) instead of host `hashlib` — reproduces the host's
absorbed-byte stream and squeezed challenges exactly (fractalyze/flock-zorch#6).
"""
from __future__ import annotations

import hashlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from zorch.byte_transcript import Sha256Transcript
from zorch.device_byte_transcript import (
    DeviceSha256Transcript,
    Sha256State,
    sha256_stream_absorb,
    sha256_stream_finalize,
    sha256_stream_init,
)


def _run(cls: Any) -> tuple[bytes, bytes, bytes]:
    """A mixed op sequence exercising every framing branch; returns the final
    absorbed buffer plus each squeeze so a caller can compare host vs device."""
    t = cls.new(b"flock-domain")
    t = t.observe_label(b"phase-A")
    t = t.observe_bytes(b"\x00\x11\x22\x33")
    t = t.observe_scalar(b"sixteen--bytes!!")
    t = t.observe_slice(b"AABBBBCCCCCCDDDD", 4)
    t, s0 = t.sample_scalar(16)
    t, s1 = t.sample_slice(3, 16)  # 48 bytes → spans 2 SHA-256 output blocks
    return t.buffer, s0, s1


class DeviceByteTranscriptTest(absltest.TestCase):
    def test_matches_host_transcript(self) -> None:
        h_buf, h_s0, h_s1 = _run(Sha256Transcript)
        d_buf, d_s0, d_s1 = _run(DeviceSha256Transcript)
        self.assertEqual(d_s0, h_s0)
        self.assertEqual(d_s1, h_s1)
        self.assertEqual(d_buf, h_buf)

    def test_multiblock_squeeze_matches(self) -> None:
        # A squeeze spanning many SHA-256 output blocks (counter mode, 160 B = 5).
        h = Sha256Transcript.new(b"d").sample_slice(10, 16)[1]
        d = DeviceSha256Transcript.new(b"d").sample_slice(10, 16)[1]
        self.assertEqual(d, h)

    def test_domain_separation_matches(self) -> None:
        for dom in (b"dom-a", b"dom-b", b""):
            h = Sha256Transcript.new(dom).sample_scalar(16)[1]
            d = DeviceSha256Transcript.new(dom).sample_scalar(16)[1]
            self.assertEqual(d, h)

    def test_reabsorb_advances_like_host(self) -> None:
        # sample-then-observe must diverge from the no-sample path, identically on
        # both — pins the re-absorb of squeezed bytes.
        h = Sha256Transcript.new(b"d").sample_scalar(16)[0].observe_scalar(b"Z" * 16)
        d = (
            DeviceSha256Transcript.new(b"d")
            .sample_scalar(16)[0]
            .observe_scalar(b"Z" * 16)
        )
        self.assertEqual(d.sample_scalar(16)[1], h.sample_scalar(16)[1])


def _u8(data: bytes) -> jnp.ndarray:
    return jnp.asarray(np.frombuffer(data, dtype=np.uint8))


def _stream_absorb_all(chunks: list[bytes]) -> Sha256State:
    state = sha256_stream_init()
    for c in chunks:
        state = sha256_stream_absorb(state, _u8(c))
    return state


class Sha256StreamTest(absltest.TestCase):
    """The fixed-shape streaming Merkle–Damgård midstate — byte-exact vs hashlib."""

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


if __name__ == "__main__":
    absltest.main()
