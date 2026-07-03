# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The field-element `Transcript` surface (`Sha256FieldTranscript`) over the
streaming SHA-256 core.

The byte transcript is the established oracle (`byte_transcript_test` pins the
framing vs `hashlib`; flock-zorch's `challenger_test` pins it to flock-core's
`FsChallenger`). This slice proves the FIELD surface reproduces the byte
transcript's slice framing exactly — the field surface is the byte surface, made
scan-threadable — and threads `zorch.sumcheck.prove` under `@jit`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from zorch.byte_transcript import ByteHashTranscript
from zorch.hash.sha256 import Sha256
from zorch.sha256_field_transcript import Sha256FieldTranscript


class Sha256FieldTranscriptTest(absltest.TestCase):
    def test_slice_framing_matches_byte_transcript(self) -> None:
        # observe(Array)/sample(n) use the same slice framing as the byte
        # transcript's observe_slice/sample_slice, so the squeezed challenge bytes
        # match — the field surface is the byte surface, made scan-threadable.
        vals = np.array([1, 2, 3, 4, 0xDEADBEEF, 5], dtype=np.uint32)
        vbytes = vals.astype("<u4").tobytes()

        b = ByteHashTranscript.new(b"dom", Sha256()).observe_slice(vbytes, vals.size)
        b, b_sq = b.sample_slice(3, 4)  # 3 elements * 4 bytes

        f = Sha256FieldTranscript.new(b"dom", np.uint32)
        f = f.observe(jnp.asarray(vals))
        f, f_el = f.sample(3)
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

        # A second squeeze pins the re-absorb of the first (both slice-framed).
        b, b2 = b.sample_slice(4, 4)
        f, f2 = f.sample(4)
        self.assertEqual(np.asarray(f2).astype("<u4").tobytes(), b2)

    def test_scalar_framing_matches_byte_transcript(self) -> None:
        # observe_scalar/sample_scalar use KIND_SCALAR framing (no count prefix),
        # byte-identical to the byte transcript's observe_scalar/sample_scalar —
        # the per-element framing a byte challenger's F128 observe/sample uses
        # (flock-zorch#9), where the count-prefixed slice path would not match.
        v = np.uint32(0xDEADBEEF)
        vbytes = v.astype("<u4").tobytes()

        b = ByteHashTranscript.new(b"dom", Sha256()).observe_scalar(vbytes)
        b, b_sq = b.sample_scalar(4)

        f = Sha256FieldTranscript.new(b"dom", np.uint32)
        f = f.observe_scalar(jnp.asarray(v))
        f, f_el = f.sample_scalar()
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

        # A second scalar squeeze pins the re-absorb of the first.
        b, b2 = b.sample_scalar(4)
        f, f2 = f.sample_scalar()
        self.assertEqual(np.asarray(f2).astype("<u4").tobytes(), b2)

    def test_threads_under_jit(self) -> None:
        vals = np.arange(6, dtype=np.uint32)

        def run(x: jnp.ndarray) -> jnp.ndarray:
            f = Sha256FieldTranscript.new(b"dom", np.uint32)
            _, r = f.observe_and_sample(x, 1)
            return r

        eager = np.asarray(run(jnp.asarray(vals)))
        jitted = np.asarray(jax.jit(run)(jnp.asarray(vals)))
        self.assertEqual(eager.tobytes(), jitted.tobytes())

    def test_threads_through_sumcheck_prove(self) -> None:
        # The acceptance-critical path: the transcript threads zorch.sumcheck.prove
        # as a lax.scan carry under jit, no host callback. A uint32 ring stands in
        # for a scalar challenge field (flock's F128 needs the GHASH FieldOps seam,
        # flock-zorch#9); the point here is the transcript threading, not the math.
        from zorch.sumcheck.prover import SumcheckRound, prove

        s0 = jnp.arange(8, dtype=jnp.uint32) + 1
        s1 = jnp.arange(8, dtype=jnp.uint32) + 2
        tr = Sha256FieldTranscript.new(b"sc", np.uint32)

        def run(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            folded, _, msgs = prove(SumcheckRound(degree=2), [a, b], tr)
            return folded[0], msgs.round_poly

        eager = jax.tree_util.tree_map(np.asarray, run(s0, s1))
        jitted = jax.tree_util.tree_map(np.asarray, jax.jit(run)(s0, s1))
        self.assertEqual(eager[0].tobytes(), jitted[0].tobytes())
        self.assertEqual(eager[1].tobytes(), jitted[1].tobytes())


if __name__ == "__main__":
    absltest.main()
