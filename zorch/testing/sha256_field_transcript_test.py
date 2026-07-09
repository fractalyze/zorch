# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The field-element `Transcript` surface (`Sha256FieldTranscript`) over the
streaming SHA-256 core.

The byte transcript is the established oracle (`byte_transcript_test` pins the
framing vs `hashlib`; flock-zorch's `challenger_test` pins it to flock-core's
`FsChallenger`). This slice proves the FIELD surface reproduces the byte
transcript's slice framing exactly — the field surface is the byte surface, made
scan-threadable — and threads zorch's sumcheck round driver under `@jit`.
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
        # observe_scalar/sample_scalar use the byte transcript's scalar framing
        # (KIND_SCALAR, no length prefix), so the squeezed bytes match — and must
        # DIFFER from the slice framing of the same single element.
        v = np.uint32(0xDEADBEEF)

        b = ByteHashTranscript.new(b"dom", Sha256()).observe_scalar(v.tobytes())
        b, b_sq = b.sample_scalar(4)  # itemsize bytes

        f = Sha256FieldTranscript.new(b"dom", np.uint32)
        f, f_el = f.observe_scalar(jnp.asarray(v)).sample_scalar()
        self.assertEqual(f_el.shape, ())  # scalar squeeze is 0-D
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

        # Same element, but sampled/observed under SLICE framing — must diverge.
        g = Sha256FieldTranscript.new(b"dom", np.uint32)
        g, g_sl = g.observe(jnp.asarray(v).reshape(1)).sample(1)
        self.assertNotEqual(np.asarray(f_el).tobytes(), np.asarray(g_sl).tobytes())

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
        # The acceptance-critical path: the transcript threads the sumcheck round
        # driver (fold_rounds over StandardRound) under jit, no host callback. A
        # uint32 ring stands in for a scalar challenge field (flock's F128 rides
        # the GHASH dtype seam, flock-zorch#9); the point here is the transcript
        # threading, not the sumcheck math.
        from zorch.prove import fold_rounds
        from zorch.sumcheck.prover import ProductSummand, StandardRound

        a = jnp.arange(8, dtype=jnp.uint32) + 1
        b = jnp.arange(8, dtype=jnp.uint32) + 2
        rnd = StandardRound(ProductSummand(degree=2))
        tr = Sha256FieldTranscript.new(b"sc", np.uint32)

        def run(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            # 3 rounds folds the 2^3 stacked factors down to width 1.
            state, _, msgs = fold_rounds(rnd, jnp.stack([x, y]), tr, 3)
            return state[:, 0], jnp.stack(msgs)

        eager = jax.tree_util.tree_map(np.asarray, run(a, b))
        jitted = jax.tree_util.tree_map(np.asarray, jax.jit(run)(a, b))
        self.assertEqual(eager[0].tobytes(), jitted[0].tobytes())
        self.assertEqual(eager[1].tobytes(), jitted[1].tobytes())


if __name__ == "__main__":
    absltest.main()
