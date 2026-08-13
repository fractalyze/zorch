# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The field-element `Transcript` surface (`Blake3FieldTranscript`) over the
streaming BLAKE3 core.

Same gate as the SHA-256 row's: the device transcript reproduces the Merlin byte
framing exactly, pinned against a host oracle assembled from other parts — the
framing from `ByteHashTranscript`, the bytes from hash-frx's host BLAKE3 row.
Fiat-Shamir makes that sharp for free, since every draw binds the whole prefix,
so one wrong byte anywhere surfaces at the next sample and never cancels out.

Where the independence stops, so a later reader does not over-read these gates:
`_len8` is imported by both sides and is therefore pinned by neither. The PoW
predicate is two implementations rather than one — the oracle takes
`byte_transcript`'s numpy `_leading_zero_bits_ok`, the transcript takes
`grind.leading_zero_bits_ok` — but they are required to agree bit for bit, so
this file compares twins rather than validating either independently.

The two BLAKE3-specific behaviours get their own gates, because neither is
implied by the framing: the squeeze is an XOF read (so it must DIFFER from the
counter chain the same buffer and hash would produce), and the proof-of-work
pre-image width is a wire parameter (so the two widths must find different
nonces).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from hash_frx.blake3 import blake3
from hash_frx.blake3.blake3 import BLAKE3_MARKER
from hash_frx.blake3.byte_hashes import HostBlake3

from zorch.blake3_field_transcript import Blake3FieldTranscript
from zorch.byte_transcript import (
    ByteHashTranscript,
    _leading_zero_bits_ok,
    _len8,
)

_DIGEST_BYTES = 32
_NO_PADDING = _DIGEST_BYTES + 8
# The width at which `_pow_digests` stops taking the marked entry, so these two
# straddle that boundary rather than merely being wide.
_ONE_BLOCK = blake3.BLOCK_LEN
_TWO_BLOCKS = 2 * blake3.BLOCK_LEN
# The grinds here run at bits <= 8, so a hit lands far inside one window. The
# default 2^16 would compress 64x more candidates per call for the same answer;
# `grind_search` tiles windows, so a wider search would still return this nonce.
_TEST_WINDOW = 1024


@dataclass(frozen=True)
class _HostBlake3Transcript(ByteHashTranscript):
    """The byte oracle: `ByteHashTranscript`'s framing, BLAKE3's squeeze and PoW.

    Host `bytes` throughout, over the reference BLAKE3 binding — no device code
    anywhere, which is what makes the comparison a gate rather than a
    restatement. The module docstring names the two helpers the two sides do
    share, and what that costs.
    """

    pow_preimage: int = _NO_PADDING

    def _absorb(self, payload: bytes) -> _HostBlake3Transcript:
        # The base constructs `ByteHashTranscript` literally rather than
        # `type(self)`, so absorbing through it would decay to the counter
        # squeeze at the first observe.
        return replace(self, buffer=self.buffer + payload)

    def _squeeze(self, n: int) -> bytes:
        # The XOF read: one finalize of the absorbed stream, taken at width `n`.
        # Reuses the base's buffer-to-digest marshalling so the oracle keeps
        # inheriting the framing it exists to represent; only the width varies.
        if n <= 0:
            return b""
        return replace(self, byte_hash=HostBlake3(n))._digest()

    def _preimage(self, state_digest: bytes, nonce: int) -> np.ndarray:
        row = np.zeros((1, self.pow_preimage), dtype=np.uint8)
        row[0, :_DIGEST_BYTES] = np.frombuffer(state_digest, dtype=np.uint8)
        row[0, _DIGEST_BYTES:_NO_PADDING] = np.frombuffer(_len8(nonce), dtype=np.uint8)
        return row

    def _passes(self, state_digest: bytes, nonce: int, bits: int) -> bool:
        digest = np.asarray(self.byte_hash.digest(self._preimage(state_digest, nonce)))
        return bool(_leading_zero_bits_ok(digest, bits)[0])

    def _grind(self, state_digest: bytes, bits: int) -> int:
        nonce = 0
        while not self._passes(state_digest, nonce, bits):
            nonce += 1
        return nonce

    def verify_pow(self, nonce: int, *, bits: int) -> tuple[ByteHashTranscript, bool]:
        ok = nonce == 0 if bits == 0 else self._passes(self._digest(), nonce, bits)
        return self.observe_bytes(_len8(nonce)), ok


def _host(
    domain: bytes = b"dom", pow_preimage: int = _NO_PADDING
) -> ByteHashTranscript:
    # Typed as the seam the tests drive it through: every op they call is
    # declared on the base and returns the base, the subclass supplying only
    # which bytes come back.
    base = ByteHashTranscript.new(domain, HostBlake3())
    return _HostBlake3Transcript(base.buffer, base.byte_hash, pow_preimage)


class Blake3FieldTranscriptTest(absltest.TestCase):
    def test_slice_framing_matches_byte_transcript(self) -> None:
        # observe(Array)/sample(n) use the byte transcript's slice framing, so the
        # squeezed challenge bytes match — the field surface is the byte surface,
        # made scan-threadable.
        vals = np.array([1, 2, 3, 4, 0xDEADBEEF, 5], dtype=np.uint32)

        b = _host().observe_slice(vals.astype("<u4").tobytes(), vals.size)
        b, b_sq = b.sample_slice(3, 4)  # 3 elements * 4 bytes

        f = Blake3FieldTranscript.new(b"dom", np.uint32)
        f, f_el = f.observe(fnp.asarray(vals)).sample(3)
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

        # A second squeeze pins the re-absorb of the first (both slice-framed).
        b, b2 = b.sample_slice(4, 4)
        f, f2 = f.sample(4)
        self.assertEqual(np.asarray(f2).astype("<u4").tobytes(), b2)

    def test_scalar_framing_matches_byte_transcript(self) -> None:
        # Scalar framing (KIND_SCALAR, no length prefix) matches, and must DIFFER
        # from the slice framing of the same single element.
        v = np.uint32(0xDEADBEEF)

        b, b_sq = _host().observe_scalar(v.tobytes()).sample_scalar(4)

        f = Blake3FieldTranscript.new(b"dom", np.uint32)
        f, f_el = f.observe_scalar(fnp.asarray(v)).sample_scalar()
        self.assertEqual(f_el.shape, ())  # scalar squeeze is 0-D
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

        g = Blake3FieldTranscript.new(b"dom", np.uint32)
        g, g_sl = g.observe(fnp.asarray(v).reshape(1)).sample(1)
        self.assertNotEqual(np.asarray(f_el).tobytes(), np.asarray(g_sl).tobytes())

    def test_label_and_bytes_framing_match_byte_transcript(self) -> None:
        label = b"zerocheck-v0"
        root = np.arange(32, dtype=np.uint8)  # a 32-byte on-device "root"

        b = _host().observe_label(label).observe_bytes(root.tobytes())
        b, b_sq = b.sample_slice(2, 4)

        f = Blake3FieldTranscript.new(b"dom", np.uint32)
        f = f.observe_label(label).observe_bytes(fnp.asarray(root))
        f, f_el = f.sample(2)
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), b_sq)

    def test_zero_width_slice_still_absorbs_framing(self) -> None:
        b, b_empty = _host().sample_slice(0, 4)
        b, b_next = b.sample_scalar(4)

        f = Blake3FieldTranscript.new(b"dom", np.uint32)
        f, f_empty = f.sample(0)
        f, f_next = f.sample_scalar()

        self.assertEqual(b_empty, b"")
        self.assertEqual(f_empty.shape, (0,))
        self.assertEqual(np.asarray(f_next).astype("<u4").tobytes(), b_next)

    def test_vector_observe_scalar_matches_scalar_chain(self) -> None:
        # observe_scalar of an [n] array frames each element as its own scalar
        # op, byte-identical to chaining n 0-d observes.
        vals = fnp.asarray(np.array([7, 0xDEADBEEF, 0, 42], dtype=np.uint32))

        chained = Blake3FieldTranscript.new(b"dom", np.uint32)
        for v in vals:
            chained = chained.observe_scalar(v)
        chained, c_el = chained.sample_scalar()

        batched = Blake3FieldTranscript.new(b"dom", np.uint32)
        batched, b_el = batched.observe_scalar(vals).sample_scalar()
        self.assertEqual(np.asarray(c_el).tobytes(), np.asarray(b_el).tobytes())

    def test_squeeze_is_an_xof_read_not_a_counter_chain(self) -> None:
        # The first BLAKE3 specific, pinned in both directions. A draw wider than
        # one digest is where the two constructions separate: the counter chain
        # hashes `buffer || ctr_le8` per 32-byte block, the XOF reads one stream.
        wide = 12  # 48 bytes > the 32-byte digest, so the chain would need 2 blocks

        _, xof = _host().sample_slice(wide, 4)
        counter = ByteHashTranscript.new(b"dom", HostBlake3())
        _, chained = counter.sample_slice(wide, 4)
        self.assertNotEqual(xof, chained)

        f = Blake3FieldTranscript.new(b"dom", np.uint32)
        _, f_el = f.sample(wide)
        self.assertEqual(np.asarray(f_el).astype("<u4").tobytes(), xof)

    def test_ghash_dtype_matches_byte_transcript(self) -> None:
        # A 16-byte element observes and samples through the same bitcast serde
        # as a 4-byte one; the wire bytes match the byte transcript over the same
        # serialization, and the samples come back as device ghash.
        import zk_dtypes  # noqa: F401  (registers fnp.binary_field_ghash)

        gh = fnp.binary_field_ghash
        lanes = np.array([1, 2, 3, 0xDEADBEEF], dtype=np.uint32)  # one 16-byte elem
        v_host = lanes.view(np.dtype(gh))  # shape (1,), known LE bytes

        b = _host(b"gh").observe_slice(lanes.tobytes(), 1)
        b, b_sq = b.sample_slice(2, 16)  # two ghash-width challenges

        f = Blake3FieldTranscript.new(b"gh", gh)
        f, f_el = f.observe(fnp.asarray(v_host)).sample(2)
        self.assertEqual(np.asarray(f_el).dtype, np.dtype(gh))  # device ghash
        self.assertEqual(f_el.shape, (2,))
        self.assertEqual(np.asarray(f_el).tobytes(), b_sq)

    def test_threads_under_jit(self) -> None:
        vals = np.arange(6, dtype=np.uint32)

        def run(x: fnp.ndarray) -> fnp.ndarray:
            f = Blake3FieldTranscript.new(b"dom", np.uint32)
            _, r = f.observe_and_sample(x, 1)
            return r

        eager = np.asarray(run(fnp.asarray(vals)))
        jitted = np.asarray(frx.jit(run)(fnp.asarray(vals)))
        self.assertEqual(eager.tobytes(), jitted.tobytes())

    def test_threads_a_jitted_loop(self) -> None:
        # The reason the module exists: the transcript is a `fori_loop` carry, so
        # a round loop stays inside the compiled program instead of decompiling
        # into a host loop. The host oracle pins that the loop's draws are the
        # same eight the byte wire specifies, in order.
        rounds = 8

        @frx.jit
        def run() -> fnp.ndarray:
            def body(
                _: fnp.ndarray, carry: tuple[Blake3FieldTranscript, fnp.ndarray]
            ) -> tuple[Blake3FieldTranscript, fnp.ndarray]:
                t, acc = carry
                t, c = t.sample_scalar()
                return t, acc.at[_].set(c)

            return frx.lax.fori_loop(
                0,
                rounds,
                body,
                (
                    Blake3FieldTranscript.new(b"dom", np.uint32),
                    fnp.zeros((rounds,), fnp.uint32),
                ),
            )[1]

        b = _host()
        want = b""
        for _ in range(rounds):
            b, drawn = b.sample_scalar(4)
            want += drawn
        self.assertEqual(np.asarray(run()).astype("<u4").tobytes(), want)

    def test_threads_through_sumcheck_prove(self) -> None:
        # The acceptance-critical path: the transcript threads the sumcheck round
        # driver (fold_rounds over StandardRound) under jit, no host callback. A
        # uint32 ring stands in for a scalar challenge field; the point here is
        # the transcript threading, not the sumcheck math.
        from zorch.challenge import ChallengePolicy
        from zorch.prove import fold_rounds
        from zorch.sumcheck.prover import (
            ProductSummand,
            StandardRound,
            initial_claim,
        )

        a = fnp.arange(8, dtype=fnp.uint32) + 1
        b = fnp.arange(8, dtype=fnp.uint32) + 2
        rnd = StandardRound(
            ProductSummand(degree=2), challenges=ChallengePolicy(fnp.uint32)
        )
        tr = Blake3FieldTranscript.new(b"sc", np.uint32)

        def run(x: fnp.ndarray, y: fnp.ndarray) -> tuple[fnp.ndarray, fnp.ndarray]:
            # 3 rounds folds the 2^3 stacked factors down to width 1.
            stacked = fnp.stack([x, y])
            carry, _, msgs = fold_rounds(
                rnd, initial_claim(stacked, fnp.sum(x * y), 3), tr, 3
            )
            return carry.state[:, 0], fnp.stack(msgs)

        eager = frx.tree_util.tree_map(np.asarray, run(a, b))
        jitted = frx.tree_util.tree_map(np.asarray, frx.jit(run)(a, b))
        self.assertEqual(eager[0].tobytes(), jitted[0].tobytes())
        self.assertEqual(eager[1].tobytes(), jitted[1].tobytes())


class Blake3ProofOfWorkTest(absltest.TestCase):
    """The grind, and the pre-image width that is the transcript's one wire
    parameter."""

    def _pair(
        self, pow_preimage: int
    ) -> tuple[ByteHashTranscript, Blake3FieldTranscript]:
        root = np.frombuffer(b"root", dtype=np.uint8)
        return (
            _host(b"pow", pow_preimage).observe_bytes(root.tobytes()),
            Blake3FieldTranscript.new(
                b"pow", np.uint32, pow_preimage_bytes=pow_preimage
            ).observe_bytes(fnp.asarray(root)),
        )

    def test_grind_check_witness_match_byte_transcript(self) -> None:
        # The device grind reproduces the byte oracle's u64-nonce PoW at every
        # pre-image width: same (lowest) nonce, and the transcripts stay in
        # lockstep afterwards. check_witness accepts the honest witness, rejects a
        # tampered one, and advances regardless (the DuplexTranscript contract).
        # `_TWO_BLOCKS` is the case a hand-rolled single-compression pre-image
        # hash could not serve: the width is the message's, not a block's.
        #
        # Flat case list, not a product: at `bits == 0` both sides short-circuit
        # before reading the width, so pairing 0 with each width would run the
        # same program three times. `bits = 5` is the only case that reaches
        # `leading_zero_bits_ok`'s partial-byte branch — every other grind in the
        # repo is a multiple of 8, which leaves that branch to the host twin.
        for pow_preimage, bits in (
            (_NO_PADDING, 0),
            (_NO_PADDING, 5),
            (_NO_PADDING, 8),
            (_ONE_BLOCK, 8),
            (_TWO_BLOCKS, 8),
        ):
            with self.subTest(pow_preimage=pow_preimage, bits=bits):
                b, f = self._pair(pow_preimage)
                b, b_nonce = b.grind_pow(bits)
                _, b_ch = b.sample_scalar(4)

                f, witness = f.grind(bits, chunk=_TEST_WINDOW)
                self.assertEqual(int(witness), b_nonce)
                _, f_ch = f.sample_scalar()
                self.assertEqual(np.asarray(f_ch).astype("<u4").tobytes(), b_ch)

                _, vf = self._pair(pow_preimage)
                vf, ok = vf.check_witness(witness, pow_bits=bits)
                self.assertTrue(bool(ok))
                _, vf_ch = vf.sample_scalar()
                self.assertEqual(np.asarray(vf_ch).astype("<u4").tobytes(), b_ch)

                if bits:
                    hb, bad = self._pair(pow_preimage)
                    _, bad_ok = bad.check_witness(int(witness) + 1, pow_bits=bits)
                    # Against the oracle, not against `False` — otherwise the
                    # rejection is asserted by nothing but itself.
                    _, host_bad_ok = hb.verify_pow(int(witness) + 1, bits=bits)
                    self.assertFalse(bool(bad_ok))
                    self.assertFalse(host_bad_ok)

    def test_pow_preimage_width_is_a_distinct_wire(self) -> None:
        # The second BLAKE3 specific. A message's length is part of what BLAKE3
        # hashes, so padding the pre-image to a whole block is a different search
        # — and a witness from one width is not a witness for the other. Without
        # this the test above would pass with the parameter ignored.
        _, unpadded = self._pair(_NO_PADDING)
        _, padded = self._pair(_ONE_BLOCK)
        unpadded, w_unpadded = unpadded.grind(8, chunk=_TEST_WINDOW)
        padded, w_padded = padded.grind(8, chunk=_TEST_WINDOW)
        self.assertNotEqual(int(w_unpadded), int(w_padded))

        _, cross = self._pair(_ONE_BLOCK)
        _, ok = cross.check_witness(w_unpadded, pow_bits=8)
        self.assertFalse(bool(ok))

    def test_marker_rides_the_pre_image_only_up_to_one_block(self) -> None:
        # The width guard is invisible to every other test here: both arms hash
        # the same bytes, so only the lowering tells them apart. Up to a block the
        # pre-image carries hash-frx's marker for an emitter to collapse; past it
        # the unmarked body is taken instead, because a marked call compiles that
        # body and it stops being affordable at a chunk.
        for pow_preimage, marked in (
            (_NO_PADDING, True),
            (_ONE_BLOCK, True),
            (_TWO_BLOCKS, False),
        ):
            with self.subTest(pow_preimage=pow_preimage, marked=marked):
                _, t = self._pair(pow_preimage)
                hlo = (
                    frx.jit(lambda x: x.grind(8, chunk=_TEST_WINDOW)).lower(t).as_text()
                )
                self.assertEqual(BLAKE3_MARKER in hlo, marked)

    def test_grind_bits_out_of_range_rejected(self) -> None:
        # Mirrors the byte transcript: > 256 (or negative) leading-zero bits on a
        # 32-byte digest is impossible and rejected up front.
        t = Blake3FieldTranscript.new(b"pow", np.uint32)
        for bits in (-1, 257):
            with self.assertRaises(ValueError):
                t.grind(bits)
            with self.assertRaises(ValueError):
                t.check_witness(0, pow_bits=bits)

    def test_pow_preimage_below_the_nonce_width_rejected(self) -> None:
        # The one real constraint: below digest+nonce the nonce does not fit.
        # There is no upper bound — the pre-image is hashed as a whole message,
        # so a width past a block is slower and still correct, which
        # `test_grind_check_witness_match_byte_transcript` pins at _TWO_BLOCKS.
        with self.assertRaises(ValueError):
            Blake3FieldTranscript.new(
                b"pow", np.uint32, pow_preimage_bytes=_NO_PADDING - 1
            )


class Blake3FieldTranscriptPytreeTest(absltest.TestCase):
    def test_only_the_hash_state_is_a_leaf(self) -> None:
        # dtype and the pre-image width are META: as data fields they would enter
        # a loop carry, where a Python int is not a traceable leaf and the dtype
        # would stop being static.
        t = Blake3FieldTranscript.new(b"dom", np.uint32, pow_preimage_bytes=_ONE_BLOCK)
        leaves, treedef = frx.tree_util.tree_flatten(t)
        self.assertEqual(len(leaves), len(frx.tree_util.tree_leaves(t.state)))

        back = frx.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(back.dtype, t.dtype)
        self.assertEqual(back.pow_preimage_bytes, _ONE_BLOCK)

    def test_threads_a_jit_boundary_as_argument_and_result(self) -> None:
        t = Blake3FieldTranscript.new(b"dom", np.uint32, pow_preimage_bytes=_ONE_BLOCK)

        def step(x: Blake3FieldTranscript) -> Blake3FieldTranscript:
            return x.observe_scalar(fnp.uint32(7))

        out = frx.jit(step)(t)
        self.assertEqual(out.pow_preimage_bytes, _ONE_BLOCK)
        for a, b in zip(
            frx.tree_util.tree_leaves(step(t)),
            frx.tree_util.tree_leaves(out),
            strict=True,
        ):
            self.assertEqual(np.asarray(a).tobytes(), np.asarray(b).tobytes())


if __name__ == "__main__":
    absltest.main()
