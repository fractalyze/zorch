# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The lnp transcript wire — encoding, the element-count convention, and
the functional contract the abort loop leans on.

The point of pinning these here rather than only through a protocol suite:
a protocol's own tests pass whichever convention it picks, because prover
and verifier both use it. Only a shared test can catch a sibling module
drifting to a different one.
"""

from __future__ import annotations

import numpy as np
from absl.testing import absltest
from hash_frx.sha256 import HostSha256

from zorch.byte_transcript import ByteHashTranscript
from zorch.lnp.transcript import absorb_stacks, stack_bytes

_LIMBS, _D = 2, 4


def _transcript() -> ByteHashTranscript:
    return ByteHashTranscript.new(b"lnp-transcript-test", HostSha256())


def _stack(rows: int, start: int = 0) -> np.ndarray:
    size = rows * _LIMBS * _D
    return np.arange(start, start + size, dtype=np.uint64).reshape(rows, _LIMBS, _D)


class StackBytesTest(absltest.TestCase):
    def test_it_is_c_order_little_endian_u64(self) -> None:
        stack = _stack(2)
        self.assertEqual(stack_bytes(stack), stack.astype("<u8").tobytes())
        self.assertEqual(len(stack_bytes(stack)), stack.size * 8)

    def test_a_non_contiguous_view_encodes_as_it_reads(self) -> None:
        """A transposed or sliced view must serialize in its logical order,
        not its buffer's — otherwise the wire depends on how the caller
        happened to build the array."""
        stack = _stack(3)
        view = stack[::2]
        self.assertEqual(stack_bytes(view), np.ascontiguousarray(view).tobytes())

    def test_a_non_uint64_stack_is_refused(self) -> None:
        """`astype` would *reinterpret* a signed array rather than convert
        it, so `-1` and the residue `2**64 - 1` would bind to identical
        transcript bytes — two values, one transcript. Floats would
        truncate. Both are refused before any binding happens."""
        residues = _stack(1)
        for name, bad in (
            ("signed", residues.astype(np.int64)),
            ("float", residues.astype(np.float64)),
            ("narrow", residues.astype(np.uint32)),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(TypeError, "uint64 residues"):
                    stack_bytes(bad)

    def test_a_signed_minus_one_would_have_collided(self) -> None:
        """Why the gate is a TypeError and not a cast: the collision it
        prevents is silent and yields a *valid-looking* transcript. Spelled
        against numpy's own `tobytes` rather than against `stack_bytes`'s
        body, so the two cannot drift into agreeing by construction."""
        self.assertEqual(
            np.array([[[-1]]], dtype=np.int64).tobytes(),
            stack_bytes(np.array([[[2**64 - 1]]], dtype=np.uint64)),
        )

    def test_an_empty_stack_is_empty_bytes(self) -> None:
        self.assertEqual(stack_bytes(np.empty((0, _LIMBS, _D), dtype=np.uint64)), b"")


class AbsorbStacksTest(absltest.TestCase):
    def test_the_count_is_ring_elements_not_words(self) -> None:
        """The slice count is domain-separation input, so counting words
        instead of ring elements yields a different transcript from the
        same bytes. This is the convention both sides of every lnp proof
        replay, and the one a sibling module must not respell."""
        stack = _stack(2)
        _, elements = absorb_stacks(_transcript(), stack).sample_scalar(32)
        _, words = (
            _transcript()
            .observe_slice(stack_bytes(stack), stack.size)
            .sample_scalar(32)
        )
        self.assertNotEqual(bytes(elements), bytes(words))

    def test_stacks_absorb_in_order(self) -> None:
        first, second = _stack(1), _stack(1, start=100)
        _, forward = absorb_stacks(_transcript(), first, second).sample_scalar(32)
        _, backward = absorb_stacks(_transcript(), second, first).sample_scalar(32)
        self.assertNotEqual(bytes(forward), bytes(backward))

    def test_absorbing_is_functional(self) -> None:
        """The caller's transcript value is untouched, which is what lets a
        rejected Fiat-Shamir-with-aborts attempt restart from it."""
        base = _transcript()
        _, before = base.sample_scalar(32)
        absorb_stacks(base, _stack(2))
        _, after = base.sample_scalar(32)
        self.assertEqual(bytes(before), bytes(after))

    def test_absorbing_nothing_still_advances_by_the_label_only(self) -> None:
        """Zero stacks is the degenerate call a pure opening makes; it must
        return the transcript unchanged rather than fail."""
        base = _transcript()
        _, untouched = base.sample_scalar(32)
        _, empty = absorb_stacks(base).sample_scalar(32)
        self.assertEqual(bytes(untouched), bytes(empty))


if __name__ == "__main__":
    absltest.main()
