# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Π_many opening — completeness, soundness rejections, and the FS chain.

Structural correctness without goldens, like `ajtai_test`: an honest prover's
proof verifies (the budgeted repeat loop included — at the test's repetition
rates most proofs survive at least one rejection), every tampered surface
(witness, relation target, response, challenge, transcript) is rejected, and
the non-interactive chain is byte-deterministic. The zero-knowledge smoke
checks the accepted responses' first two moments against the masking
Gaussian's — the full statistical gate stays with the protocol's parameter
work, not this seam.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
from absl.testing import absltest
from lattice_frx.split_ring import HostSplitRing

from zorch.byte_transcript import ByteTranscript
from zorch.commit.ajtai import AbdlopCommitment
from zorch.lnp import opening as opening_module
from zorch.lnp.opening import AbdlopOpening, OpeningProof
from zorch.lnp.testing import lnp_fixture
from zorch.lnp.testing.lnp_fixture import STD as _STD
from zorch.lnp.testing.lnp_fixture import D as _D

# This suite's module shape: n rows, m1 committed / m2 randomness columns,
# ℓ messages, N linear relations. `_M1` must match the fixture's, which is
# what its masking standard deviation was derived for.
_ROWS, _M1, _M2, _ELL, _N = 2, lnp_fixture.M1, 2, 1, 1


def _ring() -> HostSplitRing:
    return lnp_fixture.ring()


def _scheme(ring: HostSplitRing) -> AbdlopCommitment:
    return AbdlopCommitment(
        ring,
        rows=_ROWS,
        s1_cols=_M1,
        randomness_cols=_M2,
        messages=_ELL,
        beta1_inf=1,
        beta2_inf=1,
    )


def _opening(ring: HostSplitRing, **overrides: object) -> AbdlopOpening:
    """The test parameter point, with one kwarg moved per call — the
    `_Instance.prove/verify` convention applied to construction, so a test
    that varies `fail_prob` states only that."""
    params = lnp_fixture.OPENING_PARAMS | overrides
    return AbdlopOpening(_scheme(ring), **params)  # type: ignore[arg-type]


def _transcript(tag: bytes = b"") -> ByteTranscript:
    return lnp_fixture.transcript(b"lnp-opening-test", tag)


class _Instance:
    """One honest Π_many instance: publics, witness, and commitments. The
    `prove`/`verify` helpers take keyword overrides over the honest
    statement, so a tamper test states only its one deviation."""

    def __init__(self, seed: int) -> None:
        self.ring = _ring()
        self.opening = _opening(self.ring)
        rng = np.random.default_rng(seed)
        self.a1 = self.ring.uniform_stack(rng, _ROWS, _M1)
        self.a2 = self.ring.uniform_stack(rng, _ROWS, _M2)
        self.b = self.ring.uniform_stack(rng, _ELL, _M2)
        self.r1 = self.ring.uniform_stack(rng, _N, _M1)
        self.rm = self.ring.uniform_stack(rng, _N, _ELL)
        self.s1 = rng.integers(-1, 2, size=(_M1, _D)).astype(np.int64)
        self.s2 = rng.integers(-1, 2, size=(_M2, _D)).astype(np.int64)
        message = self.ring.uniform_stack(rng, _ELL)
        commitment = self.opening.scheme.commit(
            self.a1,
            self.a2,
            self.b,
            self.ring.from_signed_stack(self.s1),
            self.ring.from_signed_stack(self.s2),
            message,
        )
        self.t_a, self.t_b = commitment.t_a, commitment.t_b
        self.u = self.ring.add(
            self.ring.matvec(self.r1, self.ring.from_signed_stack(self.s1)),
            self.ring.matvec(self.rm, message),
        )
        self.rng = rng

    def prove(
        self,
        tag: bytes = b"",
        opening: AbdlopOpening | None = None,
        **overrides: np.ndarray,
    ) -> tuple[OpeningProof, ByteTranscript]:
        args = dict(a1=self.a1, a2=self.a2, b=self.b, r1=self.r1, rm=self.rm)
        args.update(overrides)
        return (opening or self.opening).prove(
            s1=self.s1,
            s2=self.s2,
            rng=self.rng,
            transcript=_transcript(tag),
            **args,
        )

    def verify(
        self, proof: OpeningProof, tag: bytes = b"", **overrides: np.ndarray
    ) -> bool:
        args = dict(
            a1=self.a1,
            a2=self.a2,
            b=self.b,
            r1=self.r1,
            rm=self.rm,
            t_a=self.t_a,
            t_b=self.t_b,
            u=self.u,
        )
        args.update(overrides)
        ok, _ = self.opening.verify(proof=proof, transcript=_transcript(tag), **args)
        return ok


class OpeningCompletenessTest(absltest.TestCase):
    def test_an_honest_proof_verifies(self) -> None:
        instance = _Instance(0)
        proof, _ = instance.prove()
        self.assertTrue(instance.verify(proof))

    def test_the_chain_is_byte_deterministic(self) -> None:
        first = _Instance(1)
        second = _Instance(1)
        p1, _ = first.prove()
        p2, _ = second.prove()
        np.testing.assert_array_equal(p1.c, p2.c)
        np.testing.assert_array_equal(p1.z1, p2.z1)
        np.testing.assert_array_equal(p1.z2, p2.z2)

    def test_distinct_transcripts_yield_distinct_challenges(self) -> None:
        instance = _Instance(2)
        p1, _ = instance.prove(tag=b"one")
        p2, _ = instance.prove(tag=b"two")
        self.assertFalse(np.array_equal(p1.c, p2.c))
        self.assertTrue(instance.verify(p1, tag=b"one"))
        self.assertFalse(instance.verify(p1, tag=b"two"))

    def test_the_repeat_loop_is_exercised_and_budgeted(self) -> None:
        """At M ≈ 2.7 per response the acceptance rate is ≈ 1/M² ≈ 0.14, so
        across a handful of proofs at least one takes more than one attempt
        — observed directly as the per-proof coin count (two coins per
        attempt, drawn whether or not the gates short-circuit) while every
        proof still verifies."""
        instance = _Instance(3)
        attempts = []
        with mock.patch.object(
            opening_module, "_coin", side_effect=opening_module._coin
        ) as spy:
            for i in range(4):
                seen = spy.call_count
                proof, _ = instance.prove(tag=bytes([i]))
                attempts.append((spy.call_count - seen) // 2)
                self.assertTrue(instance.verify(proof, tag=bytes([i])))
        for count in attempts:
            self.assertGreaterEqual(count, 1)
        # The aggregate, not the spread: "at least one proof paid a
        # rejection" is what the loop claims, and four single-attempt
        # proofs is a ≈4e-4 event, whereas four *equal* counts is a
        # likelier coincidence that says nothing about the loop.
        self.assertGreater(sum(attempts), len(attempts))

    def test_an_exhausted_budget_raises(self) -> None:
        """fail_prob = 0.5 shrinks the budget to a handful of attempts;
        forcing every rejection then exhausts it — the loop must raise,
        not spin."""
        starved = _opening(_ring(), fail_prob=0.5)
        instance = _Instance(4)
        with mock.patch.object(opening_module, "_rej1", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "prove"):
                instance.prove(opening=starved)

    def test_overflowing_repetition_rates_are_rejected_at_construction(self) -> None:
        """Past ~1e154 the product of the two rates overflows to infinity,
        so the derived acceptance underflows to exactly 0.0. Without a gate
        here the budget helper refuses `accept_prob` — a parameter this
        caller never passed — instead of this module naming the rates."""
        with self.assertRaisesRegex(ValueError, r"rep1.rep2 overflows"):
            _opening(_ring(), rep1=1e200, rep2=1e200)

    def test_an_absurd_attempt_budget_is_rejected_at_construction(self) -> None:
        """A mis-derived repetition rate must fail loudly at construction —
        the budget formula would otherwise size the loop in the 10^17s and
        prove would hang rather than err."""
        with self.assertRaisesRegex(ValueError, "attempt budget"):
            _opening(_ring(), rep1=1e9, rep2=1e9)


class OpeningSoundnessTest(absltest.TestCase):
    # One honest instance and one proof for the whole class: every method
    # here only reads them (each copies before tampering, and `verify` is
    # pure), so re-proving per method is four proofs of dead work.
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.instance = _Instance(5)
        cls.proof, _ = cls.instance.prove()

    def test_a_tampered_response_is_rejected(self) -> None:
        z1 = self.proof.z1.copy()
        z1[0, 0] += 1
        bad = OpeningProof(c=self.proof.c, z1=z1, z2=self.proof.z2)
        self.assertFalse(self.instance.verify(bad))

    def test_an_over_norm_response_is_rejected(self) -> None:
        z1 = self.proof.z1.copy()
        z1[0, 0] += 10**9
        bad = OpeningProof(c=self.proof.c, z1=z1, z2=self.proof.z2)
        self.assertFalse(self.instance.verify(bad))

    def test_a_tampered_challenge_is_rejected(self) -> None:
        c = self.proof.c.copy()
        c[0] += 1  # unconditionally a different vector, unlike a swap
        bad = OpeningProof(c=c, z1=self.proof.z1, z2=self.proof.z2)
        self.assertFalse(self.instance.verify(bad))

    def test_a_wrong_relation_target_is_rejected(self) -> None:
        """u off by one ring element: the recomputed v drifts, so the
        recomputed challenge cannot match."""
        wrong_u = self.instance.ring.add(
            self.instance.u,
            self.instance.ring.uniform_stack(np.random.default_rng(99), _N),
        )
        self.assertFalse(self.instance.verify(self.proof, u=wrong_u))

    def test_a_wrong_commitment_is_rejected(self) -> None:
        other_t_a = self.instance.ring.uniform_stack(np.random.default_rng(6), _ROWS)
        self.assertFalse(self.instance.verify(self.proof, t_a=other_t_a))


class OpeningZeroKnowledgeSmokeTest(absltest.TestCase):
    def test_accepted_responses_match_the_masking_moments(self) -> None:
        """z1's coefficients under acceptance should look like D_{s1_std}:
        mean ≈ 0 and variance within 15% of s1_std² — the coarse two-moment
        gate; the χ² shape gate belongs to the parameter work."""
        instance = _Instance(7)
        coefficients: list[np.ndarray] = []
        for i in range(12):
            proof, _ = instance.prove(tag=bytes([i]))
            coefficients.append(proof.z1.reshape(-1).astype(np.float64))
        flat = np.concatenate(coefficients)
        std = _STD
        self.assertLess(abs(float(flat.mean())), 4.0 * std / np.sqrt(flat.size))
        self.assertLess(abs(float(flat.var()) / std**2 - 1.0), 0.15)


class OpeningSurfaceTest(absltest.TestCase):
    def test_a_wrong_witness_shape_raises(self) -> None:
        instance = _Instance(8)
        with self.assertRaises(ValueError):
            instance.opening.prove(
                instance.a1,
                instance.a2,
                instance.b,
                instance.r1,
                instance.rm,
                instance.s1[:, :-1],
                instance.s2,
                instance.rng,
                _transcript(),
            )

    def test_a_malformed_wire_challenge_is_refused(self) -> None:
        """`c` is wire data like the responses, so it meets the same gate,
        and refusal is a `False` — the prover chose these bytes. Without the
        gate a list `c` verified *true* (`np.array_equal` is happy to compare
        one against an array) and a wrong length surfaced as a
        `SplitRing.mul` shape error from inside the ring."""
        instance = _Instance(10)
        proof, _ = instance.prove()
        # Typed `object`: these inputs deliberately violate `OpeningProof`'s
        # annotation, which is exactly the promise a wire decoder cannot make
        # and the runtime gate therefore has to.
        cases: tuple[tuple[str, object], ...] = (
            ("a list", [int(v) for v in proof.c]),
            ("float coefficients", proof.c.astype(np.float64)),
            ("a trailing coefficient", np.append(proof.c, 0)),
        )
        for name, c in cases:
            with self.subTest(name):
                bad = OpeningProof(c=c, z1=proof.z1, z2=proof.z2)  # type: ignore[arg-type]
                self.assertFalse(instance.verify(bad))

    def test_a_malformed_wire_response_is_refused(self) -> None:
        """`z1` and `z2` meet the same gate as `c`, and the same verdict.
        A float response would otherwise reach `_challenge_times`' int64
        cast and be scored after silent truncation."""
        instance = _Instance(11)
        proof, _ = instance.prove()
        for name in ("z1", "z2"):
            good = getattr(proof, name)
            cases: tuple[tuple[str, object], ...] = (
                ("float coefficients", good.astype(np.float64)),
                ("a dropped row", good[:-1]),
                ("a nested list", good.tolist()),
            )
            for case, z in cases:
                with self.subTest(f"{name}: {case}"):
                    fields: dict[str, object] = dict(
                        c=proof.c, z1=proof.z1, z2=proof.z2
                    )
                    fields[name] = z
                    self.assertFalse(instance.verify(OpeningProof(**fields)))  # type: ignore[arg-type]

    def test_zero_relations_degenerate_to_the_pure_opening(self) -> None:
        """No linear relations is a real statement — a bare ABDLOP opening.
        The same instance proves it: N lives in `r1`'s row count, so passing
        empty relation matrices is the whole deviation."""
        instance = _Instance(9)
        empty = instance.ring.zeros(0)
        r1 = np.empty((0, _M1) + instance.t_a.shape[1:], dtype=np.uint64)
        rm = np.empty((0, _ELL) + instance.t_a.shape[1:], dtype=np.uint64)
        proof, _ = instance.prove(r1=r1, rm=rm)
        self.assertTrue(instance.verify(proof, r1=r1, rm=rm, u=empty))


if __name__ == "__main__":
    absltest.main()
