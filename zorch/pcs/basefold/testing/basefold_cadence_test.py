# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Non-native BaseFold cadence round-trip: `open_with_basis` drives the
row-batch-prefix + multi-arity-epoch driver (`_open_with_basis_cadence`) and
`verify_with_basis` replays it (`_verify_with_basis_cadence`), on an in-repo
flock-SHAPED instance — a degree-2 product `(a, b)` sumcheck with `(u0, u2)`
messages, a basis (no opening point) statement bind, and multi-element fold
arities. This gives zorch CI regression coverage of the cadence driver + verifier
WITHOUT the flock repo (which byte-anchors the same seams cross-repo).

The instance is kept prime-field and small (`size=medium`): the codeword is built
per-lane as `encode(evals_to_coeffs(lane_message))` so the affine sumcheck fold
ties to the code's coefficient fold (the two coincide in char-2 for flock; the
`evals_to_coeffs` bridge makes the tie hold over an odd-characteristic field too),
and the bit-reversed layout makes a contiguous epoch coset fold-stable. Names may
say "flock" (the shape this covers); the driver + verifier under test stay
scheme-agnostic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import frx.numpy as fnp
from absl.testing import absltest
from frx import Array
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.pcs.basefold.choreography import BasefoldChoreography
from zorch.pcs.basefold.config import BasefoldConfig, CadenceProof
from zorch.pcs.basefold.kernel import SumcheckKernel
from zorch.pcs.basefold.prover import BasefoldProver, BasefoldProverData
from zorch.pcs.basefold.verifier import BasefoldVerifier
from zorch.poly.multilinear import mle_evals_to_coeffs
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript, GrindingTranscript, TranscriptT


def _transcript() -> DuplexTranscript:
    return DuplexTranscript.new(koalabear16_perm(), rate=8)


@dataclass(frozen=True)
class _ProductKernel(SumcheckKernel):
    """The flock-shaped degree-2 product sumcheck over the state `(a, b)`: message
    `(u0, u2) = (Σ aₑbₑ, Σ(aₒ−aₑ)(bₒ−bₑ))` (g(0), leading coeff), both vectors
    folded by the affine multilinear bind, terminal `(a[0], b[0])`. Proves
    `Σ_x a(x)·b(x) = value`. Verify side: `reduce_claim` advances the running
    target `g(r)`, `verify_final` checks `a[0]·b[0] == target` and hands the FRI
    terminal `a[0]`."""

    def initial_state(self, mle: Array, basis: Array, claim: Array) -> tuple:
        del claim  # the target rides `value` into verify, not the state
        return (mle, basis)

    def message(self, state: tuple) -> tuple[Array, Array]:
        a, b = state
        ae, ao = a[0::2], a[1::2]
        be, bo = b[0::2], b[1::2]
        u0 = fnp.sum(ae * be)
        u2 = fnp.sum((ao - ae) * (bo - be))
        return u0, u2

    def fold(self, state: tuple, message: tuple[Array, Array], r: Array) -> tuple:
        del message
        a, b = state
        one = fnp.ones((), r.dtype)
        af = (one - r) * a[0::2] + r * a[1::2]
        bf = (one - r) * b[0::2] + r * b[1::2]
        return (af, bf)

    def final(self, state: tuple) -> tuple[Array, Array]:
        a, b = state
        return (a[0], b[0])

    def reduce_claim(
        self, claim: Array, message: tuple[Array, Array], r: Array
    ) -> Array:
        u0, u2 = message
        two = fnp.ones((), r.dtype) + fnp.ones((), r.dtype)
        u1 = claim - two * u0 - u2  # g(X) = u0 + u1·X + u2·X², g(0)+g(1) = claim
        return u0 + u1 * r + u2 * r * r

    def verify_final(self, claim: Array, final_state: tuple) -> tuple[Array, Array]:
        fa, fb = final_state
        return (fa * fb == claim), fa


@dataclass(frozen=True)
class _CadenceChoreography(BasefoldChoreography[GrindingTranscript]):
    """The flock-shaped basis-entry wire: no opening point (bind root + value),
    eager `(u0, u2)` emission, bare fold-challenge sample. Drops flock's byte
    deltas (root_to_f128, masked queries) — those are consumer serialization, not
    needed for an in-repo round-trip."""

    @property
    def eager_messages(self) -> bool:
        return True

    def bind_statement(
        self,
        transcript: TranscriptT,
        root: Array,
        point: Array | None,
        value: Array,
    ) -> TranscriptT:
        del point  # no opening point on the basis entry
        return transcript.observe(root).observe(value)

    def fold_challenge(
        self, transcript: TranscriptT, msg: Array | None, level: int, fold_idx: int
    ) -> tuple[TranscriptT, Array]:
        del msg, level, fold_idx  # eager: the message is already absorbed
        t, r = transcript.sample(1)
        return t, r[0]


@dataclass(frozen=True)
class _GrindingCadenceChoreography(_CadenceChoreography):
    """The cadence wire above, but scheduling a query-phase grind — exercises
    the cadence verify's grind guard (`CadenceProof.pow_witnesses` is the wire
    slot, but the grind production + check are a deferred delta)."""

    def query_grind_bits(self, level: int) -> int | None:
        del level
        return 0


def _codeword(code: BitReversedReedSolomon, a: Array, num_ntts: int) -> Array:
    """The interleaved codeword for MLE `a`: lane `l` holds `encode` of the
    coefficient form of `a`'s lane-`l` sub-message (`a[l::num_ntts]`), stored
    position-major `[n_pos, num_ntts] -> [n_pos*num_ntts]`. `encode`/`lane_combine`/
    `evals_to_coeffs` are all linear, so `lane_combine(codeword, rb)` equals
    `encode(evals_to_coeffs(a folded over the prefix))`, and each FRI `code.fold`
    then keeps `codeword == encode(evals_to_coeffs(a folded))` — so the fully
    folded codeword is the constant `a[0]`, the sumcheck's terminal."""
    lanes = [
        code.encode(mle_evals_to_coeffs(a[lane::num_ntts])) for lane in range(num_ntts)
    ]
    return fnp.stack(lanes, axis=1).reshape(-1)


class BasefoldCadenceRoundTripTest(absltest.TestCase):
    # The open + verify ride the eager composite path (real poseidon2 Merkle +
    # NTT folds), so build the instance and prove ONCE at class scope — every case
    # reuses the one proof (tampers on a copy), keeping the whole suite inside the
    # `size=medium` budget instead of re-tracing per method.
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        prefix, arities, blowup, num_queries = 1, (2, 1), 2, 3
        log_dim = sum(arities)  # 3
        num_vars = prefix + log_dim  # 4
        num_ntts = 1 << prefix  # 2
        message_len = 1 << log_dim  # 8
        code = BitReversedReedSolomon(message_len, blowup, F)
        _, _, tree = koalabear16_merkle()
        config = BasefoldConfig(
            num_vars=num_vars,
            num_queries=num_queries,
            row_batch_prefix=prefix,
            fold_arities=arities,
        )
        kernel, chor = _ProductKernel(), _CadenceChoreography()
        prover = BasefoldProver(
            code,
            tree,
            num_queries=num_queries,
            choreography=chor,
            kernel=kernel,
            config=config,
        )
        cls.verifier = BasefoldVerifier(
            code,
            tree,
            num_queries=num_queries,
            choreography=chor,
            kernel=kernel,
            config=config,
        )
        n = 1 << num_vars
        a, cls.b = rand_field(1, (n,), F), rand_field(2, (n,), F)
        n_pos = code.block_len
        cw = _codeword(code, a, num_ntts)
        pd = BasefoldProverData(
            digest_layers=[],
            mle=a,
            codeword=cw,
            leaves=cw.reshape(n_pos, num_ntts),
            widths=(num_ntts,),
        )
        cls.value = fnp.sum(a * cls.b)  # the product-sumcheck target Σ_x a(x)·b(x)
        cls.commitment, _ = tree.commit(cw.reshape(n_pos, num_ntts))
        cls.proof, _ = prover.open_with_basis(pd, cls.b, cls.value, _transcript())

    def _verify(self, proof: CadenceProof, commitment: Array | None = None) -> bool:
        commitment = self.commitment if commitment is None else commitment
        ok, _ = self.verifier.verify_with_basis(
            commitment, self.b, self.value, proof, _transcript()
        )
        return bool(ok)

    def test_open_with_basis_takes_the_cadence_path(self) -> None:
        # The non-native schedule routes through the cadence driver, not the
        # native jitted body — so the proof is a `CadenceProof` with the expected
        # layer shape (2 committed roots: post-prefix + one epoch boundary; 3
        # opened layers; num_vars round messages).
        self.assertIsInstance(self.proof, CadenceProof)
        self.assertLen(self.proof.commit_roots, 2)
        self.assertLen(self.proof.layer_openings, 3)
        self.assertLen(self.proof.round_messages, 4)

    def test_round_trip_accepts(self) -> None:
        self.assertTrue(self._verify(self.proof))

    def test_verify_rejects_tampered_round_message(self) -> None:
        u0, u2 = self.proof.round_messages[0]
        bad = dataclasses.replace(
            self.proof, round_messages=[(u0 + F(1), u2), *self.proof.round_messages[1:]]
        )
        self.assertFalse(self._verify(bad))

    def test_verify_rejects_tampered_layer_opening(self) -> None:
        op = self.proof.layer_openings[1]  # the post-prefix coset leaves
        bad_op = dataclasses.replace(op, row=op.row + F(1))
        bad = dataclasses.replace(
            self.proof,
            layer_openings=[
                self.proof.layer_openings[0],
                bad_op,
                *self.proof.layer_openings[2:],
            ],
        )
        self.assertFalse(self._verify(bad))

    def test_verify_with_basis_raises_on_scheduled_grind(self) -> None:
        # `CadenceProof.pow_witnesses` is the frozen wire slot, but a scheduled
        # grind's production + check are a deferred delta — the cadence verify
        # must fail loud, symmetric to the prover's grind guard on
        # `_open_with_basis_cadence`.
        grinding_verifier = dataclasses.replace(
            self.verifier, choreography=_GrindingCadenceChoreography()
        )
        with self.assertRaises(NotImplementedError):
            grinding_verifier.verify_with_basis(
                self.commitment, self.b, self.value, self.proof, _transcript()
            )

    def test_verify_rejects_tampered_final_state(self) -> None:
        # Exercises the kernel's `verify_final` tie: a forged terminal value fails
        # the `a[0]·b[0] == target` check (and the codeword-constant tie).
        fa, fb = self.proof.final_state
        bad = dataclasses.replace(self.proof, final_state=(fa + F(1), fb))
        self.assertFalse(self._verify(bad))


if __name__ == "__main__":
    absltest.main()
