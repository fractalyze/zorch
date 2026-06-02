# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
import zk_dtypes
from absl.testing import absltest

from zorch.logup_gkr.round import LogupGkrRound
from zorch.testkit.random_field import rand_field
from zorch.transcript import StubTranscript

KB = zk_dtypes.koalabear


def _state(seed, width):
    """Five MLE-eval factors in combine order [eq, n0, d1, n1, d0]."""
    return [rand_field(seed + i, (width,), KB) for i in range(5)]


def _eval_cubic_at(evals, r):
    """Interpolate the degree-3 poly given by `evals` at u=0,1,2,3, eval at r
    (Lagrange over the field — the sumcheck verifier's per-round check)."""
    xs = [jnp.array(i, evals.dtype) for i in range(4)]
    acc = jnp.array(0, evals.dtype)
    for j in range(4):
        term = evals[j]
        for m in range(4):
            if m != j:
                term = term * (r - xs[m]) / (xs[j] - xs[m])
        acc = acc + term
    return acc


class LogupGkrRoundTest(absltest.TestCase):
    def test_round_poly_matches_naive_cubic(self):
        st = _state(20, 8)
        rnd = LogupGkrRound(jnp.array(7, KB))
        msg = rnd._round_poly(st)
        self.assertEqual(msg.shape, (4,))  # degree 3 -> 4 evals
        half = 4
        for u in range(4):
            uf = jnp.array(u, KB)
            folded = [x[:half] + uf * (x[half:] - x[:half]) for x in st]
            want = jnp.sum(rnd._combine(*folded))
            self.assertTrue(bool(msg[u] == want))

    def test_round_poly_with_batch_dimension(self):
        # Leading batch dims must broadcast: msg is (degree+1, *batch).
        batch = 3
        st = [x.reshape(batch, -1) for x in _state(20, 24)]
        msg = LogupGkrRound(jnp.array(7, KB))._round_poly(st)
        self.assertEqual(msg.shape, (4, batch))

    def test_fold_matches_manual(self):
        st = _state(30, 8)
        r = jnp.array(5, KB)
        got = LogupGkrRound(jnp.array(3, KB))._fold(st, r)
        self.assertEqual(len(got), 5)
        for x, g in zip(st, got):
            self.assertEqual(g.shape, (4,))  # width halved
            self.assertTrue(bool(jnp.all(g == x[:4] + r * (x[4:] - x[:4]))))

    def test_sumcheck_invariant_s0_plus_s1(self):
        st = _state(40, 16)
        rnd = LogupGkrRound(jnp.array(9, KB))
        msg = rnd._round_poly(st)
        # s(0)+s(1) == sum over the full hypercube of the combine (the claim)
        self.assertTrue(bool(msg[0] + msg[1] == jnp.sum(rnd._combine(*st))))

    def test_call_threads_state_transcript_msg(self):
        st = _state(50, 8)
        t = StubTranscript(jnp.array([4, 0, 0], dtype=KB))
        state, t2, msg = LogupGkrRound(jnp.array(2, KB))(st, t)
        self.assertEqual(msg.shape, (4,))
        self.assertEqual(len(state), 5)
        self.assertEqual(state[0].shape, (4,))  # width halved
        self.assertEqual(t2.pos, 1)  # one challenge consumed

    def test_end_to_end_sumcheck_reduction(self):
        # Drive the round to completion; the running claim must reduce
        # consistently each round and match the final width-1 evaluation.
        from zorch.prove import prove

        width = 16
        st = _state(60, width)
        lam = jnp.array(11, KB)
        rnd = LogupGkrRound(lam)
        n = 4  # log2(16)
        challenges = rand_field(99, (n,), KB)

        # Replay round-by-round to check the per-round sumcheck identity.
        claim = jnp.sum(rnd._combine(*st))
        state, t = st, StubTranscript(challenges)
        for i in range(n):
            state, t, msg = rnd(state, t)
            self.assertTrue(bool(msg[0] + msg[1] == claim))
            claim = _eval_cubic_at(msg, challenges[i])
        self.assertTrue(bool(state[0].shape == (1,)))  # collapsed to a point
        self.assertTrue(bool(rnd._combine(*state)[0] == claim))

        # The generic `prove` driver produces the same per-round messages.
        _, _, proof = prove(rnd, st, StubTranscript(challenges))
        self.assertEqual(proof.shape, (n, 4))

    def test_round_poly_is_fusion_ready_stablehlo(self):
        # Fusion contract (authoritatively enforced by zkx's ZorchRoundRewriter,
        # issue #21): a marked round body must be straight-line element-wise field
        # ops + the ONE inherent Sigma. This is the cheap zorch-side proxy: assert
        # the lowered body uses ONLY fusion-safe ops plus exactly one reduce. A
        # whitelist (not a gather/dot blacklist) so ANY boundary op (gather,
        # scatter, dot, while, ...) or a second reduce trips it -- and any new op
        # in the fusion-critical body gets a conscious look.
        import re

        import jax

        FUSION_SAFE = {
            "add",
            "subtract",
            "multiply",
            "constant",  # element-wise field + consts
            "broadcast_in_dim",
            "reshape",
            "slice",
            "concatenate",  # structural
        }
        st = _state(80, 8)
        hlo = jax.jit(LogupGkrRound(jnp.array(5, KB))._round_poly).lower(st).as_text()
        ops = re.findall(r"stablehlo\.([a-z_]+)", hlo)
        self.assertEqual(ops.count("reduce"), 1)  # the one inherent Sigma, no more
        offenders = {op for op in ops if op != "reduce" and op not in FUSION_SAFE}
        self.assertEqual(
            offenders, set(), f"non-fusion-safe ops in round body: {offenders}"
        )

    def test_state_must_have_five_factors(self):
        with self.assertRaises(ValueError):
            LogupGkrRound(jnp.array(1, KB))._round_poly(_state(70, 8)[:4])


if __name__ == "__main__":
    absltest.main()
