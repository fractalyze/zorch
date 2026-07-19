"""Add-absorb duplex sponge over a Permutation — contract + correctness.

The koalabear-16 poseidon2 is the golden permutation. Each expected output below
replays the add-absorb duplex algorithm step by step (state[:rate] += block, a
permute on rate-block-full / mode-switch, read state[:rate] on squeeze) against
that golden permutation — pinning add-absorb semantics (NOT overwrite), the
rate/capacity split, and the duplex mode transitions. This is a self-anchored
reference; the ark-sponge byte-match lives in the consumer (the accumulation
prover), not here.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.hash.duplex_sponge import DuplexSponge
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm

_WIDTH = 16
_RATE = 8


class DuplexSpongeAbsorbSqueezeTest(absltest.TestCase):
    def test_squeeze_after_absorb_permutes_then_reads_rate(self) -> None:
        perm = koalabear16_perm()
        sp = DuplexSponge(perm, rate=_RATE)
        x = fnp.arange(_RATE, dtype=F)  # exactly one rate block

        sp = sp.absorb(x)
        sp, out = sp.squeeze(_RATE)

        # Absorb adds x into the rate lanes (from zero, += equals the value);
        # the squeeze switches Absorbing -> Squeezing, which permutes first,
        # then reads the rate lanes.
        st = fnp.zeros(_WIDTH, dtype=F).at[:_RATE].add(x)
        expected = perm.permute(st)[:_RATE]
        self.assertTrue(bool(fnp.array_equal(out, expected)))

    def test_absorb_spills_adds_onto_permuted_rate(self) -> None:
        perm = koalabear16_perm()
        sp = DuplexSponge(perm, rate=_RATE)
        x = fnp.arange(_RATE + 4, dtype=F)  # 12: rate(8) + 4, spills one block

        sp = sp.absorb(x)
        sp, out = sp.squeeze(_RATE)

        # Add the first rate elements, permute, then ADD the remaining 4 onto the
        # permuted rate lanes (overwrite would replace them — this pins add-absorb).
        st = fnp.zeros(_WIDTH, dtype=F).at[:_RATE].add(x[:_RATE])
        st = perm.permute(st)
        st = st.at[:4].add(x[_RATE:])
        # squeeze: Absorbing(pos=4) -> Squeezing permutes, then reads the rate.
        expected = perm.permute(st)[:_RATE]
        self.assertTrue(bool(fnp.array_equal(out, expected)))

    def test_squeeze_spills_permutes_then_continues(self) -> None:
        perm = koalabear16_perm()
        sp = DuplexSponge(perm, rate=_RATE)
        sp = sp.absorb(fnp.arange(_RATE, dtype=F))

        sp, out = sp.squeeze(_RATE + 3)  # 11: drains one rate block then 3 more

        # Absorbing -> Squeezing permutes; read the full rate; since the request
        # exceeds the rate, permute again and read 3 more from the fresh rate.
        st = perm.permute(
            fnp.zeros(_WIDTH, dtype=F).at[:_RATE].add(fnp.arange(_RATE, dtype=F))
        )
        st2 = perm.permute(st)
        expected = fnp.concatenate([st[:_RATE], st2[:3]])
        self.assertTrue(bool(fnp.array_equal(out, expected)))

    def test_absorb_after_squeeze_permutes_first(self) -> None:
        perm = koalabear16_perm()
        y = fnp.arange(3, dtype=F) + F(5)
        sp = DuplexSponge(perm, rate=_RATE)
        sp = sp.absorb(fnp.arange(_RATE, dtype=F))  # Absorbing, pos=rate
        sp, _ = sp.squeeze(2)  # direction switch: permute; Squeezing, pos=2
        sp = sp.absorb(y)  # absorb after squeeze: must permute first, pos=0
        sp, out = sp.squeeze(_RATE)

        st = perm.permute(
            fnp.zeros(_WIDTH, dtype=F).at[:_RATE].add(fnp.arange(_RATE, dtype=F))
        )
        # absorb-after-squeeze permutes, then adds y onto the fresh rate at 0.
        st = perm.permute(st).at[:3].add(y)
        # final squeeze: Absorbing -> Squeezing permutes, then reads the rate.
        expected = perm.permute(st)[:_RATE]
        self.assertTrue(bool(fnp.array_equal(out, expected)))

    def test_squeeze_full_rate_after_drain_permutes(self) -> None:
        perm = koalabear16_perm()
        sp = DuplexSponge(perm, rate=_RATE)
        sp = sp.absorb(fnp.arange(_RATE, dtype=F))
        sp, a = sp.squeeze(_RATE)  # drains the whole rate; pos == rate
        sp, b = sp.squeeze(_RATE)  # pos == rate: must permute, not re-read stale rate

        st = perm.permute(
            fnp.zeros(_WIDTH, dtype=F).at[:_RATE].add(fnp.arange(_RATE, dtype=F))
        )
        st2 = perm.permute(st)
        self.assertTrue(bool(fnp.array_equal(a, st[:_RATE])))
        self.assertTrue(bool(fnp.array_equal(b, st2[:_RATE])))
        self.assertFalse(bool(fnp.array_equal(b, a)))


class DuplexSpongeContractTest(absltest.TestCase):
    def test_has_dedicated_fusion_delegates_to_permutation(self) -> None:
        # The construction names no hash; it forwards the permutation's flag so a
        # region consumer can gate whole-region wrapping on it (mirrors Sponge).
        perm = koalabear16_perm()
        sp = DuplexSponge(perm, rate=_RATE)
        self.assertEqual(sp.has_dedicated_fusion, perm.has_dedicated_fusion)

    def test_rate_not_less_than_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            DuplexSponge(koalabear16_perm(), rate=_WIDTH)

    def test_rate_below_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            DuplexSponge(koalabear16_perm(), rate=0)

    def test_absorb_non_1d_raises(self) -> None:
        sp = DuplexSponge(koalabear16_perm(), rate=_RATE)
        with self.assertRaises(ValueError):
            sp.absorb(fnp.arange(_RATE, dtype=F).reshape(2, 4))  # 2-D, not 1-D

    def test_squeeze_negative_raises(self) -> None:
        # n is a static Python int; a negative would silently push pos negative
        # and corrupt later reads (ark's usize cannot express it). Fail loud.
        with self.assertRaises(ValueError):
            DuplexSponge(koalabear16_perm(), rate=_RATE).squeeze(-1)

    def test_absorb_filling_rate_defers_permute_to_next_absorb(self) -> None:
        perm = koalabear16_perm()
        a = fnp.arange(_RATE, dtype=F)
        b = fnp.arange(3, dtype=F) + F(7)
        sp = DuplexSponge(perm, rate=_RATE)
        sp = sp.absorb(a)  # fills the rate exactly; permute is deferred (pos == rate)
        sp = sp.absorb(b)  # next absorb permutes first, then adds b at rate 0
        sp, out = sp.squeeze(_RATE)

        st = (
            fnp.zeros(_WIDTH, dtype=F).at[:_RATE].add(a)
        )  # first absorb does NOT permute
        st = perm.permute(st).at[:3].add(b)
        expected = perm.permute(st)[:_RATE]
        self.assertTrue(bool(fnp.array_equal(out, expected)))

    def test_empty_absorb_is_noop(self) -> None:
        # ark-sponge returns before touching state on empty input — an empty
        # absorb must NOT trigger the squeeze->absorb direction-switch permute.
        perm = koalabear16_perm()
        sp = DuplexSponge(perm, rate=_RATE).absorb(fnp.arange(_RATE, dtype=F))
        sp, _ = sp.squeeze(2)  # Squeezing, pos=2
        _, got = sp.absorb(fnp.zeros(0, dtype=F)).squeeze(2)
        _, expected = sp.squeeze(2)  # same as if the empty absorb never happened
        self.assertTrue(bool(fnp.array_equal(got, expected)))

    def test_squeeze_vmap_matches_unbatched(self) -> None:
        perm = koalabear16_perm()

        def run(x: fnp.ndarray) -> fnp.ndarray:
            sp = DuplexSponge(perm, rate=_RATE).absorb(x)
            _, out = sp.squeeze(_RATE)
            return out

        a = fnp.arange(_RATE, dtype=F)
        b = fnp.arange(_RATE, dtype=F) + F(3)
        batched = frx.vmap(run)(fnp.stack([a, b]))
        self.assertTrue(bool(fnp.array_equal(batched[0], run(a))))
        self.assertTrue(bool(fnp.array_equal(batched[1], run(b))))


if __name__ == "__main__":
    absltest.main()
