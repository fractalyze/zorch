"""The `_hash_body` zone must key on sponge VALUE, never on shape alone.

`Sponge.hash` delegates to one module-level value-keyed jit zone so that
aval-identical emissions share a single cached decomposition. The property
that makes the cache sound is the key: `Sponge.__eq__/__hash__` is
(permutation, rate, out) and `Poseidon2Params`' value key hashes every
round-constant/matrix byte. The adversarial case is two configs that differ
in ONE round constant — identical widths, identical avals, the constants
ride as composite operands so even the traced shapes cannot tell them apart.
A shape-keyed cache would silently serve config A's circuit to config B;
these tests pin the divergence and the sharing rule.
"""

from __future__ import annotations

from dataclasses import replace

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import pfinfo

from zorch.hash import sponge as sponge_mod
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_params
from zorch.hash.sponge import Sponge, SpongeParams


def _u32(a) -> np.ndarray:
    return np.asarray(a).view(np.uint32)


def _rand_bf(seed: int, shape) -> frx.Array:
    key = frx.random.PRNGKey(seed)
    return frx.random.randint(key, shape, 0, pfinfo(F).modulus, dtype=fnp.uint32).view(F)


def _single_constant_variant() -> tuple[Sponge, Sponge]:
    """Two sponges whose params differ in exactly one external round constant
    (perturbed by assignment, not arithmetic — avals stay identical)."""
    params_a = koalabear16_params()
    ec = params_a.external_constants_initial
    assert _u32(ec)[0, 0] != _u32(ec)[0, 1]
    params_b = replace(params_a, external_constants_initial=ec.at[0, 0].set(ec[0, 1]))
    sp = SpongeParams(rate=8, out=8)
    return Sponge(Poseidon2(params_a), sp), Sponge(Poseidon2(params_b), sp)


class ZoneKeyingTest(absltest.TestCase):
    def test_single_round_constant_divergence_and_no_clobber(self):
        sa, sb = _single_constant_variant()
        x = _rand_bf(424242, (11,))
        da1 = _u32(sa.hash(x))
        db = _u32(sb.hash(x))
        da2 = _u32(sa.hash(x))
        self.assertFalse(np.array_equal(da1, db))
        np.testing.assert_array_equal(da1, da2)

    def test_interleaved_configs_inside_one_trace_match_eager(self):
        sa, sb = _single_constant_variant()
        x = _rand_bf(424242, (11,))
        da, db = _u32(sa.hash(x)), _u32(sb.hash(x))

        @frx.jit
        def interleaved(inp):
            return sa.hash(inp), sb.hash(inp), sa.hash(inp)

        ja, jb, ja2 = interleaved(x)
        np.testing.assert_array_equal(_u32(ja), da)
        np.testing.assert_array_equal(_u32(jb), db)
        np.testing.assert_array_equal(_u32(ja2), da)

    def test_value_equal_shares_distinct_config_never_shares(self):
        body = sponge_mod._hash_body
        x = _rand_bf(424242, (11,))
        base = Sponge(Poseidon2(koalabear16_params()), SpongeParams(rate=8, out=8))
        d_base = _u32(base.hash(x))
        size0 = body._cache_size()

        fresh = Sponge(Poseidon2(koalabear16_params()), SpongeParams(rate=8, out=8))
        np.testing.assert_array_equal(_u32(fresh.hash(x)), d_base)
        self.assertEqual(body._cache_size(), size0)

        params = koalabear16_params()
        ect = params.external_constants_terminal
        perturbed = Sponge(
            Poseidon2(
                replace(params, external_constants_terminal=ect.at[1, 2].set(ect[1, 3]))
            ),
            SpongeParams(rate=8, out=8),
        )
        d_perturbed = _u32(perturbed.hash(x))
        self.assertGreater(body._cache_size(), size0)
        self.assertFalse(np.array_equal(d_perturbed, d_base))


if __name__ == "__main__":
    absltest.main()
