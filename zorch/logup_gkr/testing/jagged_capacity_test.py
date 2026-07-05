# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Capacity-layout jagged proving: byte-identical to the exact layout, and
compile-key-stable across layers that differ only in true row counts.

`pad_layer_to_capacity` re-stores a layer with each segment extended to a
power-of-two capacity by fold-neutral rows. Two properties gate the design:

- **Byte-identity.** The neutral fraction is a fixed point of the fold and the
  eq corrections are mass-based, so the capacity prove must reproduce the
  exact-layout prove bit for bit -- round polynomials, bound point, openings,
  and the advanced transcript.
- **Key stability.** Same capacity tuple + different true row counts must hit
  the whole-layer compile cache: every plane and schedule shape derives from
  capacities alone, so proving a second layout may not re-trace the round zone.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
from absl.testing import absltest, parameterized
from jax import Array

from zorch.logup_gkr.circuit import JaggedGkrLayer
from zorch.logup_gkr.jagged_prover import (
    JaggedLayerProof,
    _jagged_round_via_zone,
    _jagged_round_zone,
    pad_layer_to_capacity,
    prove_jagged_layer,
)
from zorch.logup_gkr.testing import random_jagged_layer, virtual_planes
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.testkit.random_field import rand_field
from zorch.testkit.transcript import cheap_transcript
from zorch.transcript import Transcript

KB = zk_dtypes.koalabear_mont

# Layouts sharing one capacity tuple: odd/even/saturated segments, true counts
# drifting by 1-2 rows (the cross-shard jitter capacity must absorb).
LAYOUTS = ((3, 1, 5, 2), (3, 2, 5, 1), (4, 1, 6, 2))
CAPACITIES = (4, 2, 8, 2)
# The memory-tight variant: capacities at the running per-segment max, not
# power-of-two, so the capacity schedule itself folds through odd counts (the
# per-round re-pad gathers the pow2 tuple never exercises).
TIGHT_CAPACITIES = (4, 2, 6, 2)
NRV = 3  # 2^NRV = 8 embeds the max capacity
NIV = 2  # 4 segments


def _u32(x: Array) -> list[int]:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint32)).reshape(-1).tolist()


def _leaves_eq(a: object, b: object) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(_u32(x) == _u32(y) for x, y in zip(la, lb))


def _virtual_claim(layer: JaggedGkrLayer, lam: Array, z: Array) -> Array:
    """Brute-force `sum_x eq(z, x) * combine(x)` over the virtual hypercube."""
    n0, n1, d0, d1 = virtual_planes(layer, NRV)
    eq = expand_eq_to_hypercube(z, jnp.ones((), z.dtype))
    return jnp.sum(eq * (lam * (n0 * d1 + n1 * d0) + d0 * d1))


def _prove(
    layer: JaggedGkrLayer, seed: int
) -> tuple[Array, Transcript, JaggedLayerProof]:
    lam = rand_field(seed + 10, (), KB)
    z = rand_field(seed + 11, (NRV + NIV,), KB)
    claim = _virtual_claim(layer, lam, z)
    return prove_jagged_layer(layer, lam, claim, z, cheap_transcript(KB))


class CapacityByteMatchTest(parameterized.TestCase):
    """The capacity prove reproduces the exact-layout prove bit for bit."""

    @parameterized.product(
        seed_counts=tuple(enumerate(LAYOUTS)),
        capacities=(CAPACITIES, TIGHT_CAPACITIES),
    )
    def test_matches_exact_layout(
        self, seed_counts: tuple[int, tuple[int, ...]], capacities: tuple[int, ...]
    ) -> None:
        seed, counts = seed_counts
        layer = random_jagged_layer(seed, counts)
        padded = pad_layer_to_capacity(layer, capacities)
        point, transcript, proof = _prove(layer, seed)
        point_c, transcript_c, proof_c = _prove(padded, seed)
        self.assertTrue(_leaves_eq(point, point_c), "bound point diverged")
        self.assertTrue(_leaves_eq(proof, proof_c), "proof diverged")
        # The advanced transcripts must be in the same state: sample once.
        _, s = transcript.sample(1)
        _, s_c = transcript_c.sample(1)
        self.assertTrue(_leaves_eq(s, s_c), "transcript state diverged")

    def test_noop_at_exact_capacity(self) -> None:
        counts = CAPACITIES
        layer = random_jagged_layer(9, counts)
        padded = pad_layer_to_capacity(layer, CAPACITIES)
        self.assertTrue(_leaves_eq(layer.numerator_0, padded.numerator_0))

    def test_rejects_undersized_capacity(self) -> None:
        layer = random_jagged_layer(0, LAYOUTS[0])
        with self.assertRaises(ValueError):
            pad_layer_to_capacity(layer, (2, 2, 8, 2))  # cap < count


class CapacityKeyStabilityTest(absltest.TestCase):
    """Same capacities, different true counts: no re-trace of the round zone."""

    def test_zone_trace_reused(self) -> None:
        carries = []
        for seed, counts in enumerate(LAYOUTS):
            layer = pad_layer_to_capacity(random_jagged_layer(seed, counts), CAPACITIES)
            num = rand_field(seed + 20, (), KB)
            den = rand_field(seed + 21, (), KB)
            z = rand_field(seed + 22, (NRV + NIV,), KB)
            carries.append((layer, (num, den, z)))

        # Warm the whole-layer trace on the first layout.
        (layer0, carry0) = carries[0]
        _jagged_round_via_zone(layer0, 1, None, carry0, cheap_transcript(KB))
        zone_traces = _jagged_round_zone._cache_size()

        for layer, carry in carries[1:]:
            _jagged_round_via_zone(layer, 1, None, carry, cheap_transcript(KB))
        self.assertEqual(
            _jagged_round_zone._cache_size(),
            zone_traces,
            "a same-capacity layout re-traced the round zone",
        )


if __name__ == "__main__":
    absltest.main()
