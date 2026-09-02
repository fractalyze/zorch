# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The golden koalabear-16 fixture is shared, and sharing it is safe."""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from zorch.testkit.koalabear16 import (
    koalabear16_params,
    koalabear16_perm,
    koalabear16_scaled_perm,
)


class Koalabear16FixtureTest(absltest.TestCase):
    def test_params_are_shared(self) -> None:
        """One instance per process, not one per call.

        Rebuilding costs six host->device constant transfers plus a
        device->host read in `__post_init__`, and leaves the params' cached
        `_value_key`/`_hash` cold -- which jit dispatch then rebuilds by
        pulling every constant array back to host. A caller building a
        transcript per iteration pays that per iteration
        (fractalyze/zorch#327).
        """
        self.assertIs(koalabear16_params(), koalabear16_params())
        self.assertIs(koalabear16_perm(), koalabear16_perm())

    def test_shared_params_still_key_by_value(self) -> None:
        """Sharing must not be what makes the pytree-aux key work: the params
        ride `DuplexTranscript`'s meta_fields, where an equal-valued rebuild
        has to compare equal or every transcript re-traces its enclosing zone.
        """
        params = koalabear16_params()
        rebuilt = koalabear16_params.__wrapped__()
        self.assertIsNot(params, rebuilt)
        self.assertEqual(params, rebuilt)
        self.assertEqual(hash(params), hash(rebuilt))

    def test_scaled_variant_does_not_disturb_the_shared_base(self) -> None:
        """`koalabear16_scaled_perm` derives from the same cached params via
        `replace`; on a frozen dataclass that builds a new instance, but a
        regression to in-place mutation would silently re-scale every other
        caller's J term.
        """
        before = np.asarray(koalabear16_params().internal_j_scale).copy()
        koalabear16_scaled_perm()
        after = np.asarray(koalabear16_params().internal_j_scale)

        np.testing.assert_array_equal(before, after)
        self.assertEqual(int(fnp.asarray(koalabear16_params().internal_j_scale)), 1)

    def test_permutation_is_unchanged_by_sharing(self) -> None:
        """The fixture's whole job is to be one real permutation to run the
        agnostic engine against, so caching must not touch what it computes.
        """
        perm = koalabear16_perm()
        state = fnp.arange(perm.width, dtype=F)

        first = np.asarray(perm.permute(state))
        second = np.asarray(koalabear16_perm().permute(state))

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    absltest.main()
