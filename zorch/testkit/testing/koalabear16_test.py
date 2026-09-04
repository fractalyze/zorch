# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The golden koalabear-16 fixtures are shared, and sharing changes nothing.

Sharing one instance across callers is only safe while a rebuild stays
indistinguishable from the shared one, and while the scaled variant does not
reach back into the base it derives from.
"""

from __future__ import annotations

import frx.numpy as fnp
from absl.testing import absltest

from zorch.testkit.koalabear16 import (
    koalabear16_params,
    koalabear16_perm,
    koalabear16_scaled_perm,
)


class SharedFixtureTest(absltest.TestCase):
    def test_callers_receive_one_instance(self) -> None:
        self.assertIs(koalabear16_params(), koalabear16_params())
        self.assertIs(koalabear16_perm(), koalabear16_perm())
        self.assertIs(koalabear16_scaled_perm(), koalabear16_scaled_perm())

    def test_shared_instance_matches_a_fresh_build(self) -> None:
        # `__wrapped__` is the uncached builder, so this compares what callers
        # get against what they would have built themselves.
        fresh = koalabear16_params.__wrapped__()
        shared = koalabear16_params()
        self.assertEqual(fresh, shared)
        for name in (
            "external_constants_initial",
            "external_constants_terminal",
            "internal_constants",
            "internal_diag",
        ):
            self.assertTrue(
                bool(fnp.all(getattr(fresh, name) == getattr(shared, name)))
            )

    def test_scaled_variant_leaves_the_base_alone(self) -> None:
        # The scaled instance is `replace`d off the shared params. `replace`
        # copies, but a variant that reached in and mutated instead would now
        # be poisoning every other caller of the base.
        self.assertNotEqual(koalabear16_scaled_perm(), koalabear16_perm())
        self.assertEqual(koalabear16_params(), koalabear16_params.__wrapped__())


if __name__ == "__main__":
    absltest.main()
