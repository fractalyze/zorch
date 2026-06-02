# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from absl.testing import absltest

from zorch.round import Round


class RoundBaseTest(absltest.TestCase):
    def test_call_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            Round()(None, None)


if __name__ == "__main__":
    absltest.main()
