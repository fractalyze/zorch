# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Package-integrity guards for accidents tooling keeps reintroducing."""

from absl.testing import absltest

import zorch


class PackageTest(absltest.TestCase):
    def test_version_attr_survives(self) -> None:
        # dev-release stamps zorch.__version__; an emptied zorch/__init__.py
        # (it has happened three times: #463, e5b5305, and a linter sweep)
        # breaks the wheel build only at release time. Fail here instead.
        self.assertTrue(getattr(zorch, "__version__", ""))


if __name__ == "__main__":
    absltest.main()
