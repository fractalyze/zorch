# zorch/commit/pcs_test.py
"""Pcs Protocol — structural conformance smoke test."""
from __future__ import annotations

from typing import Any

from absl.testing import absltest

from zorch.commit.pcs import Pcs


class _Dummy:
    def commit(self, mle: Any) -> tuple[None, None]:
        return None, None

    def open(self, prover_data: Any, point: Any, transcript: Any) -> None:
        raise NotImplementedError

    def verify(
        self, commitment: Any, point: Any, value: Any, proof: Any, transcript: Any
    ) -> None:
        raise NotImplementedError


class PcsProtocolTest(absltest.TestCase):
    def test_runtime_checkable_conformance(self) -> None:
        self.assertIsInstance(_Dummy(), Pcs)

    def test_missing_method_fails_conformance(self) -> None:
        class _Partial:
            def commit(self, mle: Any) -> tuple[None, None]:
                return None, None

        self.assertNotIsInstance(_Partial(), Pcs)


if __name__ == "__main__":
    absltest.main()
