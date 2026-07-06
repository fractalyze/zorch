# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Golden-vector regression guard for the dense LogUp-GKR circuit.

A change to the fold or extract arithmetic moves these hashes. The math was
checked once against an independent prover of the same fractional-sum circuit,
so the pinned values track a verified result, not merely the current code.
"""

from __future__ import annotations

import hashlib

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from zorch.logup_gkr.circuit import build_pyramid, extract_outputs
from zorch.logup_gkr.testing import random_first_layer

_SEED = 42

# (num_batch_variables, num_row_variables) -> SHA256 of the output MLEs.
_GOLDENS = {
    (2, 3): "09f0bbf7f0cc0223e004a83f43c58f656463d2c157f32d627a182602da27fe93",
    (2, 4): "a899f692b4ded58fd8218b37c29515ca3f3086588f5ddc6fa16e46e5e7ddb32b",
    (2, 5): "a1568612f53abfbaed673c49fdfbfdec4fd321a04e7104a62cd0897a9a47413b",
    (3, 4): "918a9179e246a562ee180d2e00f84febddfb53f3ca566de619800c5d06c535c2",
}


def _output_hash(num_batch_variables: int, num_row_variables: int) -> str:
    layer = random_first_layer(_SEED, num_batch_variables, num_row_variables)
    out = extract_outputs(build_pyramid(layer)[-1])
    words = jnp.concatenate([out.numerator, out.denominator])
    return hashlib.sha256(np.array(words.tolist(), dtype="<u4").tobytes()).hexdigest()


class DenseGkrGoldenTest(absltest.TestCase):
    def test_output_hash_matches_golden(self) -> None:
        for (niv, nrv), expected in _GOLDENS.items():
            with self.subTest(num_batch_variables=niv, num_row_variables=nrv):
                self.assertEqual(_output_hash(niv, nrv), expected)


if __name__ == "__main__":
    absltest.main()
