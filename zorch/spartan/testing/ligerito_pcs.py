# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""A Ligerito-backed PCS for the Spartan glue, bridging base↔extension field.

The toy R1CS runs over the base field, but Ligerito (a Reed-Solomon matrix code)
commits over an extension. This adapter embeds the base witness and opening point
into the extension for `LigeritoProver`/`Verifier` and projects the opened value
back to the base field, so the `WitnessOpen*` glue stays field-pure and drives
the real recursive PCS unchanged through the `zorch.pcs.protocol` seam. (A
production Spartan over the extension field would inject `LigeritoProver`/
`Verifier` directly, without this bridge.)
"""

from __future__ import annotations

from collections.abc import Sequence

import frx.numpy as fnp
from frx import Array
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.testing.koalabear16 import koalabear16_merkle
from zorch.pcs.ligerito.config import LigeritoConfig, LigeritoProof
from zorch.pcs.ligerito.prover import LigeritoProver, LigeritoProverData
from zorch.pcs.ligerito.verifier import LigeritoVerifier
from zorch.transcript import Transcript

_K = fnp.dtype(EF).itemsize // fnp.dtype(F).itemsize


def embed(x: Array) -> Array:
    """Embed a base-field array into the extension (value in the low limb)."""
    lanes = fnp.zeros((*x.shape, _K), F).at[..., 0].set(x)
    return lanes.view(EF).reshape(x.shape)


def project(y: Array) -> Array:
    """Project a subfield-valued extension array back to the base field (low
    limb); the high limbs are zero for a value that lives in the base subfield."""
    return y.view(F).reshape(*y.shape, _K)[..., 0]


def _make_code(message_len: int, log_inv_rate: int) -> ReedSolomon:
    return ReedSolomon(message_len=message_len, blowup=1 << log_inv_rate, dtype=EF)


class LigeritoSpartanProver:
    """`PcsProver` bridging the base-field witness to `LigeritoProver`."""

    def __init__(self, config: LigeritoConfig) -> None:
        _, _, tree = koalabear16_merkle()
        self._inner = LigeritoProver(_make_code, tree, config)

    def commit(self, polys: Sequence[Array]) -> tuple[Array, LigeritoProverData]:
        return self._inner.commit([embed(polys[0])])

    def open(
        self,
        prover_data: LigeritoProverData,
        points: Sequence[Array],
        transcript: Transcript,
    ) -> tuple[Array, LigeritoProof, Transcript]:
        value, proof, transcript = self._inner.open(
            prover_data, [embed(points[0])], transcript
        )
        return project(fnp.reshape(value, (1,))), proof, transcript


class LigeritoSpartanVerifier:
    """`PcsVerifier` dual of `LigeritoSpartanProver`."""

    def __init__(self, config: LigeritoConfig) -> None:
        _, _, tree = koalabear16_merkle()
        self._inner = LigeritoVerifier(_make_code, tree, config)

    def verify(
        self,
        commitment: Array,
        points: Sequence[Array],
        values: Array,
        proof: LigeritoProof,
        transcript: Transcript,
    ) -> tuple[Array, Transcript]:
        return self._inner.verify(
            commitment, [embed(points[0])], embed(values)[0], proof, transcript
        )
