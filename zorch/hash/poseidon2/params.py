"""Poseidon2Params — the fully-free parameter surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array


def default_external_matrix(width: int, dtype: Any) -> Array:
    """Standard Poseidon2 external matrix for width >= 8 (M4-circulant + 2x block).

    M[i][j] = M4[i%4][j%4] * (2 if same 4-block). Built list -> jnp.array so
    HLO sees a kConstant. Determined wholly by (width, dtype); carries no
    field/scheme identity. `width` must be a positive multiple of 4. (At width == 4
    the canonical Poseidon2 external matrix is plain M4; the 2x-diagonal-block form
    here matches references for width >= 8.)
    """
    if width % 4 != 0:
        raise ValueError(f"external matrix default needs width % 4 == 0, got {width}")
    m4 = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
    mds = [
        [m4[i % 4][j % 4] * (2 if i // 4 == j // 4 else 1) for j in range(width)]
        for i in range(width)
    ]
    return jnp.array(mds, dtype=dtype)


@dataclass(frozen=True)
class Poseidon2Params:
    """Fully-free parameter surface of a Poseidon2 permutation.

    The core treats `dtype` as opaque and names no field/scheme/zkVM. The
    internal layer is the rank-one family `internal_j_scale*J +
    Diag(internal_diag)` — the J coefficient is a free field constant just
    like the diag vector (reference kernels that fold a storage constant,
    e.g. a Montgomery R^-1, into the layer land in this slot).

    Convention baked into this implementation (not yet a parameter): the partial
    round acts on lane 0, so `internal_constants` must be zero in lanes 1..w-1.

    Contract (validated in __post_init__):
      external_matrix : (width, width) over dtype; applied as `external_matrix @
          state`, so any matrix works. Defaults to the canonical M4-circulant.
      external_constants_initial/terminal : (external_rounds, width)
      internal_constants : (internal_rounds, width); RC in lane 0, zeros elsewhere
      internal_diag : (width,)
      internal_j_scale : scalar over dtype; None normalizes to one
      alpha : positive S-box exponent; caller guarantees gcd(alpha, p-1)==1
          (the core does not know p, so it cannot check).
    """

    width: int
    dtype: Any
    alpha: int
    external_rounds: int
    internal_rounds: int
    external_constants_initial: Array
    external_constants_terminal: Array
    internal_constants: Array
    internal_diag: Array
    external_matrix: Array | None = None
    internal_j_scale: Array | None = None

    def __post_init__(self) -> None:
        if self.alpha < 1:
            raise ValueError(f"alpha must be a positive int, got {self.alpha}")
        w = self.width
        if self.external_matrix is None:
            object.__setattr__(
                self, "external_matrix", default_external_matrix(w, self.dtype)
            )
        if self.internal_j_scale is None:
            object.__setattr__(self, "internal_j_scale", jnp.array(1, dtype=self.dtype))
        checks = {
            "external_matrix": ((w, w), self.external_matrix),
            "external_constants_initial": (
                (self.external_rounds, w),
                self.external_constants_initial,
            ),
            "external_constants_terminal": (
                (self.external_rounds, w),
                self.external_constants_terminal,
            ),
            "internal_constants": ((self.internal_rounds, w), self.internal_constants),
            "internal_diag": ((w,), self.internal_diag),
            "internal_j_scale": ((), self.internal_j_scale),
        }
        for name, (want, arr) in checks.items():
            got = tuple(np.shape(arr))
            if got != want:
                raise ValueError(f"{name}: expected shape {want}, got {got}")
            if arr.dtype != self.dtype:
                raise ValueError(
                    f"{name}: expected dtype {self.dtype}, got {arr.dtype}"
                )
        if w > 1 and not np.all(
            np.asarray(self.internal_constants[:, 1:]) == self.dtype(0)
        ):
            raise ValueError(
                "internal_constants lanes 1..w-1 must be zero (lane-0 partial round)"
            )

    @property
    def uses_standard_external_matrix(self) -> bool:
        """Whether `external_matrix` is the standard M4-circulant default.

        zkx's params-driven Poseidon2Fusion GPU emitter hardcodes this matrix, so
        it is the gate for taking that dedicated route over the generic fallback.
        """
        if self.width % 4 != 0:
            return False
        return bool(
            jnp.array_equal(
                self.external_matrix,
                default_external_matrix(self.width, self.dtype),
            )
        )
