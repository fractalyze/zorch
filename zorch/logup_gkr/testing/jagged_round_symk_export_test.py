# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Symbolic-size export of the jagged sumcheck round byte-matches eager.

The jagged GKR sumcheck runs as a host loop relaunching one shape-polymorphic
round kernel at the round's real (halving) state size — one compiled kernel
dispatched at a shrinking grid each round. This locks both round families: ONE
``jax.export`` binary, symbolic over the round's state size, must produce
byte-identical output to the eager kernel for every size in a bracket.

The kernels are Fiat-Shamir-less pure field arithmetic (the poseidon sponge
stays on the host between launches), so the binary exports portably and needs no
poseidon plugin. The decidability gate is the bracket *declaration*: a halving
dim is declared as a multiple (``2*g``/``4*g``/``2*p``) so every internal
stride-2 stays decidable, and all symbolic dims share ONE scope (a free symbol
leaves the parity inconclusive — ``InconclusiveDimensionOperation`` — and split
scopes are rejected). Mont-u32, no tolerances.

- Dense interaction kernels (``_round_poly_int``/``_fix_and_sum_int``): a single
  halving leading dim, byte-matched against eager over a power-of-2 bracket.
- Jagged row kernels (``_sum_as_poly_row``/``_fix_and_sum_row``): the
  ``_pad_neutral`` segment gather + ``eq_row`` pair lookup, byte-matched against
  eager over two genuinely different jagged layer layouts through one binary.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import Array, export
from zk_dtypes import koalabear_mont as KB
from zk_dtypes import koalabearx4_mont as EF

from zorch.logup_gkr.jagged_prover import (
    _DEGREE,
    _fix_and_sum_int,
    _fix_and_sum_row,
    _InterpConsts,
    _Planes,
    _round_metadata,
    _round_poly_int,
    _round_poly_row,
    _RoundScalars,
)
from zorch.logup_gkr.testing import random_jagged_layer
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import compute_inv_vandermonde

_PRIME = 2013265921  # koalabear
_GMAX = 16


def _u32(x: Array) -> list[int]:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint32)).reshape(-1).tolist()


def _byte_eq(ref: object, got: object) -> bool:
    rl, gl = jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got)
    return len(rl) == len(gl) and all(_u32(a) == _u32(b) for a, b in zip(rl, gl))


class DenseInteractionExportTest(absltest.TestCase):
    """The dense interaction-round kernels: one halving leading dim."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.rng = np.random.default_rng(0)
        cls.consts = _InterpConsts(
            jnp.stack([jnp.array(j, EF) for j in range(_DEGREE + 1)]),
            compute_inv_vandermonde(_DEGREE, EF),
        )

    def _ef(self, shape: tuple[int, ...]) -> Array:
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        return (
            jnp.asarray(self.rng.integers(0, _PRIME, (n, 4), np.uint32))
            .view(EF)
            .reshape(shape)
        )

    def _planes(self, m: int) -> _Planes:
        return _Planes(*(self._ef((m,)) for _ in range(4)))

    def _scalars(self) -> _RoundScalars:
        return _RoundScalars(*(self._ef(()) for _ in range(5)))

    def _assert_one_binary(
        self,
        fn: Callable[..., Any],
        multiple: int,
        has_alpha: bool,
        gs: tuple[int, ...],
    ) -> None:
        consts = self.consts
        (g,) = export.symbolic_shape("g", constraints=[f"g <= {_GMAX}", "g >= 1"])
        sds_m = jax.ShapeDtypeStruct((multiple * g,), EF)
        sds_s = jax.ShapeDtypeStruct((), EF)
        planes_abst = _Planes(sds_m, sds_m, sds_m, sds_m)
        scalars_abst = _RoundScalars(sds_s, sds_s, sds_s, sds_s, sds_s)
        abst: tuple[Any, ...]
        if has_alpha:
            abst = (planes_abst, sds_m, sds_s, scalars_abst)
            exported = export.export(
                jax.jit(lambda p, e, a, s: fn(p, e, a, s, consts))
            )(*abst)
        else:
            abst = (planes_abst, sds_m, scalars_abst)
            exported = export.export(jax.jit(lambda p, e, s: fn(p, e, s, consts)))(
                *abst
            )
        for gg in gs:
            mm = multiple * gg
            planes, eq, scalars = self._planes(mm), self._ef((mm,)), self._scalars()
            args = (
                (planes, eq, self._ef(()), scalars)
                if has_alpha
                else (planes, eq, scalars)
            )
            self.assertTrue(
                _byte_eq(fn(*args, consts), exported.call(*args)), f"m={mm}"
            )

    def test_compute_only_round(self) -> None:
        # _round_poly_int: one stride-2 (in _paired_sums) -> 2*g. operands: planes,
        # eq_int, scalars (eq_adj/pad_adj/z_cur/claim/lam); consts baked.
        self._assert_one_binary(_round_poly_int, 2, has_alpha=False, gs=(1, 8, 16))

    def test_fold_and_sum_round(self) -> None:
        # _fix_and_sum_int: fold then _paired_sums -> two stride-2 -> 4*g. operands:
        # planes, eq_int, alpha, scalars; consts baked.
        self._assert_one_binary(_fix_and_sum_int, 4, has_alpha=True, gs=(1, 4, 16))


class JaggedRowExportTest(absltest.TestCase):
    """The jagged row-variable kernels: gather re-pad + segment-local eq_row.

    Two layouts (different row_counts, same nrv/niv) flow through ONE symbolic
    binary — the recompile-free relaunch the issue targets, over a genuinely
    jagged (non-power-of-2) state size.
    """

    NRV, NIV = 3, 2
    LAYOUTS = ((3, 1, 5, 2), (7, 3, 1, 5))  # both keep odd segments at rounds 0,1

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.consts = _InterpConsts(
            jnp.stack([jnp.array(j, KB) for j in range(_DEGREE + 1)]),
            compute_inv_vandermonde(_DEGREE, KB),
        )

    def _scalars(self, seed: int) -> _RoundScalars:
        rng = np.random.default_rng(seed)
        return _RoundScalars(
            *(
                jnp.asarray(rng.integers(0, _PRIME, (), np.uint32)).view(KB)
                for _ in range(5)
            )
        )

    def _eqs(self, seed: int) -> tuple[Array, Array]:
        rng = np.random.default_rng(seed + 99)
        one = jnp.ones((), KB)
        z_row = jnp.asarray(rng.integers(0, _PRIME, (self.NRV,), np.uint32)).view(KB)
        z_int = jnp.asarray(rng.integers(0, _PRIME, (self.NIV,), np.uint32)).view(KB)
        return expand_eq_to_hypercube(z_row, one), expand_eq_to_hypercube(z_int, one)

    def _round0(
        self, seed: int, counts: tuple[int, ...]
    ) -> tuple[_Planes, Array, Array, Array, Array, Array, _RoundScalars]:
        # _sum_as_poly_row operands: planes, gather, col_index, pair_index, eq_row,
        # eq_int, scalars (consts baked).
        layer = random_jagged_layer(seed, counts)
        gather, col_index, pair_index = _round_metadata(counts, self.NRV)[0]
        self.assertIsNotNone(gather, "layout must need a round-0 re-pad")
        eq_row, eq_int = self._eqs(seed)
        planes = _Planes(
            layer.numerator_0,
            layer.numerator_1,
            layer.denominator_0,
            layer.denominator_1,
        )
        return (
            planes,
            gather,
            col_index,
            pair_index,
            eq_row,
            eq_int,
            self._scalars(seed),
        )

    def _round1(
        self, seed: int, counts: tuple[int, ...]
    ) -> tuple[_Planes, Array, Array, Array, Array, Array, Array, _RoundScalars]:
        # _fix_and_sum_row operands: round 0's padded state + a challenge + meta[1].
        planes0, g0, ci0, pi0, eq_row, eq_int, scal0 = self._round0(seed, counts)
        _poly, planes = _round_poly_row(
            planes0, g0, ci0, pi0, eq_row, eq_int, scal0, self.consts
        )
        alpha = self._scalars(seed + 7).eq_adj
        gather, col_index, pair_index = _round_metadata(counts, self.NRV)[1]
        self.assertIsNotNone(gather, "layout must need a round-1 re-pad")
        return (
            planes,
            eq_row,
            alpha,
            gather,
            col_index,
            pair_index,
            eq_int,
            self._scalars(seed + 3),
        )

    def test_sum_as_poly_row_one_binary(self) -> None:
        consts = self.consts
        # L pre-pad state, 2*p post-pad/gather, p pairs — one scope. eq_row/eq_int
        # static (2^nrv, 2^niv).
        L, p = export.symbolic_shape(
            "L, p", constraints=["L>=4", "L<=64", "p>=1", "p<=32"]
        )
        sds_L, sds_s = jax.ShapeDtypeStruct((L,), KB), jax.ShapeDtypeStruct((), KB)
        abst = (
            _Planes(sds_L, sds_L, sds_L, sds_L),
            jax.ShapeDtypeStruct((2 * p,), jnp.int32),
            jax.ShapeDtypeStruct((p,), jnp.int32),
            jax.ShapeDtypeStruct((p,), jnp.int32),
            jax.ShapeDtypeStruct((1 << self.NRV,), KB),
            jax.ShapeDtypeStruct((1 << self.NIV,), KB),
            _RoundScalars(sds_s, sds_s, sds_s, sds_s, sds_s),
        )
        exported = export.export(
            jax.jit(
                lambda pl, ga, ci, pi, er, ei, sc: _round_poly_row(
                    pl, ga, ci, pi, er, ei, sc, consts
                )
            )
        )(*abst)
        for seed, counts in enumerate(self.LAYOUTS):
            args = self._round0(seed, counts)
            self.assertTrue(
                _byte_eq(_round_poly_row(*args, consts), exported.call(*args)),
                f"sum_as_poly_row diverged at layout {counts}",
            )

    def test_fix_and_sum_row_one_binary(self) -> None:
        consts = self.consts
        # 2*pp input state (even), 2*p post-pad, p pairs — one scope. eq_row static
        # (2^nrv, folds to 2^(nrv-1) inside).
        pp, p = export.symbolic_shape(
            "pp, p", constraints=["pp>=1", "pp<=32", "p>=1", "p<=32"]
        )
        sds_pp, sds_s = (
            jax.ShapeDtypeStruct((2 * pp,), KB),
            jax.ShapeDtypeStruct((), KB),
        )
        abst = (
            _Planes(sds_pp, sds_pp, sds_pp, sds_pp),
            jax.ShapeDtypeStruct((1 << self.NRV,), KB),  # eq_row
            sds_s,  # alpha
            jax.ShapeDtypeStruct((2 * p,), jnp.int32),
            jax.ShapeDtypeStruct((p,), jnp.int32),
            jax.ShapeDtypeStruct((p,), jnp.int32),
            jax.ShapeDtypeStruct((1 << self.NIV,), KB),  # eq_int
            _RoundScalars(sds_s, sds_s, sds_s, sds_s, sds_s),
        )
        exported = export.export(
            jax.jit(
                lambda pl, er, al, ga, ci, pi, ei, sc: _fix_and_sum_row(
                    pl, er, al, ga, ci, pi, ei, sc, consts
                )
            )
        )(*abst)
        for seed, counts in enumerate(self.LAYOUTS):
            args = self._round1(seed, counts)
            self.assertTrue(
                _byte_eq(_fix_and_sum_row(*args, consts), exported.call(*args)),
                f"fix_and_sum_row diverged at layout {counts}",
            )


if __name__ == "__main__":
    absltest.main()
