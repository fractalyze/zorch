# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""KoalaBear width-16 Poseidon2 parameters — SP1's vendored prover instance.

zorch ships the agnostic Poseidon2 engine but no named parameterizations: a
named instance is a consumer concern. This is the form SP1's shipped prover
(the vendored ``libsp1_gpu_jax_ffi`` kernels, and every hash byte in its
trace/proof dumps) actually computes, pinned to
https://github.com/fractalyze/sp1/blob/cc4eb1f38/slop/crates/koala-bear/src/koala_bear_poseidon2.rs

- round constants: SP1's RC16 array (canonical u32 below);
- internal diagonal: the powers-of-two family — distinct from the
  BabyBear-structured koalabear diagonal in Plonky3 4318eba
  (https://github.com/Plonky3/Plonky3/blob/4318eba/koala-bear/src/poseidon2.rs);
  the two parameterizations hash differently, and byte-matching SP1 is this
  repo's charter;
- the kernel's ``DiffusionMatrixKoalaBear::permute_mut`` computes
  ``R^-1 * M * state``, so R^-1 (R = 2^32) is folded into the internal diag
  and carried as the engine's ``internal_j_scale``.
"""

from __future__ import annotations

import frx.numpy as jnp
import numpy as np
from zk_dtypes import koalabear_mont as F

from zorch.hash.poseidon2.params import Poseidon2Params

_WIDTH, _ER, _IR, _ALPHA = 16, 4, 20, 3

_EXTERNAL_INITIAL = [
    [
        2128964168,
        288780357,
        316938561,
        2126233899,
        426817493,
        1714118888,
        1045008582,
        1738510837,
        889721787,
        8866516,
        681576474,
        419059826,
        1596305521,
        1583176088,
        1584387047,
        1529751136,
    ],
    [
        1863858111,
        1072044075,
        517831365,
        1464274176,
        1138001621,
        428001039,
        245709561,
        1641420379,
        1365482496,
        770454828,
        693167409,
        757905735,
        136670447,
        436275702,
        525466355,
        1559174242,
    ],
    [
        1030087950,
        869864998,
        322787870,
        267688717,
        948964561,
        740478015,
        679816114,
        113662466,
        2066544572,
        1744924186,
        367094720,
        1380455578,
        1842483872,
        416711434,
        1342291586,
        1692058446,
    ],
    [
        1493348999,
        1113949088,
        210900530,
        1071655077,
        610242121,
        1136339326,
        2020858841,
        1019840479,
        678147278,
        1678413261,
        1361743414,
        61132629,
        1209546658,
        64412292,
        1936878279,
        1980661727,
    ],
]

_EXTERNAL_TERMINAL = [
    [
        1139268644,
        630873441,
        669538875,
        462500858,
        876500520,
        1214043330,
        383937013,
        375087302,
        636912601,
        307200505,
        390279673,
        1999916485,
        1518476730,
        1606686591,
        1410677749,
        1581191572,
    ],
    [
        1004269969,
        143426723,
        1747283099,
        1016118214,
        1749423722,
        66331533,
        1177761275,
        1581069649,
        1851371119,
        852520128,
        1499632627,
        1820847538,
        150757557,
        884787840,
        619710451,
        1651711087,
    ],
    [
        505263814,
        212076987,
        1482432120,
        1458130652,
        382871348,
        417404007,
        2066495280,
        1996518884,
        902934924,
        582892981,
        1337064375,
        1199354861,
        2102596038,
        1533193853,
        1436311464,
        2012303432,
    ],
    [
        839997195,
        1225781098,
        2011967775,
        575084315,
        1309329169,
        786393545,
        995788880,
        1702925345,
        1444525226,
        908073383,
        1811535085,
        1531002367,
        1635653662,
        1585100155,
        867006515,
        879151050,
    ],
]

_INTERNAL_RC = [
    1423960925,
    2101391318,
    1915532054,
    275400051,
    1168624859,
    1141248885,
    356546469,
    1165250474,
    1320543726,
    932505663,
    1204226364,
    1452576828,
    1774936729,
    926808140,
    1184948056,
    1186493834,
    843181003,
    185193011,
    452207447,
    510054082,
]

# (d_i - 1) form; -2 is canonical p-2. Powers-of-two family per SP1's kernel.
_INTERNAL_DIAG = [
    2130706431,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    32768,
]

# R^-1 mod p for R = 2^32 — the kernel's permute_mut computes R^-1 * M * state.
_MONTY_INVERSE = 0x3F010000


def koalabear16_params() -> Poseidon2Params:
    """SP1's koalabear width-16 Poseidon2 parameters."""
    internal_rc = np.zeros((_IR, _WIDTH), dtype=np.int64)
    internal_rc[:, 0] = np.array(_INTERNAL_RC, dtype=np.int64)
    monty_inverse = jnp.array(_MONTY_INVERSE, dtype=F)
    return Poseidon2Params(
        width=_WIDTH,
        dtype=F,
        alpha=_ALPHA,
        external_rounds=_ER,
        internal_rounds=_IR,
        external_constants_initial=jnp.array(_EXTERNAL_INITIAL, dtype=F),
        external_constants_terminal=jnp.array(_EXTERNAL_TERMINAL, dtype=F),
        internal_constants=jnp.array(internal_rc, dtype=F),
        internal_diag=jnp.multiply(jnp.array(_INTERNAL_DIAG, dtype=F), monty_inverse),
        internal_j_scale=monty_inverse,
    )
