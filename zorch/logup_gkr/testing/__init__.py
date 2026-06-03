# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for LogUp-GKR tests."""

from __future__ import annotations

import zk_dtypes

from zorch.logup_gkr.circuit import GkrLayer
from zorch.testkit.random_field import rand_field

_KB = zk_dtypes.koalabear


def random_first_layer(
    seed: int, num_interaction_variables: int, num_row_variables: int
) -> GkrLayer:
    """A random dense first GKR layer, 2^(int+row) wide per MLE."""
    width = 1 << (num_interaction_variables + num_row_variables)
    return GkrLayer(
        numerator_0=rand_field(seed, (width,), _KB),
        numerator_1=rand_field(seed + 1, (width,), _KB),
        denominator_0=rand_field(seed + 2, (width,), _KB),
        denominator_1=rand_field(seed + 3, (width,), _KB),
        num_interaction_variables=num_interaction_variables,
    )
