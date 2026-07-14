# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Test-only assertion that a module-level jit zone is not re-traced.

The PCS jit zones are module-level functions whose static keys compare by
value (#214), so freshly built same-config provers/verifiers must reuse one
trace. ``assert_single_trace`` drives a sequence of calls and asserts the
zone's compile cache gains no entries after the first call.

Snapshot-compare rather than assert an absolute count: a module-level zone is
shared process-wide, so an earlier test may already have seeded this config's
entry. ``_cache_size()`` is a private JAX API that may change on a frx
upgrade; this helper is its single home for the no-retrace assertions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any
from unittest import TestCase


def assert_single_trace(
    test: TestCase, zone: Any, calls: Iterable[Callable[[], object]]
) -> None:
    """Run ``calls`` in order; fail if ``zone`` re-traces after the first."""
    size_after_first: int | None = None
    for call in calls:
        call()
        if size_after_first is None:
            size_after_first = zone._cache_size()
    test.assertEqual(zone._cache_size(), size_after_first)
