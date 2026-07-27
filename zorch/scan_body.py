# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Memoization for `lax.scan` bodies built outside a `@jit` zone.

`lax.scan` keys its trace cache on the identity of the function it is handed. A
body written as a `def` inside the calling function is therefore a new object on
every call, and the scan re-traces and recompiles an identical graph each time —
invisible in results, and measured here at two to three orders of magnitude over
the work being scanned.

`jit` has the same rule, which is why a scan inside a `@jit` zone needs nothing:
the jit cache absorbs it. This is for the eager paths, where the scan is the
outermost compiled thing. Note that `functools.partial` does *not* substitute:
`jit` sees through a partial to the wrapped function and its bound arguments,
`lax.scan` does not.

Decorate a factory that returns the body, keyed on whatever the body closes
over. Those arguments must hash by value, or equal configurations will each get
their own trace — frozen dataclasses of static config do this for free:

    @scan_body
    def _step(config):
        def step(carry, x): ...
        return step

    lax.scan(_step(config), init, xs)
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any, TypeVar

_Body = TypeVar("_Body", bound=Callable[..., Any])


def scan_body(factory: _Body) -> _Body:
    """Memoize a scan-body factory so repeated calls reuse one traced body."""
    # Unbounded: the keys are protocol configurations, of which a process builds
    # a handful. Bounding it would silently reintroduce the recompile on
    # eviction, which is the failure this exists to prevent.
    return lru_cache(maxsize=None)(factory)  # type: ignore[return-value]
