# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Host `ByteHash` oracles for tests — `hashlib` SHA-256 and the `blake3` binding.

`ByteHashTranscript` is parameterized by a `ByteHash`, and its tests pin the
byte-exactness of the device rows by running the same transcript twice: once over
the device row (`Sha256`, `Blake3`) and once over an independent implementation of
the same standard. These are that independent side.

They live here, and not in hash-frx, because every row hash-frx ships is a device
row (fractalyze/hash-frx#324): it takes a tracer and returns an `Array`, so the
package has one answer to "may this call sit inside a traced region" rather than
two. A host digest is the caller's to make — `hashlib`, and for BLAKE3, the one
family the standard library does not carry, the `blake3` binding. zorch is such a
caller twice over: the byte transcript is host-shaped by construction (a `bytes`
buffer read back per squeeze), and its suites need an oracle that shares no code
with the row under test.

So these are **not** a second implementation of the seam for production use. They
are the differential partner, and `blake3` is a test dependency only. It is
declared in `requirements.in`, which feeds `@zorch_pip` and is the only one of the
requirements files a bazel target can reach; what keeps it out of the `pyzorch`
wheel is `pyproject.toml`, which declares the runtime dependencies and does not
name it.

`fusion_path` is GENERIC: one call is a Python loop, not one device unit, so
`is_one_kernel` is False and `ByteHashTranscript` grinds against them one nonce at
a time — the sequential early-exit that beats a device dispatch per squeeze.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from hash_frx.fusion import FusionPath

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer
    from frx.typing import ArrayLike
    from hash_frx.byte_hash import ByteHash

_SHA256_DIGEST_LEN = 32
_BLAKE3_DIGEST_LEN = 32


def _host_digest(
    hash_one: Callable[[ReadableBuffer], bytes], digest_size: int, msg: ArrayLike
) -> np.ndarray:
    """The body both rows share: `hash_one` per message, uint8 `[B, L]` ->
    uint8 `[B, digest_size]`.

    `hash_one` reaches the contiguous row itself rather than a `tobytes()` copy —
    every hash these rows wrap takes the buffer protocol. Reading the message
    bytes is what makes this a host call: `msg` can never be a tracer.
    """
    rows = np.ascontiguousarray(np.asarray(msg, dtype=np.uint8))  # [B, L]
    out = np.empty((rows.shape[0], digest_size), dtype=np.uint8)
    for i, row in enumerate(rows):
        out[i] = np.frombuffer(hash_one(row), dtype=np.uint8)
    return out


class HostSha256:
    """`ByteHash` for host SHA-256 (FIPS 180-4), looping `hashlib` per message."""

    digest_size = _SHA256_DIGEST_LEN
    fusion_path = FusionPath.GENERIC

    def digest(self, msg: ArrayLike) -> np.ndarray:
        return _host_digest(
            lambda row: hashlib.sha256(row).digest(), self.digest_size, msg
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self))


class HostBlake3:
    """`ByteHash` for host BLAKE3 in hash mode, read out to `output_size` bytes.

    BLAKE3 is an XOF, so `output_size` is free rather than fixed at 32 — the
    transcript's squeeze reads whatever width it asked for. Two instances compare
    on it, the rule the seam carries for every implementation: a row held as
    pytree aux that compares by identity re-traces its enclosing zone on every
    freshly built instance.
    """

    fusion_path = FusionPath.GENERIC

    def __init__(self, output_size: int = _BLAKE3_DIGEST_LEN) -> None:
        if output_size < 1:
            raise ValueError(f"output_size must be at least 1, got {output_size}")
        self.digest_size = output_size

    def digest(self, msg: ArrayLike) -> np.ndarray:
        # Imported here, not at module scope, because `zorch.testkit` ships in
        # the wheel (only `*/testing` is excluded) while `blake3` is not a wheel
        # dependency. At module scope, `import zorch.testkit.byte_hash` would
        # fail for an installed consumer reaching for `HostSha256` alone.
        import blake3

        return _host_digest(
            lambda row: blake3.blake3(row).digest(self.digest_size),
            self.digest_size,
            msg,
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.digest_size == other.digest_size

    def __hash__(self) -> int:
        return hash((type(self), self.digest_size))


if TYPE_CHECKING:
    # Seam-conformance pins — the rule hash-frx's implementation modules carry.
    _bh_sha256: type[ByteHash] = HostSha256
    _bh_blake3: type[ByteHash] = HostBlake3
