# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The LNP challenge space `C ⊆ S_κ^{σ₋₁}` over an injected byte stream.

LNP proofs (eprint 2022/284, §2.7) draw challenges from

    C = { c ∈ S_κ : σ₋₁(c) = c,  ²ᵏ√‖σ₋₁(cᵏ)·cᵏ‖₁ ≤ η },

polynomials of `Z[X]/(X^d + 1)` with coefficients in `[-κ, κ]`, fixed under
the automorphism `σ₋₁ : X ↦ X⁻¹`, whose operator norm (as multiplication on
the ring) is bounded by η — the ℓ1 quantity's 2k-th root upper-bounds it,
tightening as k grows. The invariance is what makes `σ₋₁(z)·z` a quadratic
equation in the challenge for the framework's inner-product layer, and every
nonzero invariant element of `S_κ` is invertible in the partial-split ring
(`q ≡ 5 (mod 8)`, lattice-frx's Lemma-2.6 predicate), which soundness
extraction stands on.

Sampling is a deterministic function of injected transcript bytes, under
exactly the byte-stream contract of lattice-frx's sampler family (bytes /
bytearray / uint8 ndarray; consumption fixed ahead of time; identical bytes,
identical challenge; where the bytes come from is the consumer's choice —
nothing here hashes):

- Each candidate consumes one `uniform_bytes_needed(2κ+1, d/2)`-byte block:
  `d/2` uniform draws, balanced to `[-κ, κ]`, embedded as the invariant
  shape `c_0 .. c_{d/2-1}` = draws, `c_{d/2} = 0`, `c_{d-i} = -c_i`.
- A candidate is accepted iff it is nonzero and `‖σ₋₁(cᵏ)·cᵏ‖₁ ≤ η^{2k}`.
  The check is exact integer arithmetic on purpose — at the Figure-3
  parameter point (d=128, κ=2, η=59, k=32) the product's coefficients run
  hundreds of bits, past any float's mantissa, and a rounded gate would be
  a soundness parameter decided by rounding luck.
- Rejection is paid with a precomputed budget rather than an open-ended
  loop: `ceil(-log2(fail_prob))` candidate blocks, from the stated
  conservative per-candidate rejection bound of 1/2 (the measured rate at
  the Figure-3 point is ~1%, so the first block nearly always decides).
  `challenge_bytes_needed` is the resulting exact byte count, and the
  sampler requires exactly that many bytes.

The candidate blocks keep the uniform sampler's own default stream budget
(`fail_prob` here prices gate rejections; the inner `uniform_from_bytes`
budget prices chunk rejections, a separate and far smaller-probability
failure) — the two knobs are deliberately not conflated.

Host by construction — and exactly where the device boundary sits. zorch's
posture is device-first (see `docs/reference/conventions.md`); this module
is host because the η gate multiplies out `σ₋₁(cᵏ)·cᵏ` over *unreduced* ℤ,
where coefficients reach ~2^318 at the Figure-3 point — past every
fixed-width lane a device carries, while a field dtype would fold the very
magnitude the gate measures back mod q. The gate is Python-bigint
arithmetic (the object arrays are containers for exact ints, not numpy
compute), it runs once per challenge on prover and verifier alike, and it
is the one part that can never trace; no XLA path exists for it, so its
performance lever is algorithmic (candidate count, norm-bound cost), not
codegen. The draws and the invariant embedding, by contrast, are
array-expressible: a device-resident transcript consumer derives them on
device from squeezed words directly (rejection as masked prefix-compaction
over the fixed budget, no byte detour) and crosses to the host only for
the gate. The byte-stream form here is the injection seam's shape — the
host oracle every traced derivation is checked against — and its LE-uint64
chunk convention is lattice-frx's pinned stream contract, not a device
choice.
"""

from __future__ import annotations

import math
import operator
from typing import SupportsIndex

import numpy as np
from lattice_frx.roots import galois_map
from lattice_frx.sampler import uniform_bytes_needed, uniform_from_bytes


def _index_param(name: str, value: SupportsIndex) -> int:
    """`operator.index`, naming the offending parameter. Normalizing (not
    just checking) is load-bearing twice over: a float `eta` would turn the
    exact integer gate into a float comparison, and a numpy integer `eta`
    would silently wrap in `eta ** (2k)` — both soundness parameters decided
    by arithmetic accidents."""
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(
            f"challenge: {name} must be an integer, got {value!r}"
        ) from None


def _require_params(
    d: SupportsIndex,
    kappa: SupportsIndex,
    eta: SupportsIndex,
    k: SupportsIndex,
    fail_prob: float,
) -> tuple[int, int, int, int]:
    """One validation for both public entry points, returning `(d, kappa,
    eta, k)` as genuine Python ints. The failure modes stay distinct in the
    package's usual way: a wrong *kind* of parameter is a `TypeError`
    (`_index_param`), a wrong *value* a `ValueError`."""
    d = _index_param("d", d)
    kappa = _index_param("kappa", kappa)
    eta = _index_param("eta", eta)
    k = _index_param("k", k)
    if d < 4 or d & (d - 1):
        raise ValueError(
            f"challenge: degree must be a power of two >= 4 to carry the "
            f"σ₋₁-invariant shape, got {d!r}"
        )
    if kappa < 1:
        raise ValueError(f"challenge: kappa must be positive, got {kappa!r}")
    if eta < 1:
        raise ValueError(f"challenge: eta must be positive, got {eta!r}")
    if k < 1:
        raise ValueError(f"challenge: k must be positive, got {k!r}")
    if not 0.0 < fail_prob < 1.0:
        raise ValueError(f"challenge: fail_prob must be in (0, 1), got {fail_prob!r}")
    return d, kappa, eta, k


def _attempts(fail_prob: float) -> int:
    """The fixed candidate budget: with the stated per-candidate rejection
    bound of 1/2, `ceil(-log2(fail_prob))` attempts fail together with
    probability at most `fail_prob`."""
    return math.ceil(-math.log2(fail_prob))


def _layout(d: int, kappa: int, fail_prob: float) -> tuple[int, int]:
    """`(attempts, block)` — the candidate budget and one uniform block's
    byte count, defined once so `challenge_from_bytes`'s parser cannot
    drift from the count its companion quotes (the `_ternary_layout`
    discipline of lattice-frx's sampler family)."""
    return _attempts(fail_prob), uniform_bytes_needed(2 * kappa + 1, d // 2)


def _negacyclic_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Negacyclic product over exact Python ints (object arrays): full
    convolution, then `X^d ≡ -1` folds the top half in with a sign."""
    d = a.shape[0]
    full = np.convolve(a, b)
    out = full[:d].copy()
    out[: d - 1] -= full[d:]
    return out


def _negacyclic_pow(base: np.ndarray, k: int) -> np.ndarray:
    """`base^k` (`k >= 1`) by square-and-multiply — seeded from `base`
    itself, so there is no identity polynomial and no identity multiply;
    k=32 is exactly five squarings."""
    if k == 1:
        return base
    half = _negacyclic_pow(base, k >> 1)
    squared = _negacyclic_mul(half, half)
    return _negacyclic_mul(squared, base) if k & 1 else squared


def _sigma_product_l1(c: np.ndarray, k: int) -> int:
    """`‖σ₋₁(cᵏ)·cᵏ‖₁` over `Z[X]/(X^d + 1)`, exact.

    `cᵏ` by `_negacyclic_pow`, `σ₋₁` through
    `lattice_frx.roots.galois_map(d, 2d-1)` — the same index map both host
    rings build their Galois action from — and one final product."""
    d = c.shape[0]
    power = _negacyclic_pow(c.astype(object), k)
    dest, negate = galois_map(d, 2 * d - 1)
    sigma = np.empty(d, dtype=object)
    for i, (j, neg) in enumerate(zip(dest, negate)):
        sigma[j] = -power[i] if neg else power[i]
    return int(np.abs(_negacyclic_mul(sigma, power)).sum())


def _embed(draws: np.ndarray, d: int, kappa: int) -> np.ndarray:
    """Balanced draws to the σ₋₁-invariant coefficient shape (module
    docstring): lower half free, `c_{d/2}` zero, upper half mirrored
    negated."""
    v = draws.astype(np.int64) - kappa
    c = np.zeros(d, dtype=np.int64)
    c[: d // 2] = v
    c[d // 2 + 1 :] = -v[1:][::-1]
    return c


def challenge_bytes_needed(
    d: SupportsIndex,
    kappa: SupportsIndex,
    eta: SupportsIndex,
    k: SupportsIndex,
    fail_prob: float = 2.0**-128,
) -> int:
    """The exact byte count `challenge_from_bytes` consumes for these
    parameters: the candidate budget times one uniform block. Parameters
    ride in the same order as `challenge_from_bytes` minus the stream, as
    with lattice-frx's `*_bytes_needed` companions; `eta` and `k` do not
    move the count but complete the parameter point, so both entry points
    take one tuple."""
    d, kappa, eta, k = _require_params(d, kappa, eta, k, fail_prob)
    attempts, block = _layout(d, kappa, fail_prob)
    return attempts * block


def challenge_from_bytes(
    data: bytes | bytearray | np.ndarray,
    d: SupportsIndex,
    kappa: SupportsIndex,
    eta: SupportsIndex,
    k: SupportsIndex,
    fail_prob: float = 2.0**-128,
) -> np.ndarray:
    """One challenge from `C` as a deterministic function of the injected
    byte stream: the first candidate block (in stream order) that embeds
    to a nonzero polynomial and passes the η gate decides, and the stream
    length must be exactly `challenge_bytes_needed(...)`.

    Returns the length-`d` signed coefficient vector (`int64`, entries in
    `[-κ, κ]`) — host-boundary raw ints, ready for a ring's `from_signed`,
    like lattice-frx's own ternary sampler."""
    d, kappa, eta, k = _require_params(d, kappa, eta, k, fail_prob)
    attempts, block = _layout(d, kappa, fail_prob)
    needed = attempts * block
    if isinstance(data, (bytes, bytearray)):
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
    elif isinstance(data, np.ndarray):
        if data.dtype != np.uint8 or data.ndim != 1:
            raise TypeError(
                f"challenge_from_bytes: byte stream array must be "
                f"one-dimensional uint8 (a stream, not a matrix), got "
                f"dtype={data.dtype} ndim={data.ndim}"
            )
        buf = np.ascontiguousarray(data)
    else:
        raise TypeError(
            f"challenge_from_bytes: byte stream must be bytes, bytearray, or "
            f"a uint8 ndarray, got {type(data).__name__}"
        )
    if buf.size != needed:
        raise ValueError(
            f"challenge_from_bytes: expected exactly {needed} bytes for these "
            f"parameters (the challenge_bytes_needed count), got {buf.size}"
        )

    bound = eta ** (2 * k)
    for i in range(attempts):
        draws = uniform_from_bytes(
            buf[i * block : (i + 1) * block], 2 * kappa + 1, d // 2
        )
        c = _embed(draws, d, kappa)
        if c.any() and _sigma_product_l1(c, k) <= bound:
            return c
    raise RuntimeError(
        f"challenge_from_bytes: every candidate block was rejected by the η "
        f"gate — on honestly random bytes this has probability <= "
        f"{fail_prob!r}, so suspect the stream (or a caller's slicing), not "
        f"bad luck."
    )
