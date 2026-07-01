# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SHA-256 over uint32 lanes, authored in jax — byte-identical to the FIPS 180-4
standard (and any conforming implementation, e.g. Python's `hashlib.sha256`).

Bulk-parallel by construction: a batch of `B` equal-length messages is hashed in
one data-parallel call (the 64 rounds carry a per-message a..h chain, but every
message in the batch advances independently). That maps the many-independent-hash
workloads — Merkle leaf/internal levels, batched proof-of-work grinding — onto a
GPU's width. A byte hash, unlike the algebraic `Permutation`s in this package
(Poseidon2/Poseidon), so it is a standalone primitive rather than a `Permutation`.

Contract: `digest(msg)` takes uint8 `[B, L]` (a batch of `B` messages, each `L`
bytes) and returns uint8 `[B, 32]` digests, big-endian (standard SHA-256 output
order). Length `L` is static, so padding is data-independent and done once on
host. Requires no x64; all arithmetic is uint32 (wraps mod 2^32 in XLA).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from zorch.fusion import fused_region

U32 = jnp.uint32

SHA256_MARKER = "zorch.sha256"
# Marker revision riding as `composite.version`. zkx recognizes the marker by
# name + attributes and deliberately does not gate on the version; it lets a
# future contract change be staged without renaming the marker (cf. POSEIDON2).
SHA256_MARKER_VERSION = 1

# Round constants (first 32 bits of the fractional parts of the cube roots of the
# first 64 primes) and initial hash state (sqrt of first 8 primes).
_K = np.array(
    [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ],
    dtype=np.uint32,
)
_H0 = np.array(
    [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ],
    dtype=np.uint32,
)


_Kd = jnp.asarray(_K)


def _rotr(x: Array, n: int) -> Array:
    return (x >> U32(n)) | (x << U32(32 - n))


def _pad(msg: np.ndarray) -> np.ndarray:
    """SHA-256 pad a uint8 [B, L] batch -> uint32 [B, nblocks, 16] big-endian words.

    Length is static, so padding is data-independent and done once on host.
    """
    b, length = msg.shape
    bitlen = length * 8
    nblocks = (length + 8) // 64 + 1  # room for the 0x80 byte + 8-byte length
    padded = np.zeros((b, nblocks * 64), dtype=np.uint8)
    padded[:, :length] = msg
    padded[:, length] = 0x80
    padded[:, nblocks * 64 - 8 :] = np.frombuffer(
        np.uint64(bitlen).byteswap().tobytes(), dtype=np.uint8
    )
    words = padded.reshape(b, nblocks, 16, 4).astype(np.uint32)
    be = (
        (words[..., 0] << 24)
        | (words[..., 1] << 16)
        | (words[..., 2] << 8)
        | words[..., 3]
    )
    return be  # [B, nblocks, 16]


def _compress(state: Array, w16: Array) -> Array:
    """One block: state [B, 8] (a..h) + message words w16 [B, 16] -> state [B, 8].

    The 64-round compression and the message schedule are fused into ONE
    `fori_loop` carrying a [B, 16] shift-register window: round t uses the oldest
    word `w[:,0]`, appends the freshly-scheduled `w[t+16]`, and shifts. Only
    *static* column slices are used (no dynamic array indexing), so XLA keeps the
    window + a..h fusion-/register-friendly — critical for GPU throughput.
    """

    def round_t(t: Array, carry: tuple) -> tuple:
        a, b, c, d, e, f, g, h, w = carry
        word = w[:, 0]
        kt = _Kd[t]
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = h + S1 + ch + kt + word
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = S0 + maj
        # schedule next word w[t+16] = sigma1(w14) + w9 + sigma0(w1) + w0
        s0 = _rotr(w[:, 1], 7) ^ _rotr(w[:, 1], 18) ^ (w[:, 1] >> U32(3))
        s1 = _rotr(w[:, 14], 17) ^ _rotr(w[:, 14], 19) ^ (w[:, 14] >> U32(10))
        nxt = w[:, 0] + s0 + w[:, 9] + s1
        w = jnp.concatenate([w[:, 1:], nxt[:, None]], axis=1)
        return (t1 + t2, a, b, c, d + t1, e, f, g, w)

    init = (*(state[:, i] for i in range(8)), w16)
    a, b, c, d, e, f, g, h, _ = jax.lax.fori_loop(0, 64, round_t, init)
    return state + jnp.stack([a, b, c, d, e, f, g, h], axis=1)


# The SHA-256 initial hash state (sqrt of the first 8 primes) as a device array —
# the standard start for a full digest, and the resume point a streaming hash
# broadcasts from.
INITIAL_STATE = jnp.asarray(_H0)  # uint32 [8]


def block_to_words(blocks: Array) -> Array:
    """uint8 [B, nblocks*64] -> uint32 [B, nblocks, 16] big-endian message words.

    The on-device sibling of `_pad`'s host word-packing, for callers that build
    their own already-padded blocks (an incremental / streaming hash), rather than
    padding a whole message once on host.
    """
    b = blocks.shape[0]
    nblocks = blocks.shape[1] // 64
    w = blocks.reshape(b, nblocks, 16, 4).astype(U32)
    return (
        (w[..., 0] << U32(24))
        | (w[..., 1] << U32(16))
        | (w[..., 2] << U32(8))
        | w[..., 3]
    )


def compress(state: Array, blocks_words: Array) -> Array:
    """Fold `blocks_words` (uint32 [B, nblocks, 16] big-endian) into the SHA-256
    midstate `state` (uint32 [B, 8]), block by block. `INITIAL_STATE` broadcast is
    the standard start; a streaming hash resumes from a prior midstate."""
    nblocks = blocks_words.shape[1]
    for i in range(nblocks):  # nblocks is static and small
        state = _compress(state, blocks_words[:, i])
    return state


def serialize_digest(state: Array) -> Array:
    """SHA-256 midstate uint32 [B, 8] -> uint8 [B, 32] big-endian digest."""
    b = state.shape[0]
    out = jnp.stack(
        [
            (state >> U32(24)) & U32(0xFF),
            (state >> U32(16)) & U32(0xFF),
            (state >> U32(8)) & U32(0xFF),
            state & U32(0xFF),
        ],
        axis=-1,
    ).astype(
        jnp.uint8
    )  # [B, 8, 4]
    return out.reshape(b, 32)


def _digest_words(blocks: Array) -> Array:
    """blocks: uint32 [B, nblocks, 16] -> uint8 [B, 32] big-endian digest."""
    b = blocks.shape[0]
    state = jnp.broadcast_to(INITIAL_STATE, (b, 8))
    return serialize_digest(compress(state, blocks))


# Module-level jit zone: `lax.composite` re-traces its decomposition on every
# emission, and one PCS open emits the leaf + every internal level of a Merkle
# commit plus each transcript squeeze — so the uncached re-trace of the 64-round
# body would dominate the first-trace floor (cf. poseidon2._permute_body, #216).
# `inline=True` splices the cached jaxpr into the enclosing trace, so the emitted
# module (one composite marker per digest) is unchanged.
@partial(jax.jit, inline=True)
def _digest_words_marked(blocks: Array) -> Array:
    """`_digest_words`, wrapped in the name-routed `zorch.sha256` composite.

    blocks: uint32 [B, nblocks, 16] -> uint8 [B, 32]. SHA-256 is Merkle-Damgard —
    a 64-round compression over a `fori_loop`, not straight-line — so it takes the
    *name-routed* marker (exempt from the generic single-kernel rule, the way
    `zorch.poseidon2` is) and routes to a dedicated zkx Sha256Fusion emitter. With
    no emitter wired the marker inlines its decomposition, so the bytes are
    unchanged. The emitter reads the block count from the operand's shape.
    """

    def decomposition(b: Array, **_attrs: object) -> Array:
        return _digest_words(b)

    return fused_region(
        decomposition,
        blocks,
        name=SHA256_MARKER,
        version=SHA256_MARKER_VERSION,
    )


def digest(msg: Array) -> jnp.ndarray:
    """SHA-256 of a batch of equal-length messages. msg: uint8 [B, L] -> [B, 32].

    Byte-identical to the FIPS 180-4 standard per message. The device compression
    is emitted as the name-routed `zorch.sha256` marker (host padding stays out of
    the region, since it is static and data-independent).
    """
    msg_np = np.asarray(msg, dtype=np.uint8)
    blocks = jnp.asarray(_pad(msg_np))
    return _digest_words_marked(blocks)
