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

import numpy as np
import jax
import jax.numpy as jnp

U32 = jnp.uint32

# Round constants (first 32 bits of the fractional parts of the cube roots of the
# first 64 primes) and initial hash state (sqrt of first 8 primes).
_K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
], dtype=np.uint32)
_H0 = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
], dtype=np.uint32)


_Kd = jnp.asarray(_K)


def _rotr(x, n: int):
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
    padded[:, nblocks * 64 - 8:] = np.frombuffer(np.uint64(bitlen).byteswap().tobytes(), dtype=np.uint8)
    words = padded.reshape(b, nblocks, 16, 4).astype(np.uint32)
    be = (words[..., 0] << 24) | (words[..., 1] << 16) | (words[..., 2] << 8) | words[..., 3]
    return be  # [B, nblocks, 16]


def _compress(state, w16):
    """One block: state [B, 8] (a..h) + message words w16 [B, 16] -> state [B, 8].

    The 64-round compression and the message schedule are fused into ONE
    `fori_loop` carrying a [B, 16] shift-register window: round t uses the oldest
    word `w[:,0]`, appends the freshly-scheduled `w[t+16]`, and shifts. Only
    *static* column slices are used (no dynamic array indexing), so XLA keeps the
    window + a..h fusion-/register-friendly — critical for GPU throughput.
    """
    def round_t(t, carry):
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


def _digest_words(blocks):
    """blocks: uint32 [B, nblocks, 16] -> uint8 [B, 32] big-endian digest."""
    b, nblocks, _ = blocks.shape
    state = jnp.broadcast_to(jnp.asarray(_H0), (b, 8))
    for i in range(nblocks):  # nblocks is static and small
        state = _compress(state, blocks[:, i])
    # Serialize 8 u32 words big-endian -> 32 bytes.
    out = jnp.stack([
        (state >> U32(24)) & U32(0xFF),
        (state >> U32(16)) & U32(0xFF),
        (state >> U32(8)) & U32(0xFF),
        state & U32(0xFF),
    ], axis=-1).astype(jnp.uint8)  # [B, 8, 4]
    return out.reshape(b, 32)


def digest(msg) -> jnp.ndarray:
    """SHA-256 of a batch of equal-length messages. msg: uint8 [B, L] -> [B, 32].

    Byte-identical to the FIPS 180-4 standard per message.
    """
    msg_np = np.asarray(msg, dtype=np.uint8)
    blocks = jnp.asarray(_pad(msg_np))
    return _digest_words(blocks)
