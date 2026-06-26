# Transcripts — the two-transcript taxonomy

A Fiat-Shamir transcript turns prover messages into verifier challenges. zorch
carries **two** kinds, by construction-and-substrate, not by scheme:

| | `transcript.py` — `DuplexTranscript` | `byte_transcript.py` — `Sha256Transcript` |
| --- | --- | --- |
| Primitive | algebraic `Permutation` (Poseidon2) over a prime-field dtype | byte hash (SHA-256, FIPS 180-4) |
| I/O | field elements (`Array`); `observe` bitcast-flattens to the base field | opaque `bytes`; the consumer serializes its own field↔bytes |
| Substrate | **device**: `observe`/`sample` are device ops, threadable through `@jit` / a `lax.scan` carry | **host**: a strictly sequential byte chain; a Python orchestrator |
| Squeeze | sponge rate read | `SHA256(buffer ‖ ctr)` counter stream (SHA-256 is not an XOF) + re-absorb |
| `has_dedicated_fusion` | `True` (the permutation lowers to a fusion marker) | **`False`** |
| Seam | `Transcript` / `GrindingTranscript` (field-element, canonical-bit PoW) | `ByteTranscript` / `ByteGrindingTranscript` (byte, leading-zero-bit nonce PoW) |

## Why the byte transcript is exempt from the device-fusion clause

The fusion non-negotiable (CLAUDE.md) reads: "an `absorb`/`squeeze` … must each
lower to **one fused kernel**." That clause governs **device-lowered algebraic
primitives** — the Poseidon2 duplex whose `observe`/`sample` are device ops fused
into the same `@jit` region as the round body.

A SHA-256 byte Fiat-Shamir chain is a categorically different abstraction: it is
a **host orchestrator**, not a device kernel. SHA-256 is a byte hash, not
algebraic over the prime field, and the FS chain is strictly sequential (each
squeeze depends on every prior byte) — it does not lower to a GPU kernel at all.
It is in the same family as the host steps `docs/conventions.md` already
sanctions as interleaving the round loop (the eager `grind`, the `int(...)` query
index, the heterogeneous Python driver). A scheme on this transcript is
**per-island by construction**: the transcript runs on the host, and the fusion
contract is met *at the level it applies* — the bulk arithmetic *between*
challenges (NTT, sumcheck fold, Merkle build over 2^m elements) each still fuses
into one kernel.

`Sha256Transcript.has_dedicated_fusion` is therefore `False` — **the type-level
signal that this is a host orchestrator**, not a device primitive. Holding it to
the device-fusion clause would be a category error. (The *data-parallel* use of
the same hash — `zorch.hash.sha256.digest` over a batch of Merkle leaves or a PoW
window — does fuse, and is the device sibling; the sequential FS squeeze is the
only host-bound use.)

## Status / ratification

Admitting a host-side byte Fiat-Shamir family widens zorch's remit from
"algebraic, device-resident Fiat-Shamir" to "*also* host-side byte Fiat-Shamir."
That is a deliberate scope decision (driven by binary-field provers like flock,
whose verifier rests on a single SHA-256). **Flagged for ratification on epic
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).** If declined,
`byte_transcript.py` + `hash/sha256.py` move to the consumer with ~no rework
(they have no zorch-internal dependencies).
