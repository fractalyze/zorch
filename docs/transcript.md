# Transcripts — the transcript taxonomy

A Fiat-Shamir transcript turns prover messages into verifier challenges. zorch
carries **three** kinds, by construction-and-substrate, not by scheme:

| | `transcript.py` — `DuplexTranscript` | `byte_transcript.py` — `Sha256Transcript` | `device_byte_transcript.py` — `DeviceSha256Transcript` / `Sha256FieldTranscript` |
| --- | --- | --- | --- |
| Primitive | algebraic `Permutation` (Poseidon2) over a prime-field dtype | byte hash (SHA-256, FIPS 180-4) via host `hashlib` | byte hash (SHA-256) via the device `zorch.sha256` marker |
| I/O | field elements (`Array`); `observe` bitcast-flattens to the base field | opaque `bytes`; the consumer serializes its own field↔bytes | both: a `bytes` surface (`DeviceSha256Transcript`) + a field-element `Array` surface (`Sha256FieldTranscript`) |
| Substrate | **device**: `observe`/`sample` are device ops, threadable through `@jit` / a `lax.scan` carry | **host**: a strictly sequential byte chain; a Python orchestrator | **device**: a streaming Merkle–Damgård midstate (`Sha256State`) — the field surface threads `@jit` / a `lax.scan` carry |
| Squeeze | sponge rate read | `SHA256(buffer ‖ ctr)` counter stream (SHA-256 is not an XOF) + re-absorb | same `SHA256(buffer ‖ ctr)` counter stream, on device (byte-identical) |
| `has_dedicated_fusion` | `True` (the permutation lowers to a fusion marker) | **`False`** | **`True`** (the SHA-256 chain lowers via the `zorch.sha256` marker) |
| Seam | `Transcript` / `GrindingTranscript` (field-element, canonical-bit PoW) | `ByteTranscript` / `ByteGrindingTranscript` (byte, leading-zero-bit nonce PoW) | byte surface: `ByteTranscript` (+ byte/nonce PoW); field surface: `Transcript` |

The third is the **device sibling of the host byte transcript** (flock-zorch#6): the
identical Merlin-over-SHA-256 framing, but the compression runs on device via the
`zorch.sha256` marker instead of host `hashlib`, so — unlike the host byte
transcript — it *does* lower to a GPU kernel (`has_dedicated_fusion = True`). Its
streaming midstate keeps the state a fixed-shape pytree, so the field-element
surface (`Sha256FieldTranscript`) threads `zorch.sumcheck.prove`'s `lax.scan`,
collapsing a byte-Fiat-Shamir round loop into one device program — the reason the
host byte transcript's "host orchestrator" exemption below does **not** apply to
it. Both surfaces are byte-identical to the host byte transcript (and hence to
flock-core's `FsChallenger`); the byte surface backs flock's host challenger, the
field surface backs an on-device round loop.

## Why the *host* byte transcript is exempt from the device-fusion clause

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

The *host* `Sha256Transcript.has_dedicated_fusion` is therefore `False` — **the
type-level signal that this is a host orchestrator**, not a device primitive.
Holding it to the device-fusion clause would be a category error. (The
*data-parallel* use of the same hash — `zorch.hash.sha256.digest` over a batch of
Merkle leaves or a PoW window — does fuse. And `device_byte_transcript` now
lowers the *sequential* FS squeeze on device too, via the `zorch.sha256` marker
over a streaming midstate — but that is the separate device transcript above,
`has_dedicated_fusion = True`; this exemption is specifically about the *host*
`Sha256Transcript`, which stays a Python orchestrator.)

## Status / ratification

Admitting a host-side byte Fiat-Shamir family widens zorch's remit from
"algebraic, device-resident Fiat-Shamir" to "*also* host-side byte Fiat-Shamir."
That is a deliberate scope decision (driven by binary-field provers like flock,
whose verifier rests on a single SHA-256). **Flagged for ratification on epic
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).** If declined,
`byte_transcript.py` + `device_byte_transcript.py` + `hash/sha256.py` move to the
consumer with ~no rework (they have no zorch-internal dependencies).

The **device** byte transcript (`device_byte_transcript.py`) was added by
flock-zorch#6 to move flock's Fiat-Shamir on-device; it depends only on
`hash/sha256.py` (the marker) and the host byte transcript's framing constants, so
it travels with them under the same ratification.
