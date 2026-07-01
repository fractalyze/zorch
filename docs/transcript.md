# Transcripts — the transcript taxonomy

A Fiat-Shamir transcript turns prover messages into verifier challenges. zorch
carries **three** kinds, by construction-and-substrate, not by scheme:

| | `transcript.py` — `DuplexTranscript` | `byte_transcript.py` — `ByteHashTranscript(ByteHash)` | `sha256_field_transcript.py` — `Sha256FieldTranscript` |
| --- | --- | --- | --- |
| Primitive | algebraic `Permutation` (Poseidon2) over a prime-field dtype | a byte hash injected as a `ByteHash` — host `HashlibSha256` or the device `Sha256` marker | device SHA-256 streaming Merkle–Damgård midstate |
| I/O | field elements (`Array`); `observe` bitcast-flattens to the base field | opaque `bytes`; the consumer serializes its own field↔bytes | field elements (`Array`); the byte surface, made scan-threadable |
| Substrate | **device**: `observe`/`sample` are device ops, threadable through `@jit` / a `lax.scan` carry | **host**: a `bytes` buffer; the injected `ByteHash.digest` runs on `hashlib` or the marker | **device**: a streaming `Sha256State` pytree — threads `@jit` / a `lax.scan` carry |
| Squeeze | sponge rate read | `HASH(buffer ‖ ctr)` counter stream (a hash is not an XOF) + re-absorb | same `SHA256(buffer ‖ ctr)` counter stream over the streaming midstate |
| `has_dedicated_fusion` | `True` (the permutation lowers to a fusion marker) | **delegates to the `ByteHash`** — `False` for `HashlibSha256`, `True` for `Sha256` | **`True`** (the SHA-256 chain lowers via the `zorch.sha256` marker) |
| Seam | `Transcript` / `GrindingTranscript` (field-element, canonical-bit PoW) | `ByteTranscript` / `ByteGrindingTranscript` (byte, leading-zero-bit nonce PoW) | `Transcript` (field-element) |

The byte transcript is **one class parameterized by a `ByteHash`**: the same
Merlin-over-hash framing (op-tagged absorb, `HASH(buffer ‖ ctr)` counter-squeeze,
re-absorb) over an injected hash. "Host vs device" is not two classes but *which
`ByteHash`* you inject — `HashlibSha256()` for the host chain, `Sha256()` for the
`zorch.sha256` marker — and `has_dedicated_fusion` delegates to it, exactly as
`DuplexSponge` delegates to its `Permutation`. Both injections are byte-identical.

`Sha256FieldTranscript` is the **scan-threadable** surface: it keeps SHA-256's
incremental state (`Sha256State`, a fixed-shape pytree in `hash/sha256.py`) instead
of a growing buffer, so a byte-Fiat-Shamir round loop folds through one device
program (`zorch.sumcheck.prove`'s `lax.scan`). Its slice framing is byte-identical
to `ByteHashTranscript`'s `observe_slice` / `sample_slice` — the field surface *is*
the byte surface, made scan-threadable.

## Why a byte transcript is exempt from the device-fusion clause

The fusion non-negotiable (CLAUDE.md) reads: "an `absorb`/`squeeze` … must each
lower to **one fused kernel**." That clause governs **device-lowered algebraic
primitives** — the Poseidon2 duplex whose `observe`/`sample` are device ops fused
into the same `@jit` region as the round body.

A SHA-256 byte Fiat-Shamir chain is a categorically different abstraction: a
**host orchestrator**, not a device kernel. SHA-256 is a byte hash, not algebraic
over the prime field, and the FS chain is strictly sequential (each squeeze depends
on every prior byte) — it does not lower to a single GPU kernel. It is in the same
family as the host steps `docs/conventions.md` already sanctions as interleaving
the round loop (the eager `grind`, the `int(...)` query index, the heterogeneous
Python driver). A scheme on this transcript is **per-island by construction**: the
transcript runs on the host, and the fusion contract is met *at the level it
applies* — the bulk arithmetic *between* challenges (NTT, sumcheck fold, Merkle
build over 2^m elements) each still fuses into one kernel.

`ByteHashTranscript.has_dedicated_fusion` therefore **delegates to the injected
`ByteHash`**: `HashlibSha256` reports `False` — the type-level signal of a host
orchestrator — so holding that configuration to the device-fusion clause would be
a category error. Injecting `Sha256` reports `True` (the squeeze *does* lower via
the `zorch.sha256` marker), but the chain is still host-driven per op, so the
sequential byte challenger is not itself the perf win; single-dispatching the whole
loop is, which is what `Sha256FieldTranscript` (device streaming midstate) is for.
The *data-parallel* use of the same hash — `Sha256().digest` over a batch of Merkle
leaves or a PoW window — fuses regardless.

## Status / ratification

Admitting a host-side byte Fiat-Shamir family widens zorch's remit from
"algebraic, device-resident Fiat-Shamir" to "*also* host-side byte Fiat-Shamir."
That is a deliberate scope decision (driven by binary-field provers like flock,
whose verifier rests on a single SHA-256). **Flagged for ratification on epic
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).** If declined,
`byte_transcript.py` + `hash/byte_hash.py` + `hash/sha256.py` +
`sha256_field_transcript.py` move to the consumer with ~no rework (they have no
zorch-internal dependencies beyond each other).
