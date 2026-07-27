# Transcripts — the transcript taxonomy

A Fiat-Shamir transcript turns prover messages into verifier challenges. zorch
carries **three** kinds, by construction-and-substrate, not by scheme:
`zorch/transcript.py`, `zorch/byte_transcript.py`, and
`zorch/sha256_field_transcript.py`, with the squeeze width policy in
`zorch/challenge.py`.

A transcript is neither a stage nor a round: it reduces no claim and repeats no
recurrence. It is the state both roles thread — explicit in every round call and
every stage result — which is why it appears in each signature rather than being
reached through a context object.

|                        | `transcript.py` — `DuplexTranscript`                                                          | `byte_transcript.py` — `ByteHashTranscript(ByteHash)`                                      | `sha256_field_transcript.py` — `Sha256FieldTranscript`                             |
| ---------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Primitive              | algebraic `Permutation` (Poseidon2) over a prime-field dtype                                  | a byte hash injected as a `ByteHash` — host `HostSha256` or the device `Sha256` marker     | device SHA-256 streaming Merkle–Damgård midstate                                   |
| I/O                    | field elements (`Array`); `observe` bitcast-flattens to the base field                        | opaque `bytes`; the consumer serializes its own field↔bytes                                | field elements (`Array`); the byte surface, made scan-threadable                   |
| Substrate              | **device**: `observe`/`sample` are device ops, threadable through `@jit` / a `lax.scan` carry | **host**: a `bytes` buffer; the injected `ByteHash.digest` runs on `hashlib` or the marker | **device**: a streaming `Sha256State` pytree — threads `@jit` / a `lax.scan` carry |
| Squeeze                | sponge rate read                                                                              | `HASH(buffer ‖ ctr)` counter stream (a hash is not an XOF) + re-absorb                     | same `SHA256(buffer ‖ ctr)` counter stream over the streaming midstate             |
| `has_dedicated_fusion` | `True` (the permutation lowers to a fusion marker)                                            | **delegates to the `ByteHash`** — `False` for `HostSha256`, `True` for `Sha256`            | **`True`** (the SHA-256 chain lowers via the `zorch.sha256` marker)                |
| Seam                   | `Transcript` / `GrindingTranscript` (field-element, canonical-bit PoW)                        | `ByteTranscript` / `ByteGrindingTranscript` (byte, leading-zero-bit nonce PoW)             | `Transcript` (field-element)                                                       |

The byte transcript is **one class parameterized by a `ByteHash`**: the same
Merlin-over-hash framing (op-tagged absorb, `HASH(buffer ‖ ctr)` counter-squeeze,
re-absorb) over an injected hash. "Host vs device" is not two classes but *which
`ByteHash`* you inject — `HostSha256()` for the host chain, `Sha256()` for the
`zorch.sha256` marker — and `has_dedicated_fusion` delegates to it, exactly as
`DuplexSponge` delegates to its `Permutation`. Both injections are byte-identical.

`Sha256FieldTranscript` is the **scan-threadable** surface: it keeps SHA-256's
incremental state (`Sha256State`, a fixed-shape pytree in `hash/sha256.py`) instead
of a growing buffer, so a byte-Fiat-Shamir round loop folds through one device
program (`zorch.sumcheck.prove`'s `lax.scan`). Its slice framing is byte-identical
to `ByteHashTranscript`'s `observe_slice` / `sample_slice` — the field surface *is*
the byte surface, made scan-threadable.

## Device fusion: which transcripts meet it, and the host exemption

The fusion non-negotiable (CLAUDE.md) puts `absorb`/`squeeze` under the same
one-unit rule as a round body ([fusion north star](../README.md#fusion-north-star)).
It governs **device-lowered** Fiat-Shamir — the per-round `observe`/`sample`
that thread the round body's `@jit` region.

Two of the three transcripts are device and **meet it**:

- `DuplexTranscript` (Poseidon2) — `observe`/`sample` are device ops; a whole
  absorb+squeeze hop rides one `zorch.duplex_fs` marker.
- `Sha256FieldTranscript` — the **device-native SHA-256 prover path**. Its
  per-round `observe`/`sample` thread `zorch.sumcheck.prove`'s `lax.scan` as a
  device carry over the streaming `Sha256State`, so the whole Fiat-Shamir round
  loop folds into one device program with **no per-round host sync** — what a
  cuda-graph-unified scheme (e.g. flock) needs. The SHA-256 compression lowers via
  the `zorch.sha256` marker. Two honest caveats: there is no single whole-hop
  fusion marker yet (it leans on the per-compression marker + XLA), and per-hash
  SHA-256 is a worse GPU fit than Poseidon2's field mults — the win is keeping FS
  *in* the graph, not raw hash throughput.

The **exemption is only `ByteHashTranscript`** — the *host* byte transcript. It
holds a growing `bytes` buffer and orchestrates the chain per-op on the host (one
dispatch per squeeze), so it does not lower to a device kernel and must not be held
to a clause about device kernels. It is in the same family as the host steps
`docs/reference/conventions.md` sanctions between round kernels (a one-shot `grind`, an
`int(...)` query index, the heterogeneous Python driver). `has_dedicated_fusion`
delegates to the injected `ByteHash`: `HostSha256` reports `False` (the type-level
signal of a host orchestrator); injecting `Sha256` reports `True` — the squeeze
*does* lower via the marker — but the chain is still host-driven per op, so
single-dispatching the whole loop is the real win, and that is
`Sha256FieldTranscript`'s job. The *data-parallel* use of the same hash
(`Sha256().digest` over a batch of Merkle leaves or a PoW window) fuses regardless.

## Status / ratification

The SHA-256 family is **device-first**: `Sha256FieldTranscript` is the prover
path, and the host `ByteHashTranscript` is a **shrinking** surface — correctness
oracle (`test_device_substrate_matches_host` pins the device marker to stdlib
`hashlib`), verifier-side replay, and flock's legacy challenger — retired
incrementally as consumers move on-device. Its grind runs the shared
`zorch.grind` windowed device search with `DuplexTranscript`'s exact semantics,
with no host path anywhere in the field transcript.

Admitting this family widens zorch's remit beyond algebraic device-resident
Fiat-Shamir; that scope decision is **flagged for ratification on epic
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1)**. If declined,
the four modules move to the consumer with no rework — they depend on nothing in
zorch beyond each other.
