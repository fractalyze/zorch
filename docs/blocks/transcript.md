# Transcripts — the transcript taxonomy

A Fiat-Shamir transcript turns prover messages into verifier challenges. zorch
carries **four** kinds, by construction-and-substrate, not by scheme:
`zorch/transcript.py`, `zorch/byte_transcript.py`,
`zorch/sha256_field_transcript.py`, and `zorch/blake3_field_transcript.py`, with
the squeeze width policy in `zorch/challenge.py`.

A transcript is neither a stage nor a round: it reduces no claim and repeats no
recurrence. It is the state both roles thread — explicit in every round call and
every stage result — which is why it appears in each signature rather than being
reached through a context object.

|                        | `transcript.py` — `DuplexTranscript`                                                          | `byte_transcript.py` — `ByteHashTranscript(ByteHash)`                                      | `sha256_field_transcript.py` — `Sha256FieldTranscript`                             | `blake3_field_transcript.py` — `Blake3FieldTranscript`                              |
| ---------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Primitive              | algebraic `Permutation` (Poseidon2) over a prime-field dtype                                  | a byte hash injected as a `ByteHash` — host `HostSha256` or the device `Sha256` marker     | device SHA-256 streaming Merkle–Damgård midstate                                   | device BLAKE3 resumable chunk state + subtree stack                                  |
| I/O                    | field elements (`Array`); `observe` bitcast-flattens to the base field                        | opaque `bytes`; the consumer serializes its own field↔bytes                                | field elements (`Array`); the byte surface, made scan-threadable                   | field elements (`Array`); the same byte surface, made scan-threadable                |
| Substrate              | **device**: `observe`/`sample` are device ops, threadable through `@jit` / a `lax.scan` carry | **host**: a `bytes` buffer; the injected `ByteHash.digest` runs on `hashlib` or the marker | **device**: a streaming `Sha256State` pytree — threads `@jit` / a `lax.scan` carry | **device**: a streaming `Blake3Stream` pytree — threads `@jit` / a `lax.scan` carry  |
| Squeeze                | sponge rate read                                                                              | `HASH(buffer ‖ ctr)` counter stream (a hash is not an XOF) + re-absorb                     | same `SHA256(buffer ‖ ctr)` counter stream over the streaming midstate             | one XOF read of the finalized state (BLAKE3 *is* an XOF) + re-absorb                 |
| `has_dedicated_fusion` | `True` (the permutation lowers to a fusion marker)                                            | **delegates to the `ByteHash`** — `False` for `HostSha256`, `True` for `Sha256`            | **`True`** (the SHA-256 chain lowers via the `hash_frx.sha256` marker)                | **`False`** — but see below: the hop DOES carry `zorch.blake3_{absorb,squeeze,finalize}`; this flag names hash-frx's whole-message region |
| Seam                   | `Transcript` (field-element, canonical-bit PoW)                                               | `ByteTranscript` (byte, leading-zero-bit nonce PoW)                                        | `Transcript` (field-element)                                                       | `Transcript` (field-element)                                                         |

The byte transcript is **one class parameterized by a `ByteHash`**: the same
Merlin-over-hash framing (op-tagged absorb, `HASH(buffer ‖ ctr)` counter-squeeze,
re-absorb) over an injected hash. "Host vs device" is not two classes but *which
`ByteHash`* you inject — `HostSha256()` for the host chain, `Sha256()` for the
`hash_frx.sha256` marker — and `has_dedicated_fusion` delegates to it, exactly as
`DuplexSponge` delegates to its `Permutation`. Both injections are byte-identical.

`Sha256FieldTranscript` is the **scan-threadable** surface: it keeps SHA-256's
incremental state (`Sha256State`, a fixed-shape pytree in
[hash-frx's `sha256.py`](https://github.com/fractalyze/hash-frx/blob/main/hash_frx/sha256.py))
instead
of a growing buffer, so a byte-Fiat-Shamir round loop folds through one device
program (`zorch.sumcheck.prove`'s `lax.scan`). Its slice framing is byte-identical
to `ByteHashTranscript`'s `observe_slice` / `sample_slice` — the field surface *is*
the byte surface, made scan-threadable.

`Blake3FieldTranscript` is that same construction over BLAKE3's resumable state,
and the interesting part is which of its two BLAKE3-specific behaviours is a
choice. The **XOF squeeze is not one**: the counter chain exists only because a
fixed-width hash cannot stretch its output, so over an XOF the counter
construction would be a deliberately wrong wire rather than a variant, and the
row reads the stream unconditionally with no seam. The **proof-of-work pre-image
width is** one: BLAKE3 hashes a message's length along with its bytes, so
`state_digest ‖ nonce_le8` padded to a whole block is a different search from the
unpadded 40 bytes, and which one is right is fixed by whichever verifier the
consumer has to agree with. It rides the transcript as `pow_preimage_bytes` —
defaulting to the unpadded wire `ByteHashTranscript.grind_pow` already speaks —
rather than a per-call argument, so a prover and its verifier cannot pick
differently op by op. Its only bound is the lower one — below `state_digest ‖
nonce_le8` the nonce does not fit. There is no upper bound on *correctness*,
because the row hashes the pre-image as a whole message through hash-frx rather
than assembling a compression itself, so any width is the same code.

Fusion does have an upper bound, at one block. Up to it the row takes hash-frx's
marked entry, which a recognizing emitter collapses to one kernel per hashed
batch; past it the row takes the unmarked entry instead, because a marked call
compiles its whole unrolled body and that stops being affordable somewhere
between a block and a chunk. A consumer padding past 64 bytes keeps the same
wire and the same bytes, and gives up the single kernel — nothing else.

## Device fusion: which transcripts meet it, and the host exemption

The fusion non-negotiable (CLAUDE.md) puts `absorb`/`squeeze` under the same
one-unit rule as a round body ([fusion north star](../README.md#fusion-north-star)).
It governs **device-lowered** Fiat-Shamir — the per-round `observe`/`sample`
that thread the round body's `@jit` region.

Three of the four transcripts are device and **meet it**:

- `DuplexTranscript` (Poseidon2) — `observe`/`sample` are device ops; a whole
  absorb+squeeze hop rides one `zorch.duplex_fs` marker.
- `Sha256FieldTranscript` — the **device-native SHA-256 prover path**. Its
  per-round `observe`/`sample` thread `zorch.sumcheck.prove`'s `lax.scan` as a
  device carry over the streaming `Sha256State`, so the whole Fiat-Shamir round
  loop folds into one device program with **no per-round host sync** — what a
  cuda-graph-unified scheme (e.g. flock) needs. The SHA-256 compression lowers via
  the `hash_frx.sha256` marker. Two honest caveats: there is no single whole-hop
  whole-hop fusion marker for the SLICE-framed hop yet (it leans on the
  per-compression marker + XLA; the scalar-framed observe+draw pair does ride one
  `zorch.sha256_squeeze` region), and per-hash
  SHA-256 is a worse GPU fit than Poseidon2's field mults — the win is keeping FS
  *in* the graph, not raw hash throughput.
- `Blake3FieldTranscript` — device-lowered on the same terms: `observe`/`sample`
  are device ops on a fixed-shape state, so the round loop folds into one device
  program with no per-round host sync, which is the win the SHA-256 row is here
  for. Two things it does not match, and a consumer swapping the rows should read
  both from here rather than from a profile. Its Fiat-Shamir hop carries zorch's
  own markers over the resumable state, but `has_dedicated_fusion` still reads
  `False`: that flag is about hash-frx's marked whole-message region, which the
  compressions are not entered through, so consumers keep taking their plain
  decomposition paths. And its substrate is not branchless the way
  `Sha256State` is: `Blake3Stream` carries data-dependent `while_loop`s for the
  subtree merges (the merge count follows the chunk count) plus a per-block
  `lax.cond`, so "one device program" here is not the same claim as
  capturable-by-construction. That is BLAKE3's tree shape, not an omission.

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

## Calling Fiat-Shamir: two entry points, and why only two

There are exactly two ways to take a Fiat-Shamir hop, split by what the caller
wants back:

- **A typed field challenge** — `ChallengePolicy.observe_and_sample`
  (`zorch/challenge.py`). It owns the limbs↔dtype packing, so a prover and its
  verifier cannot disagree about how many squeezes make an extension element.
- **Raw squeezes in the transcript's own field** — the transcript's own
  `observe_and_sample` / `observe` / `sample`.

Everything else in `transcript.py` that looks callable is a *backend body*, one
per placement: `_observe_and_sample_body` (plain device decomposition),
`_observe_and_sample_marked` (the same hop under the `zorch.duplex_fs` marker),
`_observe_and_sample_host` (the CPU sponge). `DuplexTranscript` picks between them
through its `fs` backend (`_DeviceFs` / `_HostFs`, chosen by
`new(..., fs_on_host=)`), which is why the bodies are private.

The rule earns its keep because breaking it fails *silently*. A call site that
names a body directly still computes the right challenge — the transcript stream
stays byte-identical — but it is pinned to that body's placement, so turning on
`fs_on_host=True` leaves that hop on the device and nothing complains. That is not
hypothetical: the jagged sumcheck's per-round hop called
`_observe_and_sample_marked` directly, which is ~78% of the Fiat-Shamir hops in a
jagged LogUp-GKR prove, so host FS would have moved barely a fifth of them.
`FsEntryPointTest` (`zorch/testing/transcript_host_fs_test.py`) now holds the line
two ways: it drives the round hop through a recording backend, and it statically
forbids any module outside `transcript.py` from importing a private name out of
it.

## Status / ratification

The byte-hash family is **device-first**: the field transcripts are the prover
path, and the host `ByteHashTranscript` is a **shrinking** surface — correctness
oracle (`test_device_substrate_matches_host` pins the device marker to stdlib
`hashlib`), verifier-side replay, and legacy consumer challengers — retired
incrementally as consumers move on-device. Both field transcripts grind on the
shared `zorch.grind` windowed device search with `DuplexTranscript`'s exact
semantics, over `zorch.grind.leading_zero_bits_ok` as the one PoW predicate, with
no host path anywhere in either.

Admitting this family widens zorch's remit beyond algebraic device-resident
Fiat-Shamir; that scope decision is **flagged for ratification on epic
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1)**. If declined,
these modules move to the consumer, but not untouched: both field transcripts
depend on `byte_transcript` for the Merlin wire vocabulary and on `grind` for
the windowed search and its PoW predicate. A move either takes those two along
or leaves them as a build boundary the consumer re-crosses.
