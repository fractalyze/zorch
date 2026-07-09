# hash — symmetric primitives

The *why* behind `zorch/hash/`. The *what* lives in the code and its tests, the
executable usage that runs on every commit. Full design and open decisions: epic
issue [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Why the shape

**One `Permutation` seam, every primitive on top.** A `Permutation` is a
fixed-width permutation over a single field dtype — `width`, `dtype`, `permute`.
The Sponge (leaf hasher), the Compression (Merkle fold), and the duplex
Fiat-Shamir transcript all read `width`/`dtype` to size and allocate state, then
call `permute`; none names a concrete hash. `Poseidon2` is one implementation
and classic `Poseidon` (`zorch/hash/poseidon`, the `zorch.poseidon` marker — a
full dense MDS every round, full/partial round split) is a second; any other
fixed-width permutation drops into the same seam unchanged. This is how the
symmetric layer stays proving-scheme- and zkVM-agnostic — the non-negotiable.

**A second seam for byte hashes: `ByteHash`.** SHA-256 (and BLAKE3, Keccak) is a
*byte* hash, not an algebraic `Permutation`: it maps a batch of equal-length byte
messages to digests — `digest(uint8[B, L]) -> uint8[B, digest_size]` — with its
construction (Merkle–Damgård / tree / sponge) hidden behind `digest`. `ByteHash`
is the byte sibling of `Permutation`: the byte Fiat-Shamir transcript
(`ByteHashTranscript`) and byte Merkle read `digest_size` and call `digest`,
naming no concrete hash. `Sha256` (the `zorch.sha256` device marker) and
`HostSha256` (host `hashlib`) are two implementations of the same FIPS 180-4
bytes that differ only in substrate — carried by `has_dedicated_fusion`, exactly
as on the permutation side, so `has_dedicated_fusion` delegates from the injected
hash the way `DuplexSponge`'s does from its `Permutation`. A shared *streaming*
surface is deliberately absent from the seam: the incremental midstate shape
differs per construction, so only `digest` generalizes. SHA-256's streaming core
(`Sha256State`) lives in `sha256.py` and backs the scan-threadable
`Sha256FieldTranscript`.

**Width from the permutation; the rest are free params on a frozen object.**
`rate`/`out` (`SpongeParams`) and `arity`/`chunk` (`CompressionParams`) are the
only knobs, carried like `Poseidon2Params`. Width is *not* a free param — it is
whatever the permutation provides. Validation splits accordingly: the params
object checks the field-only invariants (`rate ≥ 1`, `arity ≥ 2`), and the
operator, which knows the width, checks the rest (`rate < width`,
`arity·chunk ≤ width`).

**Padding-free overwrite sponge; truncated-permutation compression.** The Sponge
overwrites (replace, not XOR) the first `rate` lanes per block and adds *no*
padding, so a final partial block overwrites only its own lanes — the Merkle
leaf hasher (Plonky3 `PaddingFreeSponge`). Compression zero-pads `arity·chunk`
lanes into the width, permutes, and truncates to `chunk` (collision-resistant in
the hash-tree setting). Neither adds a domain separator: that is scheme-specific
and belongs to the consumer.

**Add-absorb duplex sponge.** `DuplexSponge` is the sibling for interleaved
absorb/squeeze (Fiat-Shamir): absorb *adds* into the rate lanes (`+=`, not
overwrite), permuting on a full rate block or a duplex direction switch; squeeze
reads the rate, permuting when it drains. It is the agnostic primitive a classic
sponge prover drives — the scheme-specific challenge packing, domain separation,
and field conversions live in the consumer, not here.

## Fusion by construction

The **permutation is the fusion unit.** `Poseidon2.permute` wraps all rounds in
one `fused_region` marker (`zorch/fusion.py`) that zkx lowers to a single
custom-fusion kernel — one kernel *by construction*, not by a per-hash compiler
pattern-match. What the block owes that path is a **straight-line body**: rounds
unrolled, linear layers in normal form, nothing that lowers to a
reduce / gather / scatter and splits the kernel. The op-level specifics live in
the `poseidon2` and `fusion.py` docstrings; the rule here is that the permutation
is written *for* that one marker.

This is the per-block face of the hub's
[fusion north star](README.md#fusion-north-star): a `vmap` over a Merkle layer or
a sponge's block loop batches into one `permute`, which collapses to one GPU
kernel once the permutation is captured (the poseidon2 fusion path,
[#25](https://github.com/fractalyze/zorch/issues/25)).

## Out of scope

Loop-carrying large-`N` permutations await the in-kernel-loop emitter (#25);
today's bodies are unrolled. Domain separation, sponge-vs-compression choice for
a given protocol, and concrete hash parameters past Poseidon2 land with the
consumer that needs them.
