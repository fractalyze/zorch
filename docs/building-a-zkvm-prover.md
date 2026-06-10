# Building a zkVM prover on zorch

`zorch` is the scheme- and zkVM-agnostic substrate; a *consumer* (`openvm-zorch`,
`sp1-zorch`, `whir-zorch`, …) is the glue that turns those blocks into one
concrete prover that byte-matches a specific reference. This doc is the contract
between the two: what stays here, what a consumer must supply, how a consumer
injects its scheme-specific choices, and the conventions every consumer follows
so they read the same way.

It is the cross-repo companion to the per-block docs ([`hash.md`](hash.md),
[`pcs.md`](pcs.md), [`coding.md`](coding.md), [`sumcheck.md`](sumcheck.md),
[`logup-gkr.md`](logup-gkr.md)). Those tell you *what a block is*; this tells you
*how a prover is assembled from them and where the seams are*.

______________________________________________________________________

## The one rule

> Anything proving-scheme- or zkVM-specific lives in the consumer. Anything that
> captures *every* scheme lives in `zorch`. A block that "knows" it is inside
> WHIR, or SP1, or BabyBear is a bug in the layering.

This is the same non-negotiable as [`../CLAUDE.md`](../CLAUDE.md), stated from
the consumer's side. The test for "does this belong in zorch": *would a second,
unrelated proving scheme reuse it unchanged?* A Merkle tree, a duplex sponge, a
sumcheck scan, an RS fold — yes. A query-strided Merkle layout, a μ-batching
schedule, a constraint-DAG dialect — no; those encode one scheme's decisions.

The boundary is not about *generality of the math* but *generality of the
decision*. `eval_eq_uni` (the skip-domain Lagrange kernel) is mathematically
niche but scheme-defining — it lives in the consumer. `MerkleTree` is mundane but
universal — it lives in zorch.

______________________________________________________________________

## What lives where

| Concern | zorch (generic) | Consumer (scheme/zkVM) |
| --- | --- | --- |
| Fiat-Shamir | `DuplexTranscript`, `sample_challenge`, `grind`/`check_witness` protocols | the *parameterization* (rate, field) and the observe/sample *order* of the protocol |
| Hashing | `Permutation` seam, `Sponge`, `Compression` | the concrete `Poseidon2Params` (width, round constants, field) |
| Commitment | binary `MerkleTree` (hash-agnostic) | any scheme-specific tree layout (e.g. query-strided / jagged region packing) |
| Codes | `LinearCode`/`FoldableCode` seam, `ReedSolomon`, `fri_fold` | which code + blowup + coset, and any non-FRI fold (WHIR k-ary) |
| Sumcheck | the homogeneous scan driver + per-variable rounds | the `combine` summand (product, LogUp, constraint-RLC) and the round wiring |
| GKR | the fractional-sum circuit + layer rounds | the interaction model that feeds the input layer |
| PCS | `PcsProver`/`PcsVerifier` protocols, shared fold-phase machinery (`pcs/fold.py`) | the full PCS instance (FRI, Basefold, WHIR, KZG) |
| Composition | the `Round` abstraction + `ProveChain`/`VerifyChain` | the actual stage sequence and the carry threaded between stages |
| Constraints | — | the constraint system, its DAG/evaluator, the quotient/zerocheck shape |
| Field | works over any ZKX field dtype | the choice of base/extension dtype |

The right-hand column is the entire surface a new prover writes. The left-hand
column it imports.

______________________________________________________________________

## How a consumer expresses "only what differs"

A consumer never forks a zorch block. It supplies the scheme-specific *values*
and *order* through four injection points; everything else is reuse.

### 1. The permutation/hash — a params object behind the `Permutation` seam

`zorch.hash.poseidon2.Poseidon2` is generic over `Poseidon2Params` (width,
external/internal rounds, α, round constants, internal diagonal). The consumer
ships exactly one file: the pinned constants for *its* field and width.

- `openvm-zorch/poseidon2/babybear16.py` — BabyBear, width 16, plonky3 constants.
- `sp1-zorch/poseidon2/koalabear16.py` — KoalaBear, width 16, SP1 constants
  (note SP1's internal diagonal is a powers-of-two family and folds `R⁻¹` into
  the internal scale — a different *value*, same seam).

Then `Sponge(perm, SpongeParams(rate, out))` and
`Compression(perm, CompressionParams(arity, chunk))` parameterize the transcript
and Merkle tree. No zorch hashing code changes between the two zkVMs — only the
params object does.

### 2. The field — a ZKX dtype, threaded as data

Blocks are written over "whatever dtype the arrays carry." The consumer picks
`babybear_mont`/`babybearx4_mont` (OpenVM) or `koalabear_mont`/`koalabearx4_mont`
(SP1); zorch's NTT, sponge, and sumcheck follow the dtype with no per-field
branches. A small consumer-local `fields.py` pins the base↔extension embeddings
and canonical-constant helpers.

### 3. The PCS / code — implement the protocol, reuse the fold machinery

A scheme that opens via a Merkle-committed fold (FRI, Basefold, WHIR) implements
`PcsProver`/`PcsVerifier` but reuses `pcs/fold.py` (pre-fold pair commit, query
opener, position sampling, fold-chain verification) and `coding/reed_solomon.py`
(`fri_fold`, pair geometry). What the consumer adds is the *scheme's deviation*:

- WHIR (openvm-zorch) folds `k` sumcheck variables then re-encodes a fresh
  codeword each round — a *k-ary* step that decomposes into zorch's 2-ary
  `fold_pair`s plus a DFT, so no new fold primitive went upstream. The
  query-strided Merkle layout and the μ-batched RS message are SWIRL-specific
  and live in `openvm-zorch/whir/` + `openvm-zorch/commit/`.
- FRI/jagged (sp1-zorch) uses `BitReversedReedSolomon` + stacked Basefold open
  from `zorch.pcs.basefold`, with the jagged region packing in `sp1-zorch`.

Decision rule: a fold/code that a second scheme would reuse → zorch; the
stacking/region/batching schedule around it → consumer.

### 4. Composition — `Round`s threaded by a consumer-owned driver

zorch gives `Round`, `ProveChain`, `VerifyChain` and the homogeneous
`prove`/`verify` scan. The consumer decides the *stage sequence* and the *carry*
passed between stages. Two shapes are in use, both valid:

- **Explicit driver** (openvm-zorch `prove.py`/`verify.py`): a plain function
  threads `(transcript, state)` through stage functions. Chosen because SWIRL's
  stages are heterogeneous and the prover byte-matches a fixed reference order;
  an explicit driver makes that order legible.
- **`ProveChain` of `Round`s** (sp1-zorch `prove_shard.py`): each phase is a
  `Round` writing into a `ShardCarry`; `ProveChain` threads them. Chosen because
  SP1's phases compose cleanly as Rounds and the chain captures messages for
  serialization.

Either way the coordinator glue — stacking/sort order, protocol-derived sizes,
the preamble observes, the stage→stage handoffs — is the consumer's, because it
encodes the reference's exact transcript.

______________________________________________________________________

## The two reference implementations, side by side

| | openvm-zorch (SWIRL) | sp1-zorch (SP1) |
| --- | --- | --- |
| Reference | openvm-stark-backend `v2.0.0-beta.2` | SP1 (`slop`/`whir-zorch`) |
| Field / hash | BabyBear⁴ / Poseidon2-16 | KoalaBear⁴ / Poseidon2-16 |
| Stage 1 | stacked PCS commit (query-strided Merkle) | SMCS commit (jagged region) |
| Interactions | LogUp-GKR (`zorch.logup_gkr`) | LogUp-GKR (`zorch.logup_gkr`) |
| Constraints | batched ZeroCheck + LogUp sumcheck | multi-chip zerocheck |
| Opening reduction | stacked opening reduction | — |
| PCS | WHIR (k-ary fold + OOD + queries) | jagged eval + stacked Basefold FRI |
| Composition | explicit `prove()`/`verify()` driver | `ProveChain` of Rounds |
| Byte-match source | Rust `fixture-gen` (dumps reference transcript) | `rsp` dumps from instrumented SP1 |

The shared spine — duplex transcript, Poseidon2 seam, Merkle tree, LogUp-GKR,
sumcheck scan, RS/fold machinery — is *identical imports*. Everything in the
rows above that differs is consumer glue. That two unrelated zkVMs reuse the
same spine unchanged is the evidence the boundary is drawn correctly.

______________________________________________________________________

## Conventions a consumer follows

These make consumers interchangeable to a reader; adopt them in a new prover.

- **Layout mirrors the reference's stages.** One package per stage
  (`commit/`, `logup_gkr/`, …), named after the reference's vocabulary, plus a
  top-level `prove.py` (and `verify.py`) that composes them. A reader who knows
  the reference can navigate the port.
- **Byte-match, no tolerances.** The prover reproduces the reference prover
  exactly. Compare canonical (non-Montgomery) `u32`; a hash mismatch is almost
  always a shape/padding delta, not a hash-param bug. Vendor golden fixtures
  per module under `testdata/`.
- **Fixtures come from the reference, self-validated at generation.** Either a
  Rust dump harness (openvm-zorch's `fixture-gen`, pinned to the reference tag)
  or the reference's own instrumented dumps (sp1-zorch's `rsp`). A structural
  transcript-log walk that asserts observe/sample flags doubles as a check that
  the port understands the Fiat-Shamir sequence.
- **Grinds run natively where the reference is serial.** zorch's `grind`
  searches the lowest witness from 0; if the reference grind is serial
  (default-features off), both find the same witness, so the prover need not
  read the witness from a fixture. PoW *verification* re-checks, never re-grinds.
- **The verifier mirrors the prover's transcript exactly.** It re-derives every
  challenge from the same preamble and checks each stage's algebraic relation
  from the proof + vk alone (no traces). Test it both ways: an honest proof
  verifies; a one-field-tampered proof is rejected at the corresponding stage.
- **Scheme-agnostic gaps go upstream first.** If a block is missing and a second
  scheme would want it, add it to `zorch` and depend on it — do not fork it into
  the consumer. (This is the one rule, restated operationally.)

______________________________________________________________________

## Checklist for a new zkVM prover

1. **Pin the reference** (commit/tag) and the field + Poseidon2 params; ship the
   one `poseidon2/<field>16.py` constants file and a `fields.py`.
2. **Stand up the fixture harness** — a dump of the reference transcript +
   per-stage intermediates, self-validated at generation.
3. **Port stage by stage**, byte-matching each against its fixture; reuse the
   zorch spine (transcript, Merkle, GKR, sumcheck, RS/fold), write only the
   scheme glue.
4. **Decide the PCS seam** — reuse `pcs/fold.py` + a `zorch` code if the scheme
   is FRI-family; add any genuinely new, reusable fold to `zorch` first.
5. **Compose** the stages in a `prove()` (explicit driver or `ProveChain`),
   computing the coordinator-owned sizes/sort/preamble yourself.
6. **Mirror it in `verify()`** and test accept + tamper-reject.
7. **Exercise a second param shape** (e.g. production vs test) to catch
   assumptions fitted to one configuration.
