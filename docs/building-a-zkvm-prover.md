# Building a prover on zorch

`zorch` is the scheme- and zkVM-agnostic substrate; a *consumer* is the glue
that turns those blocks into one concrete prover that byte-matches a specific
reference. This doc is the contract between the two: what stays here, what a
consumer supplies, how a consumer injects its specific choices, and the
conventions every consumer follows so they read the same way.

It is the cross-cutting companion to the per-block docs ([`hash.md`](hash.md),
[`pcs.md`](pcs.md), [`coding.md`](coding.md), [`sumcheck.md`](sumcheck.md),
[`logup-gkr.md`](logup-gkr.md)). Those tell you *what a block is*; this tells
you *how a prover is assembled from them and where the seams are*.

______________________________________________________________________

## The one rule

> Anything specific to one proving scheme or one zkVM lives in the consumer.
> Anything that captures *every* scheme lives in `zorch`. A block that "knows"
> which scheme, zkVM, or field it is inside is a layering bug.

This restates `zorch`'s scheme- and zkVM-agnostic philosophy
([`../README.md`](../README.md)) from the consumer's side. The test for "does
this belong in `zorch`": *would a second, unrelated prover reuse it unchanged?*

The boundary is about the **generality of the decision, not of the math**. A
mathematically niche kernel that nonetheless defines one scheme's encoding is a
consumer concern; a mundane-but-universal structure like a Merkle tree is
`zorch`'s. The corollary that trips people up: a whole *PCS scheme* is
reusable. A fold-based commitment (FRI, Basefold, WHIR, …), its code, and its
query/Merkle machinery are reused by anyone running that scheme — they belong
in `zorch`. What stays in the consumer is the *way one prover arranges its data
around the scheme*: how it lays out, pads, stacks, or batches its columns, the
order it observes them into the transcript, and the constraint system it
evaluates. When a reusable block is still living in a consumer for historical
reasons, that is a migration owed to `zorch`, not a counterexample to the rule.

______________________________________________________________________

## What lives where

| Concern | `zorch` (generic) | Consumer (one scheme/zkVM) |
| --- | --- | --- |
| Fiat-Shamir | the transcript + `sample_challenge` + `grind`/`check_witness` protocols | the *parameterization* (rate, field) and the observe/sample *order* |
| Hashing | the `Permutation` seam, sponge, compression | the concrete permutation params (constants, width, field) |
| Commitment | the Merkle tree(s) and any reusable query/opening layout | which trace columns are committed, and the layout *schedule* |
| Codes | the `LinearCode`/`FoldableCode` seam + reusable codes and folds | which code + rate + coset, and any scheme-specific fold |
| PCS | the `PcsProver`/`PcsVerifier` protocols, shared fold-phase machinery, and reusable PCS instances | the per-prover stacking/region/batching *schedule* around the PCS |
| Sumcheck | the scan driver + per-variable rounds | the `combine` summand and the round wiring |
| Interaction argument | the reusable circuit + layer rounds | the interaction model that feeds it |
| Composition | the `Round` abstraction + chains | the actual stage sequence and the carry between stages |
| Constraints | — | the constraint system, its evaluator, the quotient/zero-check shape |
| Field | works over any field dtype | the choice of base/extension dtype |

The right-hand column is the entire surface a new prover writes. The left-hand
column it imports.

______________________________________________________________________

## How a consumer expresses "only what differs"

A consumer never forks a `zorch` block. It supplies the specific *values* and
*order* through four injection points; everything else is reuse.

### 1. The permutation / hash — a params object behind the `Permutation` seam

The hash is a `Permutation` plus a sponge (transcript) and compression (Merkle).
A consumer ships the pinned params for *its* permutation, field, and width;
nothing in `zorch`'s hashing code changes between two provers — only the params
object does. A consumer that uses a different permutation family entirely
supplies a different `Permutation` impl behind the same seam.

### 2. The field — a field dtype, threaded as data

Blocks are written over "whatever dtype the arrays carry," so the consumer
picks the base/extension dtypes and `zorch`'s NTT, sponge, and sumcheck follow
with no per-field branches. A small consumer-local module pins the
base↔extension embeddings and canonical-constant helpers.

### 3. The PCS / code — reuse the protocol and the fold machinery

A scheme that opens via a Merkle-committed fold implements the
`PcsProver`/`PcsVerifier` protocols but reuses the shared fold-phase machinery
([`pcs.md`](pcs.md)) and a `zorch` code ([`coding.md`](coding.md)). What the
consumer adds is the *prover's deviation*: the stacking/region/batching
schedule and any genuinely new, reusable fold — which goes upstream into
`zorch` first, then the consumer depends on it.

### 4. Composition — `Round`s threaded by a consumer-owned driver

`zorch` gives the `Round` abstraction and the chain/scan drivers; the consumer
decides the *stage sequence* and the *carry* passed between stages. Whether the
driver is an explicit function threading `(transcript, state)` or a chain of
`Round`s is a consumer choice — both are valid, and
[`stage-composition.md`](stage-composition.md) records the seam contract and
the glue-as-rounds shape for the chain form. Either way the coordinator glue
(sort order, protocol-derived sizes, the preamble observes, the stage→stage
handoffs) is the consumer's, because it encodes the reference's exact
transcript.

______________________________________________________________________

## Conventions a consumer follows

These make consumers interchangeable to a reader; adopt them in a new prover.

- **Layout mirrors the reference's stages.** One package per stage, named after
  the reference's vocabulary, plus a top-level `prove` (and `verify`) that
  composes them. A reader who knows the reference can navigate the port.
- **Byte-match, no tolerances.** The prover reproduces the reference prover
  exactly. Compare canonical (non-Montgomery) values; a hash mismatch is almost
  always a shape/padding delta, not a hash-param bug. Vendor golden fixtures
  per module.
- **Fixtures come from the reference, self-validated at generation.** A
  structural transcript-log walk that asserts the observe/sample sequence
  doubles as a check that the port understands Fiat-Shamir order.
- **Grinds run natively where the reference is serial.** `zorch`'s `grind`
  finds the lowest witness; if the reference grind is serial, both find the
  same one, so the prover need not read the witness from a fixture. PoW
  *verification* re-checks, never re-grinds.
- **The verifier mirrors the prover's transcript exactly.** It re-derives every
  challenge from the same preamble and checks each stage's relation from the
  proof + verifying key alone (no traces). Test both ways: an honest proof
  verifies; a one-field-tampered proof is rejected at the corresponding stage.
- **Scheme-agnostic gaps go upstream first.** If a block is missing and a
  second scheme would want it, add it to `zorch` and depend on it — do not fork
  it into the consumer. (This is the one rule, restated operationally.)

______________________________________________________________________

## Checklist for a new prover

1. **Pin the reference** (commit/tag), the field, and the permutation params;
   ship the one params file and a field-helpers module.
2. **Stand up the fixture harness** — a dump of the reference transcript +
   per-stage intermediates, self-validated at generation.
3. **Port stage by stage**, byte-matching each against its fixture; reuse the
   `zorch` spine and write only the scheme glue.
4. **Decide the PCS seam** — reuse the shared fold machinery + a `zorch` code if
   the scheme is fold-based; add any genuinely new, reusable fold to `zorch`
   first.
5. **Compose** the stages in a `prove`, computing the coordinator-owned
   sizes/sort/preamble yourself.
6. **Mirror it in `verify`** and test accept + tamper-reject.
7. **Exercise a second parameter shape** (e.g. production vs test) to catch
   assumptions fitted to one configuration.
