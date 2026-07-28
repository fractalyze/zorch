# zorch docs

Topic-organized reference, indexed by what you're trying to do. For project
overview, build, and install, see [`../README.md`](../README.md).

The tree mirrors the layering: **[`blocks/`](blocks)** are the reusable building
blocks, **[`schemes/`](schemes)** are full SNARKs assembled from them,
**[`composition/`](composition)** is how blocks compose into a prover, and
**[`reference/`](reference)** is conventions, environment, and learning.

______________________________________________________________________

## `blocks/` — reusable building blocks

| Question                                                                                    | Where                                   |
| ------------------------------------------------------------------------------------------- | --------------------------------------- |
| Polynomial primitives — `eq`, multilinear eval/fold, and the field-dtype gotchas            | [`poly.md`](blocks/poly.md)             |
| Symmetric primitives — the `Permutation` seam, Sponge, Compression, Poseidon2               | [`hash.md`](blocks/hash.md)             |
| Fiat-Shamir transcripts — the device-algebraic vs host-byte (SHA-256) taxonomy              | [`transcript.md`](blocks/transcript.md) |
| Merkle commitment — binary tree on Sponge + Compression                                     | [`commit.md`](blocks/commit.md)         |
| Linear codes — `LinearCode` seam, Reed-Solomon encode + FRI fold on the native NTT          | [`coding.md`](blocks/coding.md)         |
| Polynomial commitment seam — a committer plus an opening stage, with KZG / DEEP-FRI / BaseFold instances | [`pcs.md`](blocks/pcs.md)               |
| Jagged Little Polynomial — verifier point-eval (branching program)                          | [`jagged.md`](blocks/jagged.md)         |
| The sumcheck block — design rationale & gotchas                                             | [`sumcheck.md`](blocks/sumcheck.md)     |
| LogUp-GKR — fractional-sum circuit, prover/verifier round duals                             | [`logup-gkr.md`](blocks/logup-gkr.md)   |

## `schemes/` — full SNARKs assembled from blocks

| Question                                                                                     | Where                              |
| -------------------------------------------------------------------------------------------- | ---------------------------------- |
| Spartan R1CS combinators — zerocheck / RLC / lincheck / PCS-open glue, into the Spartan PIOP | [`spartan.md`](schemes/spartan.md) |

## `composition/` — assembling blocks into a prover

| Question                                                                                                                                     | Where                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Stage composition — paired prover/verifier roles, claim reduction, round boundaries, and the consumer split | [`stage-composition.md`](composition/stage-composition.md) |

## `reference/` — conventions, environment, learning

| Question                                                                                            | Where                                        |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Coding conventions — `@jit`, the WHY-not-WHAT rule, `_`-private naming, the subsystem doc skeleton  | [`conventions.md`](reference/conventions.md) |
| Dev environment — per-workspace venv pinning, the Fractalyze XLA plugin, the FRX compile-cache rule | [`development.md`](reference/development.md) |
| New to JAX — the mental models behind the conventions, and the canonical references to learn from   | [`jax.md`](reference/jax.md)                 |

Detailed design, the fusion contract, findings, and open decisions live on the
epic [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

______________________________________________________________________

## Fusion north star

A `Round` must lower to **one replayable GPU unit, by construction** — never by a
per-primitive compiler pattern-match. The target is a **single CUDA graph**
capturing the round's launch sequence (`round_poly` → absorb → squeeze → fold),
**not necessarily a single fused kernel**: a command buffer replays the captured
graph with no per-launch overhead even when it is internally several kernels.

Two enablers make a round capturable: the **whole body in one traced region**
with a **device-side** transcript, so `observe`/`sample` are device ops rather
than host steps that break the capture; and a `stablehlo.composite` marker the
emitter lowers as one unit. The first is in place (`DuplexTranscript`). Until the
second lands, bodies are written **fusion-ready** — element-wise field ops plus
the one inherent `Σ`, no gratuitous `reduce`/`gather` — so they drop into that
path unchanged.

**Measured (Fractalyze XLA GPU).** `_round_poly` compiles to a single reduction
(`kInput`) kernel and `_fold` to a single element-wise (`kLoop`) kernel, so a full
round is **two** kernels — its message and folded state are disjoint outputs —
and with a host-side transcript it is not one traced region at all. Hence "one
round = one unit" means one replayed graph, not one mega-kernel: a giant
extension-field kernel fights the compiler's field-aware fusion splitting, and
capture alone does not move warm time, where the bottleneck is per-op device
latency rather than host launch. Register-resident fused kernels are a separate
axis; rationale on epic #1.

______________________________________________________________________

This hub grows as the repo does — a new building block adds a row under
`blocks/`, a new full SNARK under `schemes/`; each lands in a folder, not a
buried flat file.
