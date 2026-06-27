# zorch docs

Topic-organized reference. **Start with the table below — it indexes by what
you're trying to do.** For project overview, build, and install, see
[`../README.md`](../README.md).

______________________________________________________________________

## Understanding the design

| Question                                                        | Where                                                                 |
| --------------------------------------------------------------- | --------------------------------------------------------------------- |
| What is `zorch`, the building blocks, and the design philosophy | [`../README.md`](../README.md)                                        |
| New to JAX — the mental models behind the conventions, and the canonical references to learn from | [`jax.md`](jax.md)                  |
| Detailed design — fusion contract, findings, open decisions     | epic [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1) |
| Polynomial primitives — `eq`, multilinear eval/fold, and the ZKX field-dtype gotchas | [`poly.md`](poly.md)                                  |
| Symmetric primitives — the `Permutation` seam, Sponge, Compression, Poseidon2 | [`hash.md`](hash.md)                                          |
| Merkle commitment — binary tree on Sponge + Compression         | [`commit.md`](commit.md)                                             |
| Polynomial commitment seam — `PcsProver`/`PcsVerifier`, with KZG (pairing) and FRI (transparent) instances | [`pcs.md`](pcs.md)                  |
| Jagged Little Polynomial — verifier point-eval (branching program) | [`jagged.md`](jagged.md)                                          |
| Linear codes — `LinearCode` seam, Reed-Solomon encode + FRI fold on the native NTT | [`coding.md`](coding.md)                                  |
| The sumcheck block — design rationale & gotchas                 | [`sumcheck.md`](sumcheck.md)                                          |
| LogUp-GKR — fractional-sum circuit, prover/verifier round duals | [`logup-gkr.md`](logup-gkr.md)                                       |
| Stage composition — the seam contract between prove stages, nested chains, glue rounds | [`stage-composition.md`](stage-composition.md)        |
| Building a prover on zorch — the consumer/zorch boundary, injection points, conventions | [`building-a-zkvm-prover.md`](building-a-zkvm-prover.md) |

## Fusion north star

A `Round` must lower to **one replayable GPU unit, by construction** — never by a
per-primitive compiler pattern-match. The realistic target is a **single CUDA
graph** that captures the round's full launch sequence (`round_poly` → absorb →
squeeze → fold), **not necessarily a single fused kernel**: an XLA command buffer
replays the captured graph with no per-launch overhead, even if it is internally
several kernels. "One fused kernel" is the ideal limit of the same idea, not a
separate goal.

Two enablers make a round capturable: (1) the **whole round body in one traced
region** with a **device-side** transcript (so `observe`/`sample` are device
ops, not host steps that break the capture), and (2) a `stablehlo.composite`
marker the emitter lowers as one unit. Enabler (1) is in place
(`DuplexTranscript`). Until (2) lands — the marker + generic zkx emitter,
Phase 3 — round bodies are written **fusion-ready**: element-wise field ops
plus the one inherent `Σ`, no gratuitous `reduce`/`gather`, so they drop into
that path unchanged. See [`sumcheck.md`](sumcheck.md).

**Measured (ZKX GPU).** The bodies already lower as intended: `prover.SumcheckRound._round_poly`
compiles to a single reduction (`kInput`) kernel — the integrand fuses into the inherent
`Σ`, no marker needed — and `_fold` to a single element-wise (`kLoop`) kernel. A full round
(`_round_poly` + `_fold`) is **two** kernels (its message and folded state are disjoint
outputs), and with a host-side transcript it is not one traced region at all. So "one round
= one unit" means **one replayed CUDA graph** over those kernels (enablers (1)+(2)), not one
mega-fused kernel: a single giant extension-field kernel fights the compiler's field-aware
fusion splitting, and graph capture alone does not move warm time — the bottleneck is per-op
device latency, not host launch. The actual perf lever (register-resident fused kernels) is
a separate axis; rationale on epic #1.

## Conventions

Coding conventions — `@jit` usage, the WHY-not-WHAT rule for comments and docs,
and `_`-private naming — live in [`conventions.md`](conventions.md).
Docs prose is English, and a doc carries what the code cannot show — why a
thing exists, its background, the design philosophy, and the rules that follow
from it. What the code already states (the API surface, a usage walkthrough)
stays in the code and its tests, not here.

## Dev environment

Per-workspace venv pinning, the ZKX GPU plugin, and the JAX compile-cache rule
live in [`dev-env.md`](dev-env.md).

______________________________________________________________________

This hub grows as the repo does — each new subsystem adds a row here, not a
buried file.
