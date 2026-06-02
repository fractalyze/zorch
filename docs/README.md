# zorch docs

Topic-organized reference. **Start with the table below — it indexes by what
you're trying to do.** For project overview, build, and install, see
[`../README.md`](../README.md).

______________________________________________________________________

## Understanding the design

| Question                                                        | Where                                                                 |
| --------------------------------------------------------------- | --------------------------------------------------------------------- |
| What is `zorch`, the building blocks, and the design philosophy | [`../README.md`](../README.md)                                        |
| Detailed design — fusion contract, findings, open decisions     | epic [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1) |
| The sumcheck `Round` block — usage, composition, gotchas        | [`sumcheck.md`](sumcheck.md)                                          |
| Linear codes — `LinearCode` seam, Reed-Solomon via the native NTT | [`coding.md`](coding.md)                                             |

## Fusion north star

A `Round` must lower to **one replayable GPU unit, by construction** — never by a
per-primitive compiler pattern-match. The realistic target is a **single CUDA
graph** that captures the round's full launch sequence (`round_poly` → absorb →
squeeze → fold), **not necessarily a single fused kernel**: an XLA command buffer
replays the captured graph with no per-launch overhead, even if it is internally
several kernels. "One fused kernel" is the ideal limit of the same idea, not a
separate goal.

Two enablers make a round capturable: (1) the **whole round body in one traced
region** with a **device-side** transcript (so `observe`/`challenge` are device
ops, not host steps that break the capture), and (2) a `stablehlo.composite`
marker the emitter lowers as one unit. Until those land — the device transcript
is [#3](https://github.com/fractalyze/zorch/issues/3); the marker + generic zkx
emitter are Phase 3 — round bodies are written **fusion-ready**: element-wise
field ops plus the one inherent `Σ`, no gratuitous `reduce`/`gather`, so they
drop into that path unchanged. See [`sumcheck.md`](sumcheck.md).

**Measured (ZKX GPU).** The bodies already lower as intended: `ProductSumcheckRound.round_poly`
compiles to a single reduction (`kInput`) kernel — the integrand fuses into the inherent
`Σ`, no marker needed — and `fold` to a single element-wise (`kLoop`) kernel. A full round
(`round_poly` + `fold`) is **two** kernels (its message and folded state are disjoint
outputs), and with a host-side transcript it is not one traced region at all. So "one round
= one unit" means **one replayed CUDA graph** over those kernels (enablers (1)+(2)), not one
mega-fused kernel: a single giant extension-field kernel fights the compiler's field-aware
fusion splitting, and graph capture alone does not move warm time — the bottleneck is per-op
device latency, not host launch. The actual perf lever (register-resident fused kernels) is
a separate axis; rationale on epic #1.

## Conventions

Coding conventions (`@jit` usage, style) live in [`conventions.md`](conventions.md).
Docs prose is English, and a doc carries what the code cannot show — why a
thing exists, its background, the design philosophy, and the rules that follow
from it. What the code already states (the API surface, a usage walkthrough)
stays in the code and its tests, not here.

______________________________________________________________________

This hub grows as the repo does — each new subsystem adds a row here, not a
buried file.
