# zorch

> **SNARK = Σ IOP Round**

JAX-native building blocks for Modern SNARKs. `zorch` sits between JAX and the
proof systems that consume it: JAX provides tracing and codegen (via Fractalyze
XLA (FXLA), the XLA backend with native finite-field dtypes); `zorch` provides
the reusable pieces a proof system is assembled from.

A Modern SNARK is **IOP + PCS**. The way deep learning stacks `Layer`s, `zorch`
stacks **`Round`s** — the one composable unit the rest is threaded through.

## Design Philosophy

- **Proving-scheme-agnostic.** The blocks capture *every* proving scheme, not a
  single one. `Round` / Fiat-Shamir / `Polynomial` / `PCS` / fold / zero-check
  compose into FRI, sumcheck, GKR, STARK, Basefold, WHIR, …; pairing-based
  schemes plug in by swapping the `PCS` block (e.g. a KZG-style commitment).
- **Implementation-agnostic.** `zorch` targets the proving scheme, not any one
  downstream implementation — a zkVM, a zkML prover, a zkTLS prover. Each plugs
  in as a *consumer*; nothing implementation-specific leaks into a block.
- **Fusion-first.** Each `Round` — and each `commit`/`open`, each
  `absorb`/`squeeze`, each fold step, and a hash permutation's internal rounds —
  must lower to a **single fused kernel**. We get there by *construction*, not
  by a per-primitive pattern-matcher in the compiler.
- **Easy to assemble.** These are building blocks; the API optimizes for snapping
  them together.

## Building blocks

**There is one composable unit: the `Round`** (implement `__call__`; it threads
the Fiat-Shamir transcript and calls `observe` / `sample` directly). Everything
else — Fiat-Shamir, `Polynomial`, `PCS`, fold, zero-check — is something a round
is *built from* or *reduces to* (the split is below), not a sibling unit.

`Round`s compose at three granularities. All three **are** `Round`s — chains
nest — so they differ only in altitude:

| Granularity | What it is                                        | Example                                                      |
| ----------- | ------------------------------------------------- | ------------------------------------------------------------ |
| **Stage**   | one phase of the argument, usually itself a chain | trace-commit · logup-gkr · zero-check · jagged-evals         |
| **Bridge**  | connects stages; transcript-only                  | a grind, a sampled-and-discarded challenge, a framed observe |
| **`Round`** | one step on the Fiat-Shamir schedule              | a per-variable sumcheck round                                |

**The split criterion is the Fiat-Shamir schedule, nothing else.** The atomic
`Round` is one `observe`→`sample`, and it lowers to one capturable unit. Coarser
rounds exist only because chains nest: you *split* a new round where the
transcript takes another `observe`/`sample`, and you *group* rounds into a stage
where the argument moves to its next phase. The full seam contract — the carry
between stages and the bridges between them — is
[`docs/composition/stage-composition.md`](docs/composition/stage-composition.md).

Two kinds of thing feed those rounds; neither is a sibling unit:

- **Materials a round body computes with** — the **Fiat-Shamir transcript**
  (`observe` / `sample`, a duplex sponge underneath), **`Polynomial`** primitives
  (univariate + multilinear eval), and the **fold** step (2-to-1 reduction, one
  challenge per round).
- **Reductions that *are* a Stage** — **`PCS`** (`commit` / `open` / `verify`;
  the other half of `IOP + PCS`, and its `open` is itself a stage) and
  **zero-check** (reduces a constraint system — injected by the consumer — to a
  sumcheck). These live at the Stage altitude, not inside a leaf round.

## Status

The core spine is up — `Round`, Fiat-Shamir, `Polynomial`, and the fusion
contract — and the blocks a prover is assembled from have landed on top of it:
Poseidon2 and device SHA-256 transcripts, the sumcheck engine, LogUp-GKR, the
`PCS` seam with KZG / FRI / BaseFold instances, the jagged PCS, and the Spartan
R1CS combinators. The current frontier is end-to-end IOP+PCS composition —
gluing the stages into a full prover (milestone `compose: e2e IOP+PCS gluing`,
[#462](https://github.com/fractalyze/zorch/issues/462)). Detailed design and open
decisions: epic issue [#1](https://github.com/fractalyze/zorch/issues/1).

## Development

`zorch` is pure Python on JAX, run against the Fractalyze XLA GPU plugin. A virtualenv with
the pinned toolchain:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The dev loop — per-workspace venvs, developing against a local Fractalyze XLA build, the
JAX compile-cache rule — lives in [`docs/reference/development.md`](docs/reference/development.md).

## Documentation

- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md) — indexes every
  design doc by what you're trying to do.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
