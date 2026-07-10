# zorch

JAX-native building blocks for Modern SNARKs. `zorch` sits between
[JAX](https://github.com/jax-ml/jax) and the proof systems that consume it
(e.g. [`whir-zorch`](https://github.com/fractalyze/whir-zorch)): JAX provides
tracing and codegen (via [Fractal XLA](https://github.com/fractalyze/xla), Fractalyze's
XLA fork with native finite-field dtypes); `zorch` provides the reusable pieces
a proof system is assembled from.

A Modern SNARK is **IOP + PCS**. `zorch` gives you those as composable blocks —
the way deep learning stacks `Layer`s, `zorch` stacks **`Round`s**.

## Design Philosophy

- **Proving-scheme-agnostic.** The blocks capture *every* proving scheme, not a
  single one. `Round` / Fiat-Shamir / `Polynomial` / `PCS` / fold / zero-check
  compose into FRI, sumcheck, GKR, STARK, Basefold, WHIR, …; pairing-based
  schemes plug in by swapping the `PCS` block (e.g. a KZG-style commitment).
- **zkVM-agnostic.** `zorch` knows nothing about any zkVM. Nothing zkVM-specific
  leaks into a block.
- **Fusion-first.** Each `Round` — and each `commit`/`open`, each
  `absorb`/`squeeze`, each fold step, and a hash permutation's internal rounds —
  must lower to a **single fused kernel**. We get there by *construction*, not
  by a per-primitive pattern-matcher in the compiler.
- **Easy to assemble.** These are building blocks; the API optimizes for snapping
  them together.

## Building blocks

- **`Round`** — the composable unit (implement `__call__`). It threads the
  Fiat-Shamir transcript and calls its `observe` / `sample` directly.
- **Fiat-Shamir / Challenge** — the transcript interface `observe` / `sample`; the
  concrete impl is a duplex sponge (`absorb` / `squeeze`) under the hood.
- **`Polynomial`** — univariate and multilinear representations.
- **`PCS`** — polynomial commitment: `commit` / `open` / `verify`.
- **Fold** — 2-to-1 reduction (same program, half the input), one random
  challenge per round.
- **Zero-check** — constraints are injected from outside the block.

## Status

Early bootstrap. The first milestone brings up the core spine (`Round`,
Fiat-Shamir, `Polynomial`, and the fusion contract) and validates it by
migrating `poseidon2` from `whir-zorch`. Detailed design and decisions:
milestone `spine: core + poseidon2 v1`, epic issue
[#1](https://github.com/fractalyze/zorch/issues/1).

## Development

`zorch` is pure Python on JAX, run against the Fractal XLA GPU plugin. A virtualenv with
the pinned toolchain:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The dev loop — per-workspace venvs, developing against a local Fractal XLA build, the
JAX compile-cache rule — lives in [`docs/dev-env.md`](docs/dev-env.md).

## Documentation

- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Detailed design & open decisions:** epic issue [#1](https://github.com/fractalyze/zorch/issues/1) (milestone `spine: core + poseidon2 v1`)

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
