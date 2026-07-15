# zorch

> **SNARK = Σ IOP Round**

frx-native building blocks for Modern SNARKs. `zorch` sits between **frx** —
Fractalyze's fork of [JAX](https://github.com/jax-ml/jax) — and the proof systems
that consume it: frx provides tracing and codegen, lowered through **Fractalyze
XLA**, its fork of stock [XLA](https://github.com/openxla/xla) that adds native
field and elliptic-curve types. `zorch` provides the reusable pieces a proof
system is assembled from.

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

**The one unit is the `Round` — a prover↔verifier interaction of an IOP**: a
message observed into the Fiat-Shamir transcript, a challenge sampled back
(`observe`→`sample`, via `__call__`). `Round`s **nest** — a single per-variable
step is a `Round`, and a whole sumcheck (its per-variable `Round`s bundled) is
itself a `Round`. This is what **`SNARK = Σ IOP Round`** says literally: a
Fiat-Shamir-compiled IOP is a tree of these rounds, `Σ` flattening it to the leaf
interactions.

Grouping `Round`s gives either a bigger `Round` (a sumcheck, from its
per-variable rounds) or — when the group is a top-level phase — a `Stage`. Two
roles organize the composition; both are `Round`s, since a chain is itself one:

- A **`Stage`** is a `Round` that is one phase of the scheme's top-level chain
  (trace-commit, logup-gkr, zero-check, a PCS opening). Chaining Stages
  instantiates the scheme.
- A **`Bridge`** is a transcript-only `Round` that connects siblings — Stages in
  the scheme, or Rounds within a stage — for a soundness or security reason: a
  PoW grind (buys security bits), a framed observe or domain separator (closes a
  Fiat-Shamir soundness gap), a sampled-and-discarded challenge (matches the
  reference's schedule).

So the shape is recursive — a scheme chains `Stage`s (joined by `Bridge`s); a
`Stage` chains `Round`s; a `Round` may itself chain `Round`s, down to the leaf
interaction:

```text
prove()  —  a scheme: Stages chained, Bridges between them
────────────────────────────────────────────────────────────────

  Stage   trace-commit        commit the witness columns
  Bridge  grind               a PoW between stages (buys security bits)
  Stage   logup-gkr           the interaction argument, a chain of Rounds:
    Round   layer L              one layer — itself a Round of Rounds:
      Round   bind x₀              a leaf: one observe → sample
      Round   bind x₁
    Round   layer L-1
  Bridge  observe(framing)    bind the transcript in order (soundness)
  Stage   zero-check          the constraint sumcheck
  Stage   jagged-evals        the PCS opening
```

|             | **`Round`**                          | **`Stage`**                                       | **`Bridge`**                                      |
| ----------- | ------------------------------------ | ------------------------------------------------- | ------------------------------------------------- |
| **Is**      | a prover↔verifier interaction; nests | a `Round` that is a top-level phase               | a `Round` that connects siblings                  |
| **Does**    | `observe`→`sample` at the leaf       | witness + real compute (an inner sumcheck, an open) | a transcript op a scheme's soundness needs      |
| **Example** | a sumcheck round, or a whole sumcheck | trace-commit, logup-gkr, zero-check, jagged-evals | a PoW grind, a framed observe, a discarded sample |

`Stage` and `Bridge` are the same `Round` interface — that is how chains nest and
how the verifier mirrors the prover *round-for-round* — but the roles are what a
reader navigates by.

**Where the boundaries fall.** A leaf `Round` is each prover↔verifier interaction
(`observe`→`sample`); Rounds bundle into a bigger `Round` or, at a top-level
phase, a `Stage`; a `Bridge` sits wherever the reference's soundness argument
demands a transcript step between siblings. The full carry-and-seam contract is
[`docs/composition/stage-composition.md`](docs/composition/stage-composition.md).

**Where the classic pieces fit.** A ZK reader expects Fiat-Shamir, `Polynomial`,
`PCS`, and sumcheck as top-level "blocks." In this picture they are not peers of
the `Round`:

- **Fiat-Shamir, `Polynomial`, and fold** are the *materials* a `Round` body
  computes with — the transcript it threads, the polynomials it evaluates, the
  fold (2-to-1 reduction, one challenge per round) it applies each step.
- **A `PCS` opening and a zero-check are `Stage`s** — each a distinct phase: a
  zero-check reduces to a sumcheck, while a `PCS` opening runs its
  commitment-opening and evaluation checks (the *jagged-evals* stage above).

## Development

`zorch` is pure Python on frx, run against its GPU plugin. A virtualenv with
the pinned toolchain:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The dev loop — per-workspace venvs, developing against a local Fractalyze XLA
build, the frx compile-cache rule — lives in [`docs/reference/development.md`](docs/reference/development.md).

## Documentation

- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md) — indexes every
  design doc by what you're trying to do.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
