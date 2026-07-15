# zorch

> **SNARK = Σ IOP Round**

frx-native building blocks for Modern SNARKs. `zorch` sits between **frx**
(Fractalyze's JAX fork) and the proof systems that consume it: frx provides
tracing and codegen, lowered through **Fractalyze XLA** — its fork of stock
[XLA](https://github.com/openxla/xla) that adds native field and elliptic-curve
types. `zorch` provides the reusable pieces a proof system is assembled from.

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

**The one unit is the `Round` — one prover↔verifier interaction of an IOP.**
Implement `__call__`: observe a message into the Fiat-Shamir transcript, sample a
challenge back (`observe`→`sample`). A per-variable sumcheck step is one Round.
This is what **`SNARK = Σ IOP Round`** says literally — a Fiat-Shamir-compiled IOP
*is* a sequence of these rounds.

Rounds compose into a scheme through two roles — both are `Round`s, since a chain
is itself a `Round`:

- A **`Stage`** groups Rounds into one phase of the argument (trace-commit,
  logup-gkr, zero-check, a PCS opening). Chaining Stages instantiates the scheme.
- A **`Bridge`** is a single connective Round between two stages, there for a
  soundness or security reason the reference demands: a PoW grind (buys security
  bits), a framed observe or domain separator (closes a Fiat-Shamir soundness
  gap), a sampled-and-discarded challenge (matches the reference's schedule).

So the shape is **scheme → Stages → Rounds**, with Bridges the connective Rounds
between stages:

```text
prove()  —  a chain of Stages joined by Bridges
──────────────────────────────────────────────────────────────

  Stage   trace-commit      commit the witness columns
  Bridge  grind             a PoW that only feeds the transcript
  Stage   logup-gkr         the interaction argument (its own sub-chain)
  Bridge  observe(framing)  absorb in the reference's exact order
  Stage   zero-check        the constraint sumcheck — a chain of Rounds:
              Round  bind x₀     one sumcheck step (observe → sample)
              Round  bind x₁     … one Round per variable
  Stage   jagged-evals      the PCS opening
```

|             | **`Round`**                     | **`Stage`**                                       | **`Bridge`**                                      |
| ----------- | ------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| **Is**      | one prover↔verifier interaction | a phase — a chain of Rounds                        | one connective Round between two stages           |
| **Does**    | `observe`→`sample`              | witness + real compute (an inner sumcheck, an open) | a transcript op a scheme's soundness needs      |
| **Example** | a per-variable sumcheck round   | trace-commit, logup-gkr, zero-check, jagged-evals | a PoW grind, a framed observe, a discarded sample |

`Stage` and `Bridge` are the same `Round` interface — that is how chains nest and
how the verifier mirrors the prover *round-for-round* — but the roles are what a
reader navigates by.

**Where the boundaries fall.** A new `Round` starts at each prover↔verifier
interaction (`observe`→`sample`); Rounds group into a `Stage` where the argument
moves to its next phase; a `Bridge` sits where the reference's soundness argument
demands a transcript step between two stages. The full carry-and-seam contract is
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
