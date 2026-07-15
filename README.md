# zorch

> **SNARK = Σ IOP Round**

JAX-native building blocks for Modern SNARKs. `zorch` sits between JAX and the
proof systems that consume it: JAX provides tracing and codegen, lowered through
**Fractalyze XLA** — Fractalyze's fork of stock
[XLA](https://github.com/openxla/xla) with native field and elliptic-curve
types. Every later "XLA" here means this fork, not the upstream compiler. `zorch`
provides the reusable pieces a proof system is assembled from.

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

**There is one composable unit: the `Round`** — implement `__call__`; it threads
the Fiat-Shamir transcript and calls `observe` / `sample` directly. A prover
stacks `Round`s into a chain, and **`Round`s nest**: a chain is itself a `Round`,
so one `Round` can expand into a whole sub-chain. At the leaf a `Round` is a
single `observe`→`sample`; higher up it is a phase built from many.

A prover's top-level chain reads as two recurring roles — **`Stage`** and
**`Bridge`**, both of them `Round`s:

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

|             | **`Stage`**                                       | **`Bridge`**                                      | **leaf `Round`**              |
| ----------- | ------------------------------------------------- | ------------------------------------------------- | ----------------------------- |
| **Role**    | one phase of the argument                         | join two stages on the transcript                 | one step inside a stage       |
| **Body**    | witness + real compute; usually a sub-chain       | transcript-facing only — a grind still hashes, but no witness crosses | one `observe`→`sample`        |
| **Example** | trace-commit, logup-gkr, zero-check, jagged-evals | a PoW grind, a framed observe, a discarded sample | a per-variable sumcheck round |

`Stage` and `Bridge` are the same `Round` interface — that is how chains nest and
how the verifier mirrors the prover *round-for-round* — but the roles are what a
reader navigates by.

**Where the boundaries fall.** A new `Round` splits where the transcript takes
another `observe`/`sample`; `Round`s group into a `Stage` where the argument
moves to its next phase; a `Bridge` sits where the reference demands a transcript
step between two stages. The Fiat-Shamir schedule is the primary boundary, with
the protocol-phase and carry seams alongside it — the full carry-and-seam
contract is [`docs/composition/stage-composition.md`](docs/composition/stage-composition.md).

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

`zorch` is pure Python on JAX, run against the XLA GPU plugin. A virtualenv with
the pinned toolchain:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The dev loop — per-workspace venvs, developing against a local XLA build, the
JAX compile-cache rule — lives in [`docs/reference/development.md`](docs/reference/development.md).

## Documentation

- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md) — indexes every
  design doc by what you're trying to do.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
