# zorch

> **SNARK = Σ IOP Round**

FRX-native building blocks for Modern SNARKs. `zorch` sits between **FRX** —
Fractalyze's fork of [JAX](https://github.com/jax-ml/jax) — and the proof systems
that consume it: FRX provides tracing and codegen, lowered through **Fractalyze
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

- A **`Stage`** is a `Round` that is one phase of the scheme's `prove_chain` —
  the sequence of Stages the scheme *is* (trace-commit, logup-gkr, zero-check, a
  PCS opening).
- A **`Bridge`** is a transcript-only `Round` for soundness or security work — a
  grind (buys security bits), a framed observe or domain separator (closes a
  Fiat-Shamir soundness gap), a sampled-and-discarded challenge (matches the
  reference's schedule). It usually sits *inside* a Stage; a scheme may also
  place one directly in the `prove_chain` between two stages (an RLC batching
  their claims).

So the shape is recursive — the `prove_chain` is `Stage`s (with the occasional
`Bridge` between them); a `Stage` chains `Round`s and `Bridge`s; a `Round` may
itself chain `Round`s, down to the leaf interaction:

```text
prove()  —  the prove_chain is Stages; a Stage holds Rounds and Bridges
──────────────────────────────────────────────────────────────────────

  Stage   trace-commit          commit the witness columns
  Stage   logup-gkr             the interaction argument:
    Bridge  grind                 a PoW inside the stage (buys security bits)
    Round   layer L                one layer — itself a Round of Rounds:
      Round   bind x₀                a leaf: one observe → sample
      Round   bind x₁
    Round   layer L-1
  Stage   zero-check            the constraint sumcheck:
    Bridge  observe(framing)      bind the transcript first (soundness)
    Round   bind x₀                a leaf: one observe → sample
    Round   bind x₁
  Stage   jagged-evals          the PCS opening
```

|             | **`Round`**                           | **`Stage`**                                            | **`Bridge`**                                  |
| ----------- | ------------------------------------- | ------------------------------------------------------ | --------------------------------------------- |
| **Is**      | a prover↔verifier interaction; nests  | a sequence of `Round`s that is one `prove_chain` phase | a transcript-only connective `Round`          |
| **Does**    | `observe`→`sample` at the leaf        | witness + real compute (an inner sumcheck, an open)    | a transcript op soundness/security needs      |
| **Example** | a sumcheck round, or a whole sumcheck | trace-commit, logup-gkr, zero-check, jagged-evals      | a grind, a framed observe, a discarded sample |

`Stage` and `Bridge` are the same `Round` interface — that is how chains nest and
how the verifier mirrors the prover *round-for-round* — but the roles are what a
reader navigates by.

**Where the boundaries fall.** A leaf `Round` is each prover↔verifier interaction
(`observe`→`sample`); Rounds bundle into a bigger `Round` or, at a `prove_chain`
phase, a `Stage`; a `Bridge` sits wherever the reference's soundness argument
needs a transcript op — inside a stage, or between two in the `prove_chain`. The
full carry-and-seam contract is
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

`zorch` is pure Python on FRX, run against its GPU plugin. A virtualenv with
the pinned toolchain:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

Install the git hooks with both stages named. Plain `pre-commit install` wires
only the `pre-commit` stage, which leaves the commit-message linter inactive —
formatting hooks fire while a malformed commit message sails through to CI:

```sh
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
a valid type, a lowercase summary with no trailing period, a header of at most
80 characters, and a body on everything but `docs`. The same linter runs in CI
over every commit in a pull request and over the PR title, which matters here
because this repo squash-merges with the title as the subject.

The dev loop — per-workspace venvs, developing against a local Fractalyze XLA
build, the FRX compile-cache rule — lives in [`docs/reference/development.md`](docs/reference/development.md).

## Documentation

- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md) — indexes every
  design doc by what you're trying to do.

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
