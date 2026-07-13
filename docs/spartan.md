# spartan — design notes

Generic Spartan R1CS combinators, assembled from zorch blocks. The design
lineage and open decisions live on
[fractalyze/zorch#462](https://github.com/fractalyze/zorch/issues/462) (milestone
`compose: e2e IOP+PCS gluing`).

______________________________________________________________________

## Why the shape

Spartan reduces R1CS satisfiability `(A·z)∘(B·z) = C·z` to two sum-checks plus a
polynomial opening: an **outer** sum-check (the zerocheck of
`eq(τ,x)·(Az·Bz − Cz)`), a random-linear-combination that batches the three
resulting claims, an **inner** sum-check (the lincheck binding the batched
matrix against the witness MLE), and a PCS opening of the witness that closes the
final identity. It is a *scheme* — so by zorch's first non-negotiable it cannot
be a block. What lives here instead is a set of **agnostic R1CS combinators** the
scheme is assembled from, plus a thin assembly that owns only the schedule:

- `OuterProver`/`OuterVerifier` (zerocheck), `InnerProver`/`InnerVerifier`
  (lincheck) — each a `zorch.round.Round` that runs the shared `zorch.sumcheck`
  machinery and adds only its schedule and terminal identity check. The
  per-variable sum-check engine is **injected** as a matched (prover, verifier)
  `StageSumcheck` pair (default `zerocheck_engine` / `lincheck_engine`), so a
  caller swaps the sum-check algorithm, evaluation domain, or wire form (e.g. the
  compressed coefficient wire) without touching the stage. The prover recovers its
  bound point by replaying the injected *verifier* round, so point collection
  stays wire-agnostic — no message form is baked into the stage. (The outer
  summand `E·(A·B−C)` is mixed-degree, so a swapped outer engine must keep a
  finite domain.)
- `RlcProver`/`RlcVerifier` — the random-linear-combination combinator that
  folds N claims under one fresh challenge (powers of `r`). This is milestone
  #7's batching/RLC combinator; it is transcript-only glue, so it emits a `None`
  message and its verifier mirrors it round-for-round.
- `WitnessOpenProver`/`WitnessOpenVerifier` — milestone #7's
  sum-check-claim → PCS-opening glue, the piece `zorch.verify` deliberately
  leaves to the consumer ("the verifier reduces; the PCS closes"). It depends
  only on the `zorch.pcs.protocol` seam, so the PCS is injected — both a
  transparent `DensePcs` reference and the real recursive **Ligerito** PCS drive
  the same glue unchanged (`testing/`, the latter through a base↔extension bridge
  since the toy R1CS is base-field), and `basefold`/`whir`/`kzg` drop in the same
  way.

The field is the caller's dtype throughout — matrices, witness, and challenges
carry no field assumption. The assignment layout is **witness-first**
`z = (W, 1, X)`: `W` fills the low half, so the inner sum-check's first-bound
variable `r_y[0]` is the half-selector between `W` and the public `(1, X)`, and
`W` opens at `r_y[1:]`. This matches microsoft/Spartan2 (`vega-prover`), which is
the structural cross-check reference — the two implementations share the
algebraic skeleton (round-poly degrees 3/2, per-round evaluation tuples, running
claims, powers-of-`r` batching, final evals) even though byte-equality across
fields is impossible. The same structural cross-check (not a byte-match) is how
flock is validated: flock's char-2 Karatsuba-∞ round message is not the
evaluation-domain form these combinators emit, so its field arithmetic stays in
flock-zorch (see epic #1).

The seam state between stages rides one accumulating **pipeline carry**
(`SpartanCarry`, per [`stage-composition.md`](stage-composition.md)): values that
skip a stage — `r_x`, read by both the inner sum-check and the PCS glue — live
there rather than in a pass-through pairwise seam type. Each stage writes its own
fields and fails loud if a predecessor has not run.

## Fusion is by construction

Each stage is a `Round`, so stage granularity is capture granularity
([`stage-composition.md`](stage-composition.md), "Fusion by construction"). The
per-variable sum-check bodies are the shared `zorch.sumcheck` rounds, already
written fusion-ready: the zerocheck summand `E·(A·B − C)` and the lincheck
product `M·Z` each lower to element-wise field ops plus the one inherent `Σ` (no
gratuitous `reduce`/`gather`), verified by `assert_fusion_ready`. The combinators
add no kernel boundary — the RLC and PCS-glue rounds are transcript-and-arithmetic
only, and the terminal identity checks are scalar. So a Spartan stage lowers to
one replayable unit by construction, never by a per-primitive compiler
pattern-match.
