# Spartan

Spartan reduces R1CS satisfiability `(A·z)∘(B·z) = C·z` to an outer
zerocheck, a random-linear-combination of its terminal claims, an inner
lincheck, and a PCS opening of the witness.

`Spartan` is the reference composite `Stage` for
[`stage-composition.md`](../composition/stage-composition.md). It owns three
child stages and spells out their non-linear dataflow:

- `OuterStage` reduces a `ZerocheckClaim` to a `RowEvaluationClaim` at
  `r_x`, containing `(Az, Bz, Cz)(r_x)`; its reduction proof contains the outer
  round polynomials and claimed evaluations.
- `batch_claims` samples `r` and derives `Az + r·Bz + r²·Cz`. Both sides
  call this shared transcript operation, which has no proof section.
- `InnerStage` reduces a `LincheckClaim`, assembled from the instance, row
  evaluation claim, and batched value, to a `ColumnEvaluationClaim` at `r_y`.
- `WitnessOpenStage` reduces the resulting `WitnessOpeningClaim` to `None` by
  opening the committed witness at `r_y[1:]` and closing the lincheck identity.

Both `Spartan.prove` and `Spartan.verify` accept the same public `SpartanClaim`;
only `prove` additionally accepts a private `SpartanWitness`. `SpartanProof`
exposes named `outer`, `inner`, and `witness_open` sections. There are no
separate free prover/verifier entry points: pairing the methods on the object
keeps the child configuration and proof contract together.

## Injection points

The outer and inner stages each own an ordinary `SumcheckStage` child. That
child pairs proving and verification and owns its recurrence schedule, proof
assembly, transcript advancement, and wire format; its configured `Round`
objects are internal recurrence kernels. Callers can replace the child stage
without introducing a separate engine or chain abstraction. `PcsProver` and
`PcsVerifier` are injected into `Spartan`, which constructs the terminal
`WitnessOpenStage`; they are static stage configuration rather than per-proof
claim or witness data. Both the transparent test PCS and Ligerito exercise the
same witness-opening stage.

The field is the caller's dtype. The assignment layout is witness-first
`z = (W, 1, X)`: `r_y[0]` selects the witness/public half and `W` opens at
`r_y[1:]`.

`ChallengePolicy()` keeps challenges in the transcript field. Selecting an
extension dtype raises the soundness floor; state promotion is an engine choice,
not an inherent cost of extension challenges. The default outer engine keeps the
full `(Az, Bz, Cz)` tables in the base field during round zero by factoring
`eq(tau, ·)`. Binding the first extension challenge promotes only the folded
half-size tables. Runtime remains backend-dependent, so consumers must benchmark
their concrete field/backend before opting in.

This module is a dense protocol prototype, not a succinct indexed verifier.
The transcript currently absorbs all of `A`, `B`, and `C`, and terminal matrix
evaluation is also dense, so verification performs two `O(mn)` operations. A
production indexed Spartan must preprocess the matrices, bind a verifier-key
digest instead, and replace dense evaluation with the corresponding succinct
opening.

## Validation

The structural cross-check pins the protocol independently of the orchestration:

- outer and inner round-polynomial degrees are 3 and 2;
- running claims follow the sampled challenges;
- batching uses powers of one challenge;
- the final lincheck claim equals the combined matrix evaluation times
  `z̃(r_y)`.

Each child stage also has direct accept/tamper-reject tests, and the composite is
tested with both the transparent PCS and recursive Ligerito PCS.
