# Spartan

Spartan reduces R1CS satisfiability `(A·z)∘(B·z) = C·z` to an outer
zerocheck, a random-linear-combination of its terminal claims, an inner
lincheck, and a PCS opening of the witness.

`SpartanProver` and `SpartanVerifier` are the reference composite stage roles
for [`stage-composition.md`](../composition/stage-composition.md). Each owns only
its corresponding child roles and spells out the same non-linear dataflow:

- `OuterProver` and `OuterVerifier` reduce a `ZerocheckClaim` to a
  `RowEvaluationClaim` at `r_x`, containing `(Az, Bz, Cz)(r_x)`.
- `batch_claims` samples `r` and derives `Az + r·Bz + r²·Cz`. Both roles call
  this shared transcript operation, which has no proof section.
- `InnerProver` and `InnerVerifier` reduce a `LincheckClaim`, assembled from the
  instance, row claim, and batched value, to a `ColumnEvaluationClaim` at `r_y`.
- `WitnessOpenProver` and `WitnessOpenVerifier` reduce the resulting
  `WitnessOpeningClaim` to `None` by opening the committed witness at `r_y[1:]`
  and closing the lincheck identity.

`SpartanProver.prove` and `SpartanVerifier.verify` accept the same public
`SpartanClaim`; only the prover additionally accepts a private
`SpartanWitness`. `SpartanProof` exposes named `outer`, `inner`, and
`witness_open` sections. The role objects are deliberately separate: a deployed
verifier is constructible with a `PcsVerifier` alone and never retains a PCS
proving key.

## Injection points

The outer and inner roles own corresponding sumcheck roles. Those children own
their recurrence schedule, proof assembly, transcript advancement, and wire
format; configured `Round` objects remain internal recurrence kernels. Callers
can replace one role without introducing an engine or chain abstraction.

`PcsProver` is injected only into `SpartanProver`, and `PcsVerifier` only into
`SpartanVerifier`. They construct `WitnessOpenProver` and
`WitnessOpenVerifier`, respectively, so role-specific keys remain static
configuration without crossing the deployment boundary. Both the transparent
test PCS and Ligerito exercise this separation.

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
