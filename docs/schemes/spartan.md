# Spartan

Spartan reduces R1CS satisfiability `(A·z)∘(B·z) = C·z` to an outer
zerocheck, a random-linear-combination of its terminal claims, an inner
lincheck, and a PCS opening of the witness.

`Spartan` is the reference composite `Stage` for
[`stage-composition.md`](../composition/stage-composition.md). It owns three
child stages and spells out their non-linear dataflow:

- `OuterStage` pairs the zerocheck prover and verifier. Its typed output is the
  point `r_x` and `(Az, Bz, Cz)(r_x)`; its proof contains the outer round
  polynomials and those claims.
- `batch_claims` samples `r` and derives `Az + r·Bz + r²·Cz`. Both sides
  call this shared transcript operation, which has no proof section.
- `InnerStage` pairs the lincheck prover and verifier. It consumes the typed
  outer and batch outputs and produces the point `r_y` plus the reduced final
  claim.
- `WitnessOpenStage` opens the committed witness at `r_y[1:]` and closes the
  lincheck identity.

`Spartan.prove` accepts a `SpartanWitness`; `Spartan.verify` accepts the public
`SpartanStatement` and the resulting proof. `SpartanProof` exposes named
`outer`, `inner`, and `witness_open` sections. There are no separate free
prover/verifier entry points: pairing the methods on the object keeps the child
configuration and proof contract together.

## Injection points

The outer and inner stages accept a matched `StageSumcheck` pair, so callers
can replace the per-variable sumcheck algorithm, domain, or wire form without
changing the phase schedule. The PCS is injected through `PcsProver` and
`PcsVerifier`; both the transparent test PCS and Ligerito exercise the same
witness-opening stage.

The field is the caller's dtype. The assignment layout is witness-first
`z = (W, 1, X)`: `r_y[0]` selects the witness/public half and `W` opens at
`r_y[1:]`.

## Validation

The structural cross-check pins the protocol independently of the orchestration:

- outer and inner round-polynomial degrees are 3 and 2;
- running claims follow the sampled challenges;
- batching uses powers of one challenge;
- the final lincheck claim equals the combined matrix evaluation times
  `z̃(r_y)`.

Each child stage also has direct accept/tamper-reject tests, and the composite is
tested with both the transparent PCS and recursive Ligerito PCS.
