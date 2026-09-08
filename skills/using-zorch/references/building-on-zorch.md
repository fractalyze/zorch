# Building a prover on zorch

Your project (a zkVM, zkML, zkTLS prover, …) is a **consumer**: it assembles
zorch blocks into one concrete proof system, usually byte-matching a reference
implementation. This page is the boundary contract and the assembly rules,
condensed from
[stage-composition.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/composition/stage-composition.md).

## The boundary

> A component belongs in zorch when a second, unrelated prover would reuse it
> unchanged. The test is the generality of the *decision*, not the math.

| zorch owns (import it) | Your repo owns (write it) |
| --- | --- |
| Stages, round drivers, transcripts, PCS protocols, math blocks | Protocol schedule — which stages, in what order |
| The `Permutation` seam + sponge/compression | Your pinned permutation params (width, field, constants) |
| The `LinearCode` seam + RS/FRI folds | Your code + rate + coset choice |
| PCS instances (`fri`, `kzg`, `basefold`, …) | The stacking/batching schedule around them |
| The field-generic kernels | The base/extension dtype choice, threaded as data |
| — | Root-claim layout, transcript framing, constraint system, serialization |

Two operational corollaries:

- **Never fork a zorch block to specialize it** — inject your values through
  its seam (a params object, a dtype, a `Protocol` impl, a summand).
- **Scheme-agnostic gaps go upstream first.** If a block is missing and a
  second scheme would want it, contribute it to zorch and depend on it; don't
  grow a private copy.

## Assembly rules (the ones that bite)

- **The transcript is explicit everywhere** — an argument and a result of
  every round call and stage; never ambient state. Prover and verifier stay
  two explicit programs; neither is derived from the other.
- **Role capabilities don't mix.** Proving keys never enter verifier objects,
  verification keys never enter prover objects. Construct only the role you
  deploy; a test needing both builds both.
- **Challenges are carry, not message.** Anything both sides can derive from
  the transcript goes in the carry; only prover→verifier data is a message.
  Name the carry for what it holds (`RunningClaim`, `FoldingClaim`), never
  `carry`.
- **Domain separators are pinned wire format** — stable tags owned by your
  protocol, never derived from class or display names.
- **Not every transcript sequence is a round recurrence.** Setup, a special
  first round, the repeated middle, and terminal claim derivation are distinct
  operations — orchestrate them explicitly in your stage; use
  `prove_rounds`/`verify_rounds` only for the genuinely homogeneous part.
- **Failure semantics.** Malformed proof *shape* → `ValueError`; a
  well-formed proof failing an algebraic/transcript/opening check →
  `VerifyResult(ok=False)`. You decode untrusted bytes into typed proofs
  yourself and translate structural exceptions into rejection.

## Testing your prover (what the types can't enforce)

- Both roles' transcripts agree and both derive the **same reduced claim** at
  every stage boundary.
- Honest proofs verify; mutating **each named proof section** rejects at the
  corresponding stage.
- The verifier constructs with no prover capability.
- Porting a reference prover? **Byte-match, no tolerances** — compare
  canonical (non-Montgomery) values; a hash mismatch is almost always a
  shape/padding delta, not a hash-param bug. Vendor golden fixtures generated
  from the reference, self-validated at generation.
- Exercise a **second parameter shape** (e.g. production vs test sizes) to
  catch assumptions fitted to one configuration.

## Worked example

[`zorch.spartan`](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/schemes/spartan.md)
ships in the package: `SpartanProver` / `SpartanVerifier` compose zerocheck +
lincheck stages, a transcript-only batching step, and PCS-open glue — read it
as the reference for consumer assembly.
