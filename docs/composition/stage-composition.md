# Stage composition

ZK protocols compose at two different scales:

- a **round** repeats one recurrence, such as eliminating a sumcheck variable or
  reducing one GKR layer;
- a **stage** is one paired prover/verifier component, such as zerocheck,
  lincheck, a PCS opening, or a complete proof system.

Keeping those scales separate makes it clear which code repeats an algorithm,
which code owns a proof boundary, and where proof-system dataflow belongs.

## Round: repeat one contract

Every round has one generic transition contract:
`(carry, transcript, incoming)` maps to `(carry, transcript, outgoing)`.
A prover round receives `None` and emits its proof message; the verifier round
receives that message and emits its `ok` verdict. `ProverRound` and
`VerifierRound` are type aliases specializing this one `Round` protocol. The
`incoming` position remains required: `None` is an explicit unit input for the
prover role, while a verifier must supply the corresponding proof message.

Use `prove_rounds` and `verify_rounds` when every step shares that recurrence
contract. The concrete rounds may hold different data and produce
shape-varying messages; what stays invariant is the meaning of the carry and
message passed from one step to the next. Sumcheck and polynomial folding have
specialized drivers because their recurrences impose stronger shape and fusion
requirements.

A sequence of transcript-bearing operations is not automatically a round
recurrence. Setup, a special first round, a repeated sumcheck, and terminal claim
derivation are distinct operations even when they execute consecutively. Write
that orchestration explicitly inside the owning stage.

Rounds are normally implementation details of a stage. A stage chooses its
prover and verifier recurrences and packages their messages into its proof type.

## Stage: pair one protocol component

`Stage` is the common base class for complete paired protocol components. Every
stage implements both methods:

```python
prove(prover_input, transcript) -> ProveResult[prover_output, proof]
verify(verifier_input, proof, transcript) -> VerifyResult[verifier_output]
```

Each side accepts one semantic domain value. For example, Spartan's inner stage
accepts `LincheckWitness` while its verifier accepts `BatchedClaims`; `None` is
used when a side needs no input beyond the proof and transcript. Prover and
verifier inputs and outputs are intentionally different types because the two
sides have different knowledge.

A stage owns:

- the pairing between its prover and verifier;
- one proof type;
- typed values exported to its parent protocol;
- reusable static configuration;
- an independently testable protocol boundary.

A stage may drive round recurrences, perform PCS operations, or own child
stages. Compilation boundaries are an independent performance decision. A
prover-only helper, transcript schedule, or performance region is an ordinary
function or class rather than an incomplete stage with a placeholder verifier.

## Composition is explicit

A composite stage owns child stages and writes their dataflow in ordinary
`prove` and `verify` methods, like a PyTorch module with a custom `forward`:

```text
                         +---- outer ---- batch ---- inner ----+
statement / witness ----+                                      +---- opening
                         +---- commitment / PCS data -----------+
```

This is the default composition model, not an exceptional DAG escape hatch.
Execution order is often linear while dataflow is not: commitments survive to a
later opening, claims feed more than one consumer, and original statement or
witness data remains live across several phases.

The parent constructs each child's semantic input from named values:

```python
outer = self.outer.prove(outer_witness, transcript)
batch, transcript = batch_claims(outer.output.claims, outer.transcript)
inner = self.inner.prove(
    LincheckWitness(instance, assignment, outer.output, batch),
    transcript,
)
```

A child output therefore does not need to equal the next child's complete input.
The parent can combine it with statement data, witness data, or another earlier
result without an adapter stage or accumulating context object.

`Spartan` is the reference composite stage. Its outer result feeds both the
inner stage and the terminal opening claim, while the witness commitment and
PCS prover data skip directly to the opening stage. Its proof is a frozen
dataclass with named child proof fields.

## State and ownership rules

- The transcript is explicit in every round call and stage result.
- Static configuration belongs on reusable round or stage instances.
- Per-proof statements and witnesses are semantic input dataclasses.
- Stage outputs contain only values the parent protocol consumes.
- Prover-only state and transmitted proof data remain distinct.
- Skip-level values remain named locals in a composite stage.
- Transcript-only schedule operations are shared named functions called by both
  prover and verifier paths.
- Proof serialization is separate from execution and follows the named proof
  structure.
- There is no shared bridge or universal protocol context to populate over time.

## Reuse boundaries

Reuse the exact stage when the cryptographic subprotocol and transcript schedule
match. Reuse rounds and recurrence drivers when the repeated transition matches
but the surrounding framing differs. Reuse lower mathematical, transcript, and
PCS primitives when even the recurrence is protocol-specific.

A component should expose only its intrinsic dependencies. It should not know
which sibling runs before it or carry unrelated values merely because another
component needs them later. Consumer-specific schedules remain in consumer
composite stages; reusable protocol components remain in zorch.

## Testing the pairing

Tests should pin the properties the abstractions cannot enforce themselves:

- prover and verifier transcripts agree at every component boundary;
- honest proofs verify;
- mutating each named proof section rejects;
- alternate injected round or PCS implementations preserve the stage contract;
- compile count, runtime, and peak memory do not regress.

## Consumer boundary

zorch supplies reusable stages, round drivers, transcripts, PCS protocols, and
mathematical blocks. A consumer owns its concrete protocol schedule, statement
layout, transcript framing, and proof serialization. A component belongs in
zorch when another proof system can reuse it without inheriting one consumer's
private orchestration.
