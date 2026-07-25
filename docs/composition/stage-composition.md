# Stage composition

ZK protocols compose at two different scales:

- a **round** repeats one recurrence, such as eliminating a sumcheck variable or
  reducing one GKR layer;
- a **stage** is one paired prover/verifier component, such as zerocheck,
  lincheck, LogUp-GKR, a wrapper around a PCS opening, or a full proof system.

Keeping those scales separate makes it clear which code repeats an algorithm,
which code owns a proof boundary, and where proof-system dataflow belongs.

## Round: repeat one contract

The two roles are separate protocols, because they carry different data.
`ProverRound` maps `(carry, transcript)` to `(carry, transcript, message)`.
`VerifierRound` maps `(carry, transcript, message)` to
`(carry, transcript, ok)`.

Only the message crosses between the roles, so only the message gets a position
of its own. A value both sides can derive — any challenge squeezed from the
transcript — belongs in the carry instead: a sumcheck round records its
challenge into the point it is building, a GKR layer folds its own into the
running claim, and both report only a verdict. That is what lets one verifier
protocol serve every recurrence shape rather than each driver needing its own
round type. Name the carry for what it holds (`RunningClaim`, `LayerClaim`,
`FoldState`), never `carry`.

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

## Stage: reduce one claim

A stage is one mathematical proof-reduction contract implemented by two
separately deployable role interfaces:

```python
ProverStage.prove(claim, witness, transcript)
    -> ProveResult[reduced_claim, reduction_proof]
VerifierStage.verify(claim, reduction_proof, transcript)
    -> VerifyResult[reduced_claim]
```

`Stage(prover, verifier)` is an optional pairing for conformance tests and local
orchestration. It is not a deployment handle: prover and verifier objects may
hold asymmetric capabilities, so deployed code constructs only its role. This
is load-bearing for preprocessing schemes whose PCS proving key is large while
the verification key is small.

A **claim** is the public assertion entering the stage. A **witness** is the
private evidence used by the prover. Both sides derive the same **reduced
claim**. The reduction proof does not normally establish that reduced claim;
it establishes the source claim *conditional on* the reduced claim:

```text
reduced claim is true  =>  source claim is true
```

up to the reduction's soundness error. A later stage proves the reduced claim,
and a terminal stage closes the final claim. “Statement” is useful prose for a
proof system's root claim, but is not a separate code concept. “Reduction”
names the operation performed by a stage, not a generic result object.

`ProveResult.reduced_claim` is an execution result used to continue the prover;
it is not serialized separately. The verifier independently reconstructs the
same value from the source claim, reduction proof, and transcript.

The stage contract owns:

- the pairing between its prover and verifier role types;
- one proof type;
- one source-claim and reduced-claim contract shared by both roles;
- reusable static configuration;
- an independently testable protocol boundary.

A role implementation may drive round recurrences, perform PCS operations, or
own child role implementations. Compilation boundaries are an independent
performance decision. A prover-only helper, transcript schedule, or performance region is an ordinary
function or class rather than an incomplete stage with a placeholder verifier.

## Composition is explicit

A composite prover and verifier own only their corresponding child roles and
write their dataflow in ordinary `prove` and `verify` methods, like PyTorch
modules with custom `forward` methods:

```text
claim + witness --------+---- outer ---- batch ---- inner ----+
                        +---- commitment / PCS data -----------+---- opening
```

This is the default composition model, not an exceptional DAG escape hatch.
Execution order is often linear while dataflow is not: commitments survive to a
later opening, claims feed more than one consumer, and root-claim or witness
data remains live across several phases.

The parent constructs each child's claim and witness from named values:

```python
outer = self.outer.prove(outer_claim, outer_witness, transcript)
batch, transcript = batch_claims(outer.reduced_claim.values, outer.transcript)
inner_claim = LincheckClaim(instance, outer.reduced_claim, batch)
inner = self.inner.prove(
    inner_claim,
    LincheckWitness(assignment),
    transcript,
)
```

A child reduced claim therefore need not be the next child's complete source
claim. The parent can combine it with root-claim data or another earlier claim.
Private skip-level values remain explicit parent locals and can contribute to a
later witness without entering the public claim.

`SpartanProver` and `SpartanVerifier` are the reference composite roles. Their
outer reduced claim feeds both the inner role and the terminal opening claim,
while the witness commitment and PCS prover data skip directly to the prover's
opening role. `LogUpGkrProver` and `LogUpGkrVerifier` are the second production
pair: their source claim owns the public output and layer count, while their
reduced claim is the input-layer claim for a consumer's PCS opening. Both proofs use frozen dataclasses with named sections.

## State and ownership rules

- The transcript is explicit in every round call and stage result.
- Shared static configuration belongs on corresponding role instances.
- Role-specific capabilities belong only on that role: proving keys never enter
  verifier objects, and verification keys never enter prover objects.
- Per-proof claims and witnesses are separate semantic input dataclasses.
- Prover and verifier produce the same reduced-claim type.
- Reduction proofs are conditional proofs of their source claims; they do not
  establish their reduced claims.
- Prover-only state and transmitted proof data remain distinct.
- Skip-level values remain named locals in a composite stage.
- Transcript-only schedule operations are shared named functions called by both
  prover and verifier paths.
- Stages do not derive transcript domain separators from class or display
  names. The owning protocol specifies stable tags and framing as part of its
  wire format; reusable framing belongs upstream only once that encoding is
  shared by more than one protocol.
- Proof serialization is separate from execution and follows the named proof
  structure.
- There is no shared bridge or universal protocol context to populate over time.

“No bridge” does not mean these operations are security-neutral. Claim batching
is a randomized reduction and grinding amplifies security; both belong in the
parent's soundness accounting. They are functions rather than stages because
they do not own an independently reusable paired proof section.

## Verification failures

Verification distinguishes structure from cryptographic validity:

- malformed inputs whose static proof shape cannot represent the configured
  protocol raise `ValueError`;
- well-formed proofs that fail an algebraic, transcript, or opening check return
  `VerifyResult(ok=False)`.

Wire-facing callers must validate/decode untrusted bytes into the expected
static proof shape and translate structural exceptions into their external
rejection response. Stage methods operate on typed in-memory proofs rather than
serving as a non-throwing byte parser.

## Reuse boundaries

Reuse the exact stage when the cryptographic subprotocol and transcript schedule
match. Reuse rounds and recurrence drivers when the repeated transition matches
but the surrounding framing differs. Reuse lower mathematical, transcript, and
PCS primitives when even the recurrence is protocol-specific.

A component should expose only its intrinsic dependencies. It should not know
which sibling runs before it or carry unrelated values merely because another
component needs them later. Consumer-specific schedules remain in consumer
composite stages; reusable protocol components remain in zorch.

Univariate skip illustrates why the reduced claim is part of that contract.
It is an ordinary sumcheck stage, but its first reduced coordinate binds a
subgroup interpolation of several Boolean variables. It therefore exports a
prism-evaluation claim, not an ordinary multilinear evaluation claim. A parent
expecting an ordinary MLE claim cannot silently substitute it.

A stage can also be applied recurrently. A folding parent threads its
accumulator through repeated calls to one fold stage, and the stage's semantic
input can contain a k-ary batch of instances. Recurrence is a use of the paired
component, not a second inheritance relationship.

## Testing the pairing

Tests should pin the properties the abstractions cannot enforce themselves:

- prover and verifier transcripts agree at every component boundary;
- prover and verifier derive the same reduced claim at every boundary;
- honest proofs verify;
- mutating each named proof section rejects;
- alternate injected round or PCS implementations preserve the role contract;
- a verifier role can be constructed without any prover capability;
- compile count, runtime, and peak memory do not regress.

The parent prover and verifier remain two explicit programs; one is not derived
from the other. This keeps their knowledge boundaries honest, while making
transcript-boundary and structural-proof tests mandatory.

## Consumer boundary

zorch supplies reusable stages, round drivers, transcripts, PCS protocols, and
mathematical blocks. A consumer owns its concrete protocol schedule, root-claim
layout, transcript framing, and proof serialization. A component belongs in
zorch when another proof system can reuse it without inheriting one consumer's
private orchestration.
