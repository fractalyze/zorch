# Stage composition

ZK protocols compose at two different scales:

- a **round** repeats one recurrence, such as eliminating a sumcheck variable or
  reducing one GKR layer;
- a **stage** is one paired prover/verifier component, such as zerocheck,
  lincheck, LogUp-GKR, a wrapper around a PCS opening, or a full proof system.

Keeping those scales separate makes it clear which code repeats an algorithm,
which code owns a proof boundary, and where proof-system dataflow belongs.

## Round: repeat one contract

The two roles are separate protocols (`zorch/round.py`), because they carry
different data. `ProverRound` maps `(carry, transcript)` to
`(carry, transcript, message)`. `VerifierRound` maps
`(carry, transcript, message)` to `(carry, transcript, ok)`.

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

A stage is one proof-reduction contract implemented by two separately deployable
role interfaces (`zorch/stage.py`):

```python
ProverStage.prove(claim, witness, transcript)
    -> ProveResult[reduced_claim, reduction_proof]
VerifierStage.verify(claim, reduction_proof, transcript)
    -> VerifyResult[reduced_claim]
```

No paired handle bundles the two. Roles hold asymmetric capabilities, so code
constructs only the one it deploys — load-bearing where a PCS proving key is
large and the verification key small. A test needing both builds both.

The **claim** is the public assertion entering the stage, the **witness** the
prover's private evidence. Both sides derive the same **reduced claim**, and the
reduction proof establishes the source claim *conditional on* it:

```text
reduced claim is true  =>  source claim is true
```

up to the reduction's soundness error. A later stage proves the reduced claim; a
terminal stage reduces to `TrivialClaim`, which holds by construction, so an
argument of knowledge is exactly a reduction to the trivial claim. Naming the
terminal case as a type rather than `None` is what lets a parent read back that
the chain ends there.

`ProveResult` carries the reduced claim, its proof, and the advanced transcript;
`VerifyResult` carries the verifier's own reduced claim, transcript, and verdict.
The reduced claim is an execution value, never serialized — the verifier
reconstructs it from the source claim, proof, and transcript.

The stage contract owns the pairing between its role types, one proof type, one
source- and reduced-claim contract shared by both roles, reusable static
configuration, and an independently testable protocol boundary. A role may drive
round recurrences, perform PCS operations, or own child roles; compilation
boundaries are an independent performance decision. A prover-only helper or
transcript schedule is an ordinary function, not an incomplete stage with a
placeholder verifier.

## Which one is it?

The recurring question is not what a stage is, but which of four things a new
step is. Answer it by what the step *does to a claim*, never by how big it is:

| The step… | is a | because |
| --- | --- | --- |
| takes a claim and produces a weaker one both roles can derive | **stage** | it owns a reduction, so it owns a proof section and a role pair |
| repeats one transition, carrying the same meaning each time | **round** | its contract is the carry and the message, not a claim boundary |
| runs before any claim exists and produces prover-only data | **committer** | its output cannot ride a reduced claim; it reaches later roles through their witness |
| only advances the transcript | **shared function** | it proves nothing independently, so both roles call the same named function |

The last two are the ones mistaken for stages. A commitment is the object a
later claim is *about*, so it precedes the first claim rather than reducing one;
it stays a local in the composite and enters a later role as witness. A domain
separator, a grind, a framed observation, or a sampled batching challenge changes
the transcript and can change soundness, but owns no independently reusable proof
section — so it is a function both roles call at the same point, with its
security contribution documented there.

One shape is mistaken for a round: a step whose carry comes out unchanged. If the
carry is the same on the way out, nothing recurred, and the step is a function.

## Composition is explicit

A composite prover and verifier own only their child roles and write their
dataflow in ordinary `prove` and `verify` methods, like PyTorch modules with
custom `forward` methods:

```text
claim + witness --------+---- outer ---- batch ---- inner ----+
                        +---- commitment / PCS data -----------+---- opening
```

This is the default model, not a DAG escape hatch: execution order is often
linear while dataflow is not — commitments survive to a later opening, claims
feed more than one consumer, and root-claim data stays live across phases. The
parent builds each child's claim and witness from named values, so a child's
reduced claim need not be the next child's complete source claim, and private
skip-level values stay parent locals that reach a later witness without entering
any public claim.

```python
outer = self.outer.prove(outer_claim, outer_witness, transcript)
batch, transcript = batch_claims(outer.reduced_claim.values, outer.transcript)
inner_claim = LincheckClaim(instance, outer.reduced_claim, batch)
inner = self.inner.prove(inner_claim, LincheckWitness(assignment), transcript)
```

`SpartanProver` / `SpartanVerifier` are the reference composite roles: the outer
reduced claim feeds both the inner role and the terminal opening claim, while the
commitment and PCS prover data skip straight to the prover's opening role.
`LogUpGkrProver` / `LogUpGkrVerifier` are the second production pair, reducing a
public output-and-layer-count claim to an input-layer claim for a consumer's PCS
opening.

The jagged layout reuses those same claim types — a consumer drives
`jagged_prover`'s layer rounds from its own seam. Only the witness and layer
proofs differ, which generalizes:
**per-input structure is witness, per-class structure is role configuration** —
the jagged fold schedule rides the witness, the round width caps configure the
rounds once.

## State and ownership rules

- The transcript is explicit in every round call and stage result.
- Role-specific capabilities stay on that role: proving keys never enter verifier
  objects, verification keys never enter prover objects.
- Claims and witnesses are separate per-proof dataclasses; both roles produce the
  same reduced-claim type.
- Prover-only state and transmitted proof data stay distinct, and skip-level
  values stay named locals in the composite.
- Transcript-only operations are shared named functions both paths call.
- Domain separators are never derived from class or display names — the owning
  protocol specifies stable tags as part of its wire format, and reusable framing
  moves upstream only once two protocols share the encoding.
- Serialization is separate from execution and follows the named proof structure.
- There is no shared bridge or universal context populated over time.

"No bridge" is not a claim of security-neutrality: claim batching is a randomized
reduction and grinding amplifies security, so both belong in the parent's
soundness accounting. They are functions because they own no independently
reusable paired proof section.

## Verification failures

A malformed input whose static proof shape cannot represent the configured
protocol raises `ValueError`; a well-formed proof failing an algebraic,
transcript, or opening check returns `VerifyResult(ok=False)`. Stage methods take
typed in-memory proofs and are not non-throwing byte parsers, so a wire-facing
caller decodes untrusted bytes into the expected shape and translates structural
exceptions into its own rejection response.

## Reuse boundaries

Reuse the exact stage when the subprotocol and transcript schedule match; reuse
rounds and drivers when the transition matches but the framing differs; reuse the
mathematical, transcript, and PCS primitives when even the recurrence is
protocol-specific.

A component exposes only its intrinsic dependencies — it does not know which
sibling ran before it, nor carry values only a later component needs. Univariate
skip shows why the reduced claim is part of that contract: an ordinary sumcheck
stage, but its first reduced coordinate binds a subgroup interpolation, so it
exports a prism-evaluation claim that a parent expecting an ordinary MLE claim
cannot silently substitute. A stage may also be applied recurrently — a folding
parent threading its accumulator through repeated calls — which is a use of the
paired component, not a second inheritance relationship.

## Testing the pairing

Pin what the abstractions cannot enforce: that both roles' transcripts agree and
both derive the same reduced claim at every boundary; that honest proofs verify
and mutating each named proof section rejects; that an alternate injected round
or PCS preserves the contract; that a verifier constructs with no prover
capability; and that compile count, runtime, and peak memory do not regress.

Parent prover and verifier stay two explicit programs, neither derived from the
other — which keeps their knowledge boundaries honest and makes those
transcript-boundary and structural tests mandatory.

## Consumer boundary

zorch supplies reusable stages, round drivers, transcripts, PCS protocols, and
mathematical blocks. A consumer owns its protocol schedule, root-claim layout,
transcript framing, and serialization. A component belongs in zorch when another
proof system can reuse it without inheriting one consumer's private
orchestration.
