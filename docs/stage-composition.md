# Stage composition — design notes

How a multi-stage prover (interaction argument → constraint sumcheck → PCS
opening) composes from `Round`s, and what crosses each stage seam. The
per-block docs cover the stages themselves; this page covers the seams. It is
the decision record for the stage-level composition protocol
([fractalyze/zorch#155](https://github.com/fractalyze/zorch/issues/155));
design lineage on epic
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

______________________________________________________________________

## Why this shape

A real shard prover sequences heterogeneous stages, each with its own claims,
points, and witness. With one or two stages, a hand-written driver threading
`(transcript, state)` is fine; at three and beyond, every consumer rewrites the
same schedule logic, and each rewrite is a fresh chance for the prover and
verifier transcripts to drift apart at a seam. The composition protocol exists
to make the seams explicit and the duals structurally aligned.

**The transcript is the only universal carry.** Every seam *has* claims and a
point, which makes them look like the uniform part — but their shapes are
seam-specific: logup-gkr hands `(numerator eval, denominator eval, point)`, a
zerocheck consumes that point plus per-chip openings and hands per-column
claims and a row point, a jagged opening consumes claims plus row/column
points and the committed buffer. The transcript is the one object that crosses
every seam unchanged in type. A universal "stage carry" type covering the rest
would be a bag of `Any` — structure lost, nothing checked. So claims and
points are *carried but pairwise-typed*, and the contract is per-seam.

## The stage carry contract

A stage is a `ProverRound`; the spine stays
`(carry, transcript) -> (carry, transcript, msg)` with no new chain machinery.
Three placement rules say where each piece of state lives:

- **Uniform: the transcript.** Threaded through every round by the chain;
  never inside a carry or a message.
- **Seam-crossing: a typed pytree per seam** (a frozen dataclass, or a small
  named tuple where the seam is three arrays). Stage N's carry-out type *is*
  stage N+1's carry-in type — the two adjacent stages co-own it.
  Where a reference makes two stages disagree (e.g. the next stage reads only
  the row-variable tail of the previous stage's point, or RLCs openings into
  claims under a fresh challenge), that reshaping is an explicit consumer
  round at the seam, not a silent slice inside the next stage — the seam is
  where the reference's schedule lives, so it must be visible in the chain.
- **Stage-local: the witness, on the `Round` instance.** Traces, circuit
  layers, dense buffers, and other big inputs are constructor state, never
  carry. This split is what bounds a lazy chain's peak memory (`ProveChain`'s
  generator consumption, `round.py`): a stage's witness is released once it
  has proved — provided the chain is built over a generator and no host-side
  reference pins the witness — and only the small seam carry survives to the
  next stage.

The seam type also pins dtypes. The transcript/challenge field is part of
the contract: a scan-shaped stage requires a scan-invariant carry type, and
base-field leaves folded by extension-field challenges promote the carry
mid-scan and fail to trace. Adjacent stages agree on the field embedding at
the seam, in the seam type, not implicitly at the first fold.

## Pipelines as nested chains

Chains nest (`round.py`), so a pipeline is a `ProveChain` of stage rounds,
each possibly a chain (the interaction argument is a chain of layer rounds).
Messages nest correspondingly: the pipeline's proof is one entry per stage,
and a nested chain's entry is its own per-round list.

The verifier is a `VerifyChain` mirroring the prover chain *round for round*,
glue included. `VerifyChain`'s one-message-per-round check then guarantees
alignment: a stage (or glue step) present on one side and absent on the other
fails loud instead of silently desynchronizing the Fiat-Shamir stream. A
round with nothing to send still emits a `None` message to hold its position —
the placeholder is what keeps the two chains structurally equal.

## Consumer-owned scheme glue

zorch ships the agnostic stage rounds and the chain machinery; the consumer
owns everything that encodes its reference's exact transcript (the one rule,
[`building-a-zkvm-prover.md`](building-a-zkvm-prover.md)). The glue between
stages — a PoW grind, a reference's sampled-and-discarded challenge,
length-prefixed observes — composes into the consumer's chain as small
consumer-local `Round`s rather than as free-floating statements in a driver
function:

- a grind round emits the witness as its message; its verifier dual re-checks
  the proof-of-work bits;
- a discard round samples and drops; its dual replays the same sample;
- an observe round absorbs with the reference's exact framing; its dual
  absorbs the same values from the proof.

Glue-as-rounds is what gives the verifier its mirror for free: the consumer
writes the schedule once as a list of rounds, and the dual chain has exactly
one slot per step. The alternative — glue inline in `prove`, re-derived by
hand in `verify` and again in every diagnostic harness — duplicates the
schedule at every copy, and a drift between copies is invisible until a
byte-match fails end to end.

**What stays outside rounds:** pure data preparation with no transcript
access — circuit and pyramid construction, dense packing, indicator
materialization. These run host-side and feed stage rounds at construction.
A round exists to put a step *on the Fiat-Shamir schedule*; transcript-free
computation inside one would smuggle eager work into the IOP order for no
soundness gain.

## Fusion by construction

Stage granularity is capture granularity. Each stage round's body remains one
traced region over the device-side transcript, so it lowers to one replayable
unit (the [fusion north star](README.md#fusion-north-star)); the chain stays
a host-side Python loop, exactly as the per-layer chain is. Glue rounds move
no kernel boundary — their absorbs and squeezes already execute between stage
captures; the rounds change bookkeeping, not lowering. A glue round is
transcript-only and does not warrant its own capture.

## Out of scope

- **A monadic transcript DSL.** `Round`'s plain tuple threading already
  composes; what composition needs is the seam contract, not more
  abstraction.
- **A universal proof container.** Proofs stay the chain's nested messages;
  framing them for the wire is per-consumer.
- **Implementation tracking** lives on
  [fractalyze/zorch#155](https://github.com/fractalyze/zorch/issues/155).
