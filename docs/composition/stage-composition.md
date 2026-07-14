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
  stage N+1's carry-in type — the two adjacent stages co-own it. That
  contract is per-adjacent-seam; outputs that cross more than one seam, and
  inputs read by several stages, ride one pipeline-owned carry instead
  ([The pipeline carry](#the-pipeline-carry)).
  Where a reference makes two stages disagree (e.g. the next stage reads only
  the row-variable tail of the previous stage's point, or RLCs openings into
  claims under a fresh challenge), that reshaping is an explicit consumer
  round at the seam, not a silent slice inside the next stage — the seam is
  where the reference's schedule lives, so it must be visible in the chain.
- **Stage-local: the witness, on the `Round` instance.** Traces, circuit
  layers, dense buffers, and other big inputs one stage consumes are
  constructor state, never carry. This split is what bounds a lazy chain's
  peak memory (`ProveChain`'s generator consumption, `round.py`): a stage's
  witness is released once it has proved — provided the chain is built over
  a generator and no host-side reference pins the witness — and only the
  small seam carry survives to the next stage. Witness that several stages
  read is not stage-local — it is a chain-wide input and rides the pipeline
  carry, sized by the discipline there.

The seam type also pins dtypes. The transcript/challenge field is part of
the contract: a scan-shaped stage requires a scan-invariant carry type, and
base-field leaves folded by extension-field challenges promote the carry
mid-scan and fail to trace. Adjacent stages agree on the field embedding at
the seam, in the seam type, not implicitly at the first fold.

## The pipeline carry

The pairwise rule above is exact for an output read by the next stage and
nothing else. The real chains have two flows it does not cover:

- **Skip-level outputs.** A stage's output can be read more than one seam
  downstream: the trace-commit stage's committed-witness residue is consumed
  by the PCS-opening stage at the end of the pipeline, two seams past the
  type a pairwise contract would place it in, the stages between passing it
  through untouched.
- **Chain-wide inputs.** Some values are no stage's output but several
  stages' input — the committed regions and the public values are read by up
  to every stage in the chain.

Pairwise seam types can express both only as pass-through fields: every
intermediate seam re-declares data its stage never touches, and adding one
skip-level field means editing every type between writer and reader. So a
pipeline instead threads **one accumulating carry type** — a single frozen
dataclass owned by the whole chain, one field per seam-crossing value. The
first real pipeline's prover chain and its verifier dual settled on this
shape independently of each other, which is what promoted it from deviation
to sanctioned pattern; the evidence (a writer/reader table of both carries)
and the decision live on
[fractalyze/zorch#240](https://github.com/fractalyze/zorch/issues/240).

What keeps one shared type from decaying into the bag of `Any` this page
warns about:

- **Every field declares its writer and readers.** The carry is the
  pipeline's dataflow graph; the field comments are where a reviewer reads
  it.
- **A stage writes its own fields and passes the rest untouched**
  (`dataclasses.replace` on a frozen carry). Chain-wide inputs are written
  by no stage at all.
- **Unwritten fields are `None` until their writer runs**, and a reader
  fails loud naming the missing predecessor stage. Mis-sequencing a chain is
  a construction bug caught on the first call, not a silent wrong proof.
- **Fields are sized for the chain's lifetime.** A carry field is not
  released until its last reader runs, so a skip-level field carries the
  small residue (the mle and digest tree the open re-encodes from), never
  the bulky recomputable (the blown-up codeword — re-encode at the open). This
  is the peak-memory rule above, restated for state that outlives its stage.
- **The carry is one registered pytree**, unwritten fields empty subtrees,
  so the whole chain can cross a `@jit` boundary as a single donatable
  argument — a property a per-seam type change at every seam would break.

The two shapes are not in tension: pairwise seam types stay the contract for
the agnostic stage rounds zorch ships (`JaggedEvalInputs` →
`JaggedEvalRound`), which cannot know any consumer's pipeline; the
consumer's stage Round adapts between its pipeline carry and those types.
The accumulating carry is the pipeline's composition shape, the pairwise
type the stage's.

## Pipelines as nested chains

Chains nest (`round.py`), so a pipeline is a `ProveChain` of stage rounds,
each possibly a chain (the interaction argument is a chain of layer rounds).
Messages nest correspondingly: the pipeline's proof is one entry per stage,
and a nested chain's entry is its own per-round list.

The verifier is a `VerifyChain` mirroring the prover chain *round for round*,
bridges included. `VerifyChain`'s one-message-per-round check then guarantees
alignment: a stage (or bridge) present on one side and absent on the other
fails loud instead of silently desynchronizing the Fiat-Shamir stream. A
round with nothing to send still emits a `None` message to hold its position —
the placeholder is what keeps the two chains structurally equal.

## The consumer boundary

zorch ships the agnostic blocks; a *consumer* is the code that turns them into
one concrete prover byte-matching a specific reference. The split follows the
implementation-agnostic non-negotiable ([`../CLAUDE.md`](../../CLAUDE.md)): the test
for "does this belong in zorch" is *would a second, unrelated prover reuse it
unchanged?* It is about the generality of the *decision*, not the math — a niche
kernel that defines one scheme's encoding is the consumer's; a
mundane-but-universal structure like a Merkle tree, or a whole reusable PCS
scheme, is zorch's.

| Concern | zorch (generic) | Consumer (one scheme / application) |
| --- | --- | --- |
| Fiat-Shamir | the transcript + `grind` / `check_witness` | the rate/field parameterization and the observe/sample *order* |
| Hashing | the `Permutation` seam, sponge, compression | the concrete permutation params (constants, width, field) |
| Commitment | the Merkle tree(s) + reusable query/opening layout | which columns are committed, and the layout *schedule* |
| Codes / PCS | the `LinearCode` / `PcsProver` / `PcsVerifier` seams + reusable instances and fold machinery | the per-prover stacking/region/batching schedule, any scheme-specific fold |
| Sumcheck | the scan driver + per-variable rounds | the `combine` summand and the round wiring |
| Composition | the `Round` abstraction + chains | the actual stage sequence and the carry between stages |
| Constraints / field | works over any field dtype | the constraint system, the quotient/zero-check shape, the base/extension dtypes |

A consumer never forks a block; it supplies only the differing *values* and
*order* through four injection points: (1) a params object behind the
`Permutation` seam, (2) the field dtype threaded as data, (3) the
`PcsProver` / `PcsVerifier` protocols over the shared fold machinery, and (4) a
consumer-owned driver threading `Round`s. A scheme-agnostic gap goes *upstream*
into zorch first, then the consumer depends on it — never a fork.

## Bridges — consumer-owned scheme glue

zorch ships the agnostic stage rounds and the chain machinery; the consumer
owns everything that encodes its reference's exact transcript (the
implementation-agnostic non-negotiable above). A **bridge** is a small,
transcript-only `Round` between stages — a PoW grind, a reference's
sampled-and-discarded challenge, length-prefixed observes. Bridges compose into
the consumer's chain as consumer-local `Round`s rather than as free-floating
statements in a driver function:

- a grind round emits the witness as its message; its verifier dual re-checks
  the proof-of-work bits;
- a discard round samples and drops; its dual replays the same sample;
- an observe round absorbs with the reference's exact framing; its dual
  absorbs the same values from the proof.

Modeling bridges as rounds is what gives the verifier its mirror for free: the consumer
writes the schedule once as a list of rounds, and the dual chain has exactly
one slot per step. The alternative — bridges inlined in `prove`, re-derived by
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
unit (the [fusion north star](../README.md#fusion-north-star)); the chain stays
a host-side Python loop, exactly as the per-layer chain is. Bridges move
no kernel boundary — their absorbs and squeezes already execute between stage
captures; the rounds change bookkeeping, not lowering. A bridge is
transcript-only and does not warrant its own capture.

## Out of scope

- **A monadic transcript DSL.** `Round`'s plain tuple threading already
  composes; what composition needs is the seam contract, not more
  abstraction.
- **A universal proof container.** Proofs stay the chain's nested messages;
  framing them for the wire is per-consumer.
- **Implementation tracking** lives on
  [fractalyze/zorch#155](https://github.com/fractalyze/zorch/issues/155).
