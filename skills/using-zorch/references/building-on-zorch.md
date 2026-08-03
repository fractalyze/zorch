# Building a prover on zorch

Your project (a zkVM, zkML, zkTLS prover, …) is a **consumer**: it assembles
zorch blocks into one concrete proof system, usually byte-matching a reference
implementation. This page is the boundary contract and the assembly rules,
condensed from
[stage-composition.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/composition/stage-composition.md).

## The boundary

> A component belongs in zorch when a second, unrelated prover would reuse it
> unchanged. The test is the generality of the *decision*, not the math.

| zorch owns (import it) | Your repo owns (write it) |
| --- | --- |
| The transcript + `grind`/`check_witness` | The rate/field parameterization and the observe/sample **order** |
| The `Permutation` seam + sponge/compression | Your pinned permutation params (constants, width, field) |
| Merkle trees + reusable query/opening layout | Which columns are committed; the layout schedule |
| `LinearCode`/`PcsProver`/`PcsVerifier` seams + instances + fold machinery | The stacking/region/batching schedule; any scheme-specific fold |
| Sumcheck scan driver + per-variable rounds | The `combine` summand and the round wiring |
| The `Round` abstraction + chains | The actual stage sequence and the carry between stages |
| Field-generic kernels | The constraint system, quotient/zero-check shape, base/extension dtypes |

A consumer **never forks a block** — it supplies only the differing values and
order through four injection points: a params object behind the `Permutation`
seam; the field dtype threaded as data; the `PcsProver`/`PcsVerifier`
protocols over the shared fold machinery; a consumer-owned chain of `Round`s.
A scheme-agnostic gap goes **upstream into zorch first**, then your repo
depends on it.

## The carry contract (where state lives)

- **Uniform: the transcript** — threaded through every round by the chain;
  never inside a carry or a message.
- **Seam-crossing: a typed pytree per seam.** Stage N's carry-out type *is*
  stage N+1's carry-in type. Reshaping between two stages (slicing a point,
  RLC-ing claims) is an explicit consumer round at the seam — never a silent
  slice inside the next stage. The seam type also pins the field embedding;
  base-field carry folded by extension challenges promotes mid-scan and fails
  to trace.
- **Stage-local: the witness, on the `Round` instance** — traces and dense
  buffers are constructor state, never carry. Build `ProveChain` over a
  generator so each stage's witness is released once proved.

## Bridges — your scheme glue, as rounds

Everything that encodes the reference's exact transcript — a PoW grind, a
sampled-and-discarded challenge, length-prefixed observes — is a **`Bridge`**:
a transcript-only `Round` in your chain, with a verifier dual replaying the
same ops. Writing the schedule once as a list of rounds gives the verifier
its mirror for free; inlining those steps in `prove` and re-deriving them in
`verify` duplicates the schedule, and drift between copies surfaces only as an
end-to-end byte-match failure.

**Stays outside rounds:** transcript-free data preparation (circuit
construction, dense packing) — host-side, feeding rounds at construction. A
round exists to put a step on the Fiat-Shamir schedule.

## Failure semantics

Malformed proof *shape* raises `ValueError` (e.g. `VerifyChain` on a
message-count mismatch); an algebraic/transcript failure comes back as the
`ok` array a `VerifyChain` ANDs. You decode untrusted bytes into typed proofs
yourself and translate structural exceptions into rejections.

## Testing your prover (what the types can't enforce)

- Both roles' transcripts agree and both derive the same reduced claim at
  every seam.
- Honest proofs verify; mutating each named proof section rejects.
- Porting a reference? **Byte-match, no tolerances** — compare canonical
  (non-Montgomery) values; a hash mismatch is almost always a shape/padding
  delta, not a hash-param bug. Vendor golden fixtures from the reference,
  self-validated at generation.
- Exercise a second parameter shape (production vs test sizes) to catch
  assumptions fitted to one configuration.

## Worked example

[`zorch.spartan`](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/schemes/spartan.md)
ships in the package: zerocheck + lincheck stages, an RLC bridge, PCS-open
glue over `ProveChain`/`VerifyChain` — read it as the reference for consumer
assembly.
