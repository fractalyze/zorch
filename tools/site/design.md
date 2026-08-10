# Design digest

The load-bearing decisions, compressed. Each row links the full rationale on
GitHub — read those when you need the *argument*, not just the rule.

## The two non-negotiables

1. **Scheme- and implementation-agnostic.** No block assumes a proving scheme
   or any one downstream prover (a zkVM, zkML, zkTLS). Anything
   scheme-specific lives in the *consumer*; the test for "belongs in zorch"
   is *would a second, unrelated prover reuse it unchanged?*
2. **Fusion is a correctness-of-design property.** A round, an
   `absorb`/`squeeze`, a `commit`/`open`, a fold step, and a hash permutation
   must each lower to **one replayable device unit by construction** — never
   by hoping a compiler pattern-match recovers it.

## Fusion north star, compressed

The unit is a **captured device graph replay**, not necessarily one kernel:
measured today, a sumcheck round lowers to two kernels (round poly = one
reduction kernel, fold = one element-wise kernel) and that is fine — what
breaks the contract is a **host round-trip mid-round**. Two enablers keep a
round capturable: the whole body in one traced region, and a **device-side
transcript** (`DuplexTranscript`) so `observe`/`sample` are device ops.
Bodies are written *fusion-ready* — element-wise field ops plus the one
inherent `Σ`.
[Full statement](https://github.com/fractalyze/zorch/blob/main/docs/README.md#fusion-north-star)

## One decision per block

| Block | The decision that shapes it | Full WHY |
| --- | --- | --- |
| Rounds & stages | Only the **message** crosses roles; challenges both sides derive ride the carry. A stage's malformed-shape input raises `ValueError`; an algebraic failure returns `ok=False`. | [stage-composition](https://github.com/fractalyze/zorch/blob/main/docs/composition/stage-composition.md) |
| Transcript | Two flavours, on purpose: device-algebraic (`DuplexTranscript`, keeps rounds capturable) vs host-byte (SHA-256, for byte-matching references). | [transcript](https://github.com/fractalyze/zorch/blob/main/docs/blocks/transcript.md) |
| Polynomials | `eq` ships two forms (prover's 2ⁿ expansion, verifier's O(n) closed form); the additive Basefold fold and the multilinear bind are **separate functions** because conflating them is a silent bug. | [poly](https://github.com/fractalyze/zorch/blob/main/docs/blocks/poly.md) |
| Hashing | One `Permutation` seam; a consumer ships only a params object (width, field, constants) — sponge and compression code never fork. Lives in `hash-frx`, below zorch, so a signature scheme can take it without a proving-block dependency. | [hash-frx](https://github.com/fractalyze/hash-frx/blob/main/docs/blocks/hash.md) |
| Merkle | A binary tree over Sponge (leaves) + Compression (nodes) — commitment reuses the hash seams instead of owning hashing. | [commit](https://github.com/fractalyze/zorch/blob/main/docs/blocks/commit.md) |
| Codes | `LinearCode` is a seam; the FRI fold lives with the **code** (it is a codeword-domain operation), not inside any PCS. | [coding](https://github.com/fractalyze/zorch/blob/main/docs/blocks/coding.md) |
| PCS | A PCS is a **committer plus an opening stage**; FRI/BaseFold/WHIR/… are instances sharing the fold machinery, so a consumer swaps schemes without new plumbing. | [pcs](https://github.com/fractalyze/zorch/blob/main/docs/blocks/pcs.md) |
| Sumcheck | The smallest complete stage; zerocheck, lincheck and LogUp-GKR **configure** it (a summand + round wiring) rather than reimplement it. | [sumcheck](https://github.com/fractalyze/zorch/blob/main/docs/blocks/sumcheck.md) |
| LogUp-GKR | A fractional-sum circuit reducing a public output claim layer by layer to an input claim a PCS opening discharges. | [logup-gkr](https://github.com/fractalyze/zorch/blob/main/docs/blocks/logup-gkr.md) |
| Spartan | The worked composite, shipped as importable code — proof that the blocks assemble into a full SNARK without private plumbing. | [spartan](https://github.com/fractalyze/zorch/blob/main/docs/schemes/spartan.md) |

## If you remember three things

- Inject parameters through seams; **never fork a block**.
- The transcript is explicit everywhere; **challenges are carry, not
  message**.
- Keep round bodies element-wise + one `Σ`; **no host round-trips
  mid-round**.
