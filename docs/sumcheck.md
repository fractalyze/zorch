# sumcheck — design notes

The *why* behind the sumcheck block. The *what* already lives in two drift-proof
places: the code (`zorch/sumcheck/`, `zorch/prove.py`, `zorch/verify.py`) and the
tests under `zorch/sumcheck/testing/`, which are the executable usage and run on
every commit. This file carries only what neither can. Full design and open
decisions: epic issue
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Why the shape

**One summand seam, many summands, one verifier.** The folding skeleton is shared
as free functions — `split_halves` / `factors_on_domain` / `fold` — and each round
supplies only its summand `_combine`: `prover.SumcheckRound` multiplies a product,
`logup_gkr.prover.LogupSumcheckRound` evaluates the LogUp combine. `prove` is the
homogeneous scan driver, generic over that summand — it reads a round's `degree` +
`_combine` (the `SumcheckSummand` seam in `prove.py`) and owns the split / fold /
scan, so the product and LogUp per-variable loops share one driver; `verify` is the
dual, generic over any verifier `Round`. The verifier is a single
`verifier.SumcheckRound` that pairs with all of them: it sees only the round
polynomials, so the summand is purely the prover's concern. Its observe→challenge
order mirrors the prover's exactly — that shared ordering is the only thing keeping
the two Fiat-Shamir transcripts from diverging.

**Prover and verifier in symmetric namespaces.** Each side is a `Round` in its
own module — `prover.SumcheckRound` / `verifier.SumcheckRound` — so a caller picks
the side by namespace, and the shared summand (e.g. `logup_combine`) plus the
mirrored transcript order keep them from drifting rather than one fused
description.

**Composing rounds (nn.Sequential).** A composite protocol is itself a `Round`:
`ProveChain` / `VerifyChain` (in `zorch/round.py`) sequence sub-rounds, threading
the carry + transcript, so chains nest. `prove` / `verify` are the *homogeneous*
case — one round repeated per variable, the sumcheck inner loop; the chains are
the *heterogeneous* case — distinct rounds in sequence, e.g. one GKR layer per
link.

**The verifier reduces; the PCS closes.** `verify` stops at the point-claim
`(point, final_claim)`. The final `final_claim == f(point)` check needs the
polynomial's value at `point`, which a PCS opening supplies — keeping it outside
the block is what makes sumcheck here proving-scheme- and PCS-agnostic (a
non-negotiable). A wrong claimed sum or a tampered message comes back as a
returned `ok`, not a raise, so a round body carries no value-dependent host
control flow and stays trace-friendly for the device transcript
([#3](https://github.com/fractalyze/zorch/issues/3)) and single-kernel
directions. A *malformed* proof (wrong rank, or a round message not `degree+1`
wide) is a structural error, not a soundness one — that raises.

## Fusion is by construction

`_round_poly` / `_fold` stay to element-wise field ops plus the one inherent `Σ`
— no gratuitous `reduce` / `gather` — so a round body is foldable without a marker
or a `@jit` decorator. Measured on ZKX GPU, `_round_poly` lowers to one reduction
(`kInput`) kernel and `_fold` to one element-wise (`kLoop`) kernel; a full round
is **two** kernels, since its message and folded state are disjoint outputs.
Collapsing a round into one replayed unit is the marker + generic zkx emitter
step (Phase 3, cross-repo).

## Gotchas

- **The sumcheck domain is built in the base field, not via `jnp.arange`.** Iota
  is unimplemented for extension dtypes, so index/domain arrays are built in the
  base field and `.astype(EF)` (or `jnp.stack`) — one of the
  [ZKX field-dtype gotchas](poly.md#zkx-field-dtype-gotchas). Extension arithmetic
  and base→extension embedding otherwise work, so a base-field MLE folds correctly
  against extension-field challenges.
- **Prover and verifier must start the transcript in the same state.** That
  shared initial state is the block's entire local soundness assumption.
  `StubTranscript` replays preset challenges; the real duplex-sponge transcript
  ([#3](https://github.com/fractalyze/zorch/issues/3)) implements the same
  `Transcript` Protocol and drops in without touching any round.
