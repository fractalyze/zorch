# sumcheck — the folding `Round`

## Why this module exists

Sumcheck, FRI, Basefold, and a GKR layer are different protocols with one shape
in common: each runs a sequence of rounds that fold a carry 2-to-1 — sending one
message and consuming one challenge per round — until the carry collapses to a
single value. `zorch` factors that shape into two pieces: a `Round` (one fold
step) and a generic driver, `prove`, that runs a `Round` to completion. A new
protocol is a new `Round`, not a new `prove()`.

This keeps the layer proving-scheme- and zkVM-agnostic. A `Round` is a bare IOP
step: it knows how to compute its message and fold its state, and nothing about
which scheme it serves. Fiat-Shamir enters through an injected `Transcript`
Protocol — stubbed today, a duplex sponge later — threaded as an immutable carry,
so no scheme's challenge derivation is baked in. A composite protocol (LogUp-GKR
running over a product sumcheck) nests because a composite is itself a `Round`.

## The one rule: a round body is element-wise field ops plus the one Σ

A round must lower to one replayable GPU unit by construction (the fusion
non-negotiable). The rule that buys this: `_round_poly` and `_fold` use only
element-wise field arithmetic plus the single `Σ` that sumcheck inherently needs
— no `reduce`/`gather`/`scatter` beyond it. The whole round-polynomial domain
`[0..degree]` is evaluated in one batched reduction, not `degree+1` separate
sums.

Measured on ZKX GPU the bodies already lower this way: `_round_poly` to one
reduction (`kInput`) kernel, `_fold` to one element-wise (`kLoop`) kernel — two
kernels per round, since a round's message and its folded state are disjoint
outputs. Collapsing those into one replayed unit is a later phase; the
why-not-one-kernel target and the `stablehlo.composite` marker live in the fusion
north star in [`README.md`](README.md). The body is written fusion-ready so it
drops into that path unchanged, which is also why it is not `@jit`'d — see
[`conventions.md`](conventions.md).

## Design rules

- **The summand is the only thing a new protocol writes.** `SumcheckRound` owns
  the split, the fold, and the observe→challenge Fiat-Shamir step; a subclass
  supplies `_round_poly` — its summand over the round domain — and nothing else.
  `ProductSumcheckRound` sums a product of factors; `LogupGkrRound` sums the LogUp
  combine over five factors, overriding `_split` only to add its factor-count
  check. Zerocheck and the next summands are the same one-method shape.

- **State is `list[Array]`, one MLE per factor, folded in place.** A factor is
  its evaluation vector over the boolean hypercube — there is no separate
  polynomial object. The round splits each factor on the current variable and
  folds at the challenge directly, so a polynomial layer would be indirection
  with no payer.

- **The transcript is an immutable carry, not hidden state.** `observe` and
  `challenge` take and return a `Transcript`; a round never mutates one in place.
  That is what lets `StubTranscript` and the real sponge swap with no change to
  any round, and what keeps `prove` a pure fold over `(state, transcript)`.

- **The driver derives the round count from the data.** `prove` runs
  `log2(width)` rounds — one per hypercube variable — so a protocol never carries
  a round count; the width of the state says it.

## Gotcha

`jnp.arange(dtype=<extension field>)` raises — `iota` is unimplemented for
extension dtypes such as `koalabearx4`. The round-polynomial domain is therefore
built with `jnp.stack` of per-point scalars, not `arange`; build any index or
domain array in the base field and `.astype(EF)`. This is the same `iota`
constraint [`coding.md`](coding.md) hits on its coset ramp. Base→extension
embedding and extension arithmetic otherwise work, so a base-field MLE folds
correctly against extension-field challenges.

The stubbed transcript is not Fiat-Shamir-sound — `StubTranscript` replays preset
challenges for tests; the real duplex sponge is
[#3](https://github.com/fractalyze/zorch/issues/3).

## Deliberately out of scope

The single-kernel fusion marker and the device-side transcript are future phases,
not present gaps to work around. Until the transcript is device-side,
`observe`/`challenge` are host steps, so a full protocol is not yet one traced
region; both phases are tracked on the epic
([#1](https://github.com/fractalyze/zorch/issues/1)), and the round bodies are
written so neither needs a restructuring when it arrives.

## Tests

`PYTHONPATH=. python zorch/sumcheck/testing/prove_test.py` (and the per-module
`*_test.py`). The oracle is the sumcheck consistency identity itself, checked
independently of the prover: the claimed sum equals `s₀(0) + s₀(1)`, each round's
`sᵢ(0) + sᵢ(1)` equals the previous round's univariate at the challenge, and the
final fold equals the factors' MLEs evaluated at the challenge point — recomputed
by a standalone `_eval_mle`, so a bug in `_round_poly`/`_fold` cannot satisfy the
identity by sharing the prover's own code path. A separate case drives a
base-field MLE with `koalabearx4` challenges; `round_test.py` pins `_round_poly`
to one reduction kernel (`assert_fusion_ready`) and the bare `SumcheckRound`
summand as abstract.
