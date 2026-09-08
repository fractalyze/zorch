# logup-gkr — fractional-sum GKR

The *why* behind `zorch/logup_gkr/`. The *what* lives in the code and its tests,
the executable usage that runs on every commit. Full design and open decisions:
epic issue [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Roles and claims

`LogUpGkrProver` / `LogUpGkrVerifier` (`zorch/logup_gkr/stage.py`) are the stage
roles: they reduce a `LogUpOutputClaim` — the public fractional-sum output and
layer count — to an `InputLayerClaim`, the evaluation claim a consumer's PCS
opening closes. The jagged layout reuses those claim types: a consumer proving
over it drives `jagged_prover`'s layer rounds from its own stage seam, so only
the witness and the layer proofs differ.

`GkrLayerRound` and the per-variable sumcheck rounds are the recurrence inside
those roles, driven layer by layer. They are internal: the pyramid is a round
recurrence, and the reduction across it is the stage.

## Why the shape

**A circuit, a per-variable round, a layer round — chained.** `circuit.py` is
pure data: a `GkrLayer` is four equal-length MLEs `(n0, n1, d0, d1)`,
`layer_transition` folds each fraction pair
(`n0/d0 + n1/d1 = (n0·d1 + n1·d0)/(d0·d1)`) and halves every MLE, and
`build_pyramid` folds from the input layer down to the interaction floor.
`prover.LogupSumcheckRound` is one per-variable sumcheck round whose summand is
`logup_combine` over five factors `[eq, n0, d1, n1, d0]` — the direct sibling of
the product `sumcheck.prover.SumcheckRound`. `GkrLayerRound` proves one layer (its
per-variable sumcheck via `fold_rounds`, then the child-selector reduction), and
the GKR prover is `prove_rounds([GkrLayerRound(l) for l in reversed(layers[:-1])])`
— a round recurrence whose carry and message contract stays fixed while layer
shapes vary, one bound variable per layer, interaction floor outward to the
input. See [`sumcheck.md`](sumcheck.md) for how round recurrence differs from
stage composition.

**Prover and verifier share the summand and the head.** `logup_combine` and
`bind_output` are module-level and reused by both sides: a drift between the
prover's round body and the verifier's oracle would break soundness *silently*,
so there is exactly one expression for each. The verifier is the dual
`verify_rounds`, threading the same `(num_eval, den_eval, eval_point)` carry and
ANDing each layer's check. Points are MSB-first (matching [`poly.eq`](poly.md))
with the pyramid's child selector appended as the low bit, so the prover's and
verifier's points align with no flip. The verifier evaluates `eq` with the
`O(n)` `eval_eq`, never a `2ⁿ` vector, so it stays succinct.

**Stops at the point-claim; the PCS closes.** `verify` reduces to a
`(point, claim)`; the final `claim == leaf_mle(point)` check needs a PCS opening
of the input trace and is the consumer's — keeping this block PCS-agnostic.
Interaction fingerprinting, padding schedules, and trace openings likewise stay
in the consumer: the circuit and provers carry only the two layouts (dense and
jagged), never a proving scheme.

## The jagged pair

A `JaggedGkrLayer` materializes only `sum(row_counts)` of its virtual
`2^(niv+nrv)` positions; everything else is the fold-neutral fraction
(n=0, d=1). Two consequences shape `jagged_prover` / `jagged_verifier`:

- **Virtual mass is closed-form, not memory.** A neutral position's summand
  collapses to its eq weight, and a full hypercube's eq weights sum to the
  bound prefix's eq product — so the per-round correction is `pad_adj − eq_sum`
  and the prover never densifies. The verifier is layout-blind: the corrections
  make the round polynomials exactly those of the virtual dense hypercube, so
  it replays them knowing nothing about row counts.
- **Coefficient form, LSB-first.** The summand carries the current variable's
  eq factor, whose root is known to both sides, so a degree-3 round travels as
  coefficients interpolated through {0, 1, 1/2, b} — two materialized
  evaluations plus the claim (Gruen, https://eprint.iacr.org/2024/108) where
  the dense round's value form needs the whole natural domain. Binding is
  LSB-first because the layout is interaction-major: the row LSB is the
  in-segment pair dimension, so the stride-2 fold respects segment boundaries.
  Reversing the challenges lands back on the dense chain's MSB-first carry, so
  dense and jagged chains thread the same `(num_eval, den_eval, eval_point)`.

The two rounds are two wire forms of the same LogUp summand — `logup_combine`
stays the one shared definition — and the jagged verifier replays through the
same agnostic `zorch.verify` scan via `sumcheck.verifier.CoeffsSumcheckRound`.

A jagged layer is a *capacity* layer: the four MLEs sit in a static
`width >= sum(row_counts)` buffer with a dead-zero tail and the counts ride as
one traced i32 vector, so a consumer whose per-segment counts vary per input —
total staying under one shared bound — reuses every compile across inputs, and
the zero-slack layer is just the tight case. Transitions derive their gathers
in-trace, byte-identical to the host-built fold; rounds are caps-mandatory with
the schedule as traced operands, so any same-caps layer shares the zone
executable. What a host guard cannot read it cannot check: truncation-safety and
the virtual-row-space fit are the consumer's capacity-class obligations,
discharged against class bounds dominating every admitted input.

The stage owns what a chain would: it builds the pyramid, hands each layer to its
round and drops it, scoping one `LayerBuffers` to the prove so the cap-wide
planes die with it rather than pinning a card.

## Fusion by construction

Inverted from a single fused `Round`: the pyramid is folded and proved *eagerly*
(`build_pyramid` is a plain Python loop), **not** one fused program — the full
pyramid does not fit one `@jit` at scale, and each transition's output feeds the
next layer's per-variable sumcheck independently. `prove_rounds` consumes its
rounds lazily for the same scale reason: a generator-built chain releases each
layer once its round is proved, so the pyramid's planes never need to be live
together. Fusion-by-construction lives one
level down: each `LogupSumcheckRound` body reuses sumcheck's `split`/`fold` plus
an element-wise combine and the one inherent `Σ`, so a round body folds to one
kernel without a marker, exactly as in [`sumcheck.md`](sumcheck.md). The
host-driven pyramid over fused round bodies is the deliberate shape here, not a
missing optimization.

## Gotcha

The round polynomial evaluates its whole `u`-domain `[0, …, degree]` in one
batched reduction; `us` is built with `jnp.stack`, not `jnp.arange` (iota is
unsupported for extension dtypes — see
[`poly.md`](poly.md#field-dtype-gotchas)).
