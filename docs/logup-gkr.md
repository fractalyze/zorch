# logup-gkr — fractional-sum GKR

The *why* behind `zorch/logup_gkr/`. The *what* lives in the code and its tests,
the executable usage that runs on every commit. Full design and open decisions:
epic issue [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

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
the GKR prover is `ProveChain([GkrLayerRound(l) for l in reversed(layers[:-1])])`
— the **heterogeneous-chain** case of the `Round` composition contract (distinct
rounds in sequence; see [`sumcheck.md`](sumcheck.md)), one bound variable per
layer, interaction floor outward to the input.

**Prover and verifier share the summand and the head.** `logup_combine` and
`bind_output` are module-level and reused by both sides: a drift between the
prover's round body and the verifier's oracle would break soundness *silently*,
so there is exactly one expression for each. The verifier is the dual
`VerifyChain`, threading the same `(num_eval, den_eval, eval_point)` carry and
ANDing each layer's check. Points are MSB-first (matching [`poly.eq`](poly.md))
with the pyramid's child selector appended as the low bit, so the prover's and
verifier's points align with no flip. The verifier evaluates `eq` with the
`O(n)` `eval_eq`, never a `2ⁿ` vector, so it stays succinct.

**Stops at the point-claim; the PCS closes; dense only.** `verify` reduces to a
`(point, claim)`; the final `claim == leaf_mle(point)` check needs a PCS opening
of the input trace and is the consumer's — keeping this block PCS-agnostic. The
circuit is dense/uniform (every interaction shares one row count, so a layer is a
flat power of two with no padding); jagged real-chip layouts, interaction
fingerprinting, and trace openings are an SP1-trace concern in the consumer
(whir-zorch), not here.

## Fusion by construction

Inverted from a single fused `Round`: the pyramid is folded and proved *eagerly*
(`build_pyramid` is a plain Python loop), **not** one fused program — the full
pyramid does not fit one `@jit` at scale, and each transition's output feeds the
next layer's per-variable sumcheck independently. Fusion-by-construction lives one
level down: each `LogupSumcheckRound` body reuses sumcheck's `split`/`fold` plus
an element-wise combine and the one inherent `Σ`, so a round body folds to one
kernel without a marker, exactly as in [`sumcheck.md`](sumcheck.md). The
host-driven pyramid over fused round bodies is the deliberate shape here, not a
missing optimization.

## Gotcha

The round polynomial evaluates its whole `u`-domain `[0, …, degree]` in one
batched reduction; `us` is built with `jnp.stack`, not `jnp.arange` (iota is
unsupported for extension dtypes — see
[`poly.md`](poly.md#zkx-field-dtype-gotchas)).
