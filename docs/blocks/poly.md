# poly — polynomial primitives

The *why* behind `zorch/poly/`. The *what* lives in the code and its tests under
`zorch/poly/testing/`, the executable usage that runs on every commit. Full design and open decisions: epic issue
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Why the shape

The reusable polynomial pieces a PCS/IOP stands on, naming no field — equality
polynomial, multilinear evaluation and fold, univariate interpolation. Each is a
specific closed form chosen for a cost or soundness reason, not a generic
`numpy`-style util.

**`eq` — two forms, prover and verifier.** `expand_eq_to_hypercube(x, s)`
materializes `s·eq(w, x)` over all `w ∈ {0,1}ⁿ` (the prover's `2ⁿ` weight
vector); `eval_eq(w, x)` is the `O(len)` / `O(1)` closed form
`Π (wx + (1−w)(1−x))` for a single pair, so a verifier evaluating `eq` at a bound
point stays succinct without ever materializing a `2ⁿ` vector. `eq` is indexed
**MSB-first** (`w[0]` is the high bit); the sumcheck binds the high variable
first, so a point produced in one place is consumed in another with no
reordering. `eval_eq` is symmetric and order-agnostic (a product over
coordinates), so it pairs with either side.

**`multilinear` — eval via `eq`, and the two distinct folds.** `eval_mle` is the
`eq` inner product on a chosen axis (leading/trailing batch axes ride through).
`mle_fold` is the **additive** Basefold/FRI combine `e0 + β·e1` — deliberately
*not* the multilinear partial-evaluation bind `(1−β)·e0 + β·e1` that
`sumcheck.prover.SumcheckRound` uses. Conflating the two is the easy, silent bug,
so they are separate functions with the distinction stated at each. Basefold is
`mle_fold`'s first consumer. (The codeword-domain fold is `coding`'s
[`fri_fold`](coding.md#the-fold-rides-the-encoders-domain).)

**`univariate` — direct Lagrange, not barycentric.** `eval_univariate`
interpolates over the integer domain `[0, …, len−1]` in direct form: barycentric
divides by `(x − node)`, which an `x` landing on a node would zero — and round
polynomials are routinely evaluated at the small integer nodes themselves.

## Fusion by construction

These are leaf numeric helpers — element-wise field ops plus the one inherent `Σ`
in `eval_mle` — so they fold into a caller's kernel without a marker or a `@jit`
of their own (the `@jit` guidance in [`conventions.md`](../reference/conventions.md#jit)
applies). They carry no host control flow, so they drop unchanged into the
[fusion north star](../README.md#fusion-north-star) path.

## Field-dtype gotchas

The canonical list the other blocks point to. The finite-field dtypes are not
general integer arrays. Two limits hold on the current toolchain (measured on
the frx `0.10.1.dev20260803` wheels, CPU and CUDA tiers):

- **No `lax.shift`.** `field >> int` fails to lower
  (`'stablehlo.shift_right_arithmetic' op operand #0 must be … integer`);
  bit-decompose host-side in `numpy` (the jagged `msb_first_bits`).
- **No power by a traced exponent.** `jnp.power(field, int_array)` raises
  (`field/EF base requires a Python-int exponent; use lax.integer_pow`). A
  static Python-int exponent is fine; a coset / geometric ramp is built with
  `jnp.cumprod`, not power-by-index (see [`coding.md`](coding.md)).

Three earlier limits — iota over an extension dtype, extension-field
`reduce_sum`, and `jnp.tile` (plus iterating a field array, which dispatched an
unimplemented `lax.sign`) — no longer reproduce on the current wheels. Their
workarounds are still in the tree and explain those shapes: the per-element
domain ramps in the sumcheck domain, the LogUp round-poly `us`, and
`eval_univariate`'s nodes; the `functools.reduce` unroll in the jagged
`eval_jagged_mle`; the `vmap`'d-matmul broadcast in the jagged transition; the
explicit indexing in `expand_eq_to_hypercube`. Simplify a site to the direct
op when touching it — with the compile-count/runtime checks that motivated the
workaround — rather than in a blanket pass.
