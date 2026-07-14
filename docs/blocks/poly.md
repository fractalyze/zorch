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
general integer arrays:

- **No iota over an extension dtype.** `jnp.arange(dtype=<extension field>)`
  raises, so an integer domain / index ramp is built per element (`jnp.array` in
  a static loop, `jnp.stack`) or in the base field and `.astype(EF)`. Hits the
  sumcheck domain, the LogUp round-poly `us`, and `eval_univariate`'s nodes.
- **Iterating a field `Array` dispatches `lax.sign`.** `for coord in x` over a
  field array trips an unimplemented `sign`; index explicitly (`x[j]`) instead —
  see `expand_eq_to_hypercube`.
- **No `lax.shift`, no type-promoted power.** `field >> int` and
  `jnp.power(field, int_array)` are unsupported; bit-decompose host-side in
  `numpy` (the jagged `msb_first_bits`) and build a coset ramp with `jnp.cumprod`,
  not arange/power (see [`coding.md`](coding.md)).
- **Extension-field `reduce_sum` is unsupported.** `jnp.sum` over an extension
  array can abort with an MLIR assertion; where the trip count is static, unroll
  with `functools.reduce` (the jagged `eval_jagged_mle`) and revert to a single
  reduction once the backend supports it.
- **`jnp.tile` aborts on a field dtype.** Tiling trips an MLIR bit-width
  assertion; broadcast with a `vmap`'d matmul or `jnp.stack` / `reshape` instead
  (the jagged transition broadcast).
