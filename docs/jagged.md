# jagged — Jagged Little Polynomial eval

The *why* behind `zorch/commit/jagged/`. The *what* lives in the code and its
tests under `zorch/commit/jagged/testing/`. Full design and open decisions: epic
issue [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Why the shape

`jagged` is the verifier-side point-eval of the **Jagged Little Polynomial** — the
indicator MLE
`J̃(z_row, z_col, z_index) = Σ_c eq(z_col, c)·h(z_row, z_index; t_c, t_{c+1})`
that a jagged (ragged-height) commitment opens against. It is the structural
counterpart to the [Merkle commitment](commit.md): the tree commits the flattened
data, this is the polynomial the opening is checked against.

**The indicator is a branching program, not a closed form.** `H(r, c, i) = 1` iff
`t_c ≤ i < t_{c+1}` and `i − t_c = r` — a per-bit addition check (`i = r + t_c`)
AND-ed with an MSB-first range comparison. `h` evaluates it as a four-state
`(carry, comparison_so_far)` automaton folded MSB→LSB: each layer reads the four
relevant bits, expands them to a 16-way `eq`, and applies one `[4, 4]` transition
matrix. Casting the indicator as a fixed-size DP keeps the eval `O(num_vars)`
rather than materializing a hypercube.

**Empty-range padding is mandatory.** Columns past the real count pad with an
*empty* range (`t_c = t_{c+1} = t_L`), never zero bits — a zero-bit pad injects a
phantom range `[0, t_0)` that corrupts `J̃`.

**Static config carries the tiers, names no field.** `JaggedStaticConfig` (frozen,
so it hashes as a `jit` static arg) fixes `l_max` (compiled-max column count),
`n_c` / `n_r` / `n_d` (column / row / layer bit-widths), and the field `dtype`.
The real-chip layout that *produces* the column heights (per-interaction row
counts, gather-pad) is the consumer's trace concern; here it is just the agnostic
eval. Adapted from whir-zorch's `jagged/poly.py` to AOT-clean form — a static
`l_max` column axis and a `lax.fori_loop` layer loop in place of host-driven
shapes.

## Fusion by construction

`eval_jagged_mle` is a single `jax.jit` whose output shape is a function of input
*shapes* only — no value dependence — with a `lax.fori_loop` over the bit layers,
so it carries no host control flow. It is verifier-once work, not a per-round hot
path, so it is `@jit`'d directly rather than written for a fused-region marker
(see the hub [fusion north star](README.md#fusion-north-star)).

## Gotchas

The eval routes around three [ZKX field-dtype limits](poly.md#zkx-field-dtype-gotchas)
— `jnp.tile`, extension `reduce_sum`, and `lax.shift` — with the standard
workarounds (a `vmap`'d matmul, host-side `numpy` bit decomposition). One has a
jagged-specific cost worth flagging: the `reduce_sum` workaround unrolls the
column sum at trace time, which grows the trace **linearly in `l_max`** — fine for
the small-`l_max` verifier case here, and the reason `l_max` stays a tight
compiled bound.
