# jagged — Jagged PCS (Little Polynomial eval + opening)

The *why* behind `zorch/pcs/jagged/`. The *what* lives in the code and its
tests under `zorch/pcs/jagged/testing/`. Full design and open decisions: epic
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

**Static config carries the tiers, names no field.** `JaggedStaticConfig` is
frozen, so it hashes as a `jit` static arg, and fixes `l_max` (compiled-max
column count), the column / row / layer bit-widths, and the field `dtype`. The
real-chip layout *producing* the column heights is the consumer's trace concern;
here it is the agnostic eval in AOT-clean form — a static `l_max` column axis and
a `lax.fori_loop` layer loop instead of host-driven shapes.

## Opening

The opening proves the evaluation of the original sparse, variable-height
polynomial via the committed dense poly `D` (the [BaseFold matrix](pcs.md#basefold-transparent-multilinear)
commitment) and `J̃`. It is **two sumchecks plus a stacked dense open**, all
zorch-native (natural-order fold, [transcript](transcript.md), no domain separators or
PoW — mathematical fidelity, not byte-equality with any external prover):

- **Outer Hadamard sumcheck** `Σ_i D(i)·J̃(i)` (degree 2) reduces to a point
  `z_final`, leaving `dense_eval·jagged_eval == outer_final_eval` to check. Its
  second factor is `partial_eval` — `J̃(z_row, z_col, ·)` materialized over the
  dense hypercube.
- **Inner jagged-assist sumcheck** reproves `jagged_eval = J̃(z_final)`. That is a
  `Σ_c eq(z_col, c)·h(…)` sum over all `L` columns; rather than make the verifier
  pay `O(L)`, the inner sumcheck reduces it — over the merged-prefix-bit hypercube,
  exploiting that the column eq-weight is sparse (`L` deltas) — to a **single**
  branching-program leaf `h` the verifier evaluates. The succinctness mechanism,
  not a soundness afterthought.
- **Stacked dense open.** `z_final` splits into `(z_K, stack_point)`; BaseFold
  opens at `stack_point` → the `K` per-column evals, eq-combined by `z_K` into
  `D(z_final)`. (`D` and `J̃` share the dense index width, so `log_m == n_d`.)

The verifier replays both sumchecks, recomputes the `h` leaf, checks the product,
verifies the BaseFold opening, and re-derives the structure bind — tying the proof
to the commitment.

### Verifier input contract

The column heights arrive with the statement, and what the verifier checks about
them today is **shape, not magnitude**. `build_jagged_layout` rejects only more
real columns than the capacity `l_max`; `verify_jagged_eval_msg` and
`stacked_basefold_verify` then reject a proof whose static shape cannot represent
the configured protocol — one degree-`_DEGREE` polynomial per merged prefix bit,
agreement between component commitments / round widths / batch evals / component
openings, at least one stacking variable and one query, one committed fold layer
per stacking variable, and a folded point wide enough to carry them.

Booleanity and monotonicity of the prefix-sum bits need **no** runtime check:
prover and verifier both *rebuild* the bits from the supplied counts, so they
hold by construction. (A verifier that instead consumed prover-supplied prefix
sums would have to check both — a design property of this layout-recomputing
shape, worth preserving.)

**The magnitude bounds are a gap, not a guarantee.** Nothing rejects a negative
or oversized height, a zero-area layout, or a log-area tier past the point where
`2^tier` still embeds canonically below every supported field prime and decodes
within int32. A caller handing this verifier heights from an untrusted statement
is relying on its own validation, because this block performs none. Closing it
is a trust-model decision — which side owns statement validation — rather than a
missing line.

**Composite fusion is deferred.** The opening is a procedural composition of `@jit`
device zones; folding the whole protocol into one fused kernel (the
[fusion north star](../README.md#fusion-north-star)) needs the Fiat-Shamir-internal
sumcheck marker and is gated on `frx.lax.composite` accepting field dtypes —
tracked on the epic, not this slice.

## Fusion by construction

`eval_jagged_mle` is a single `frx.jit` whose output shape is a function of input
*shapes* only — no value dependence — with a `lax.fori_loop` over the bit layers,
so it carries no host control flow. It is verifier-once work, not a per-round hot
path, so it is `@jit`'d directly rather than written for a fused-region marker
(see the hub [fusion north star](../README.md#fusion-north-star)).

The per-column indicator fold `bp_eval_core` is the opposite case: the inner
sumcheck `vmap`s it over every column each round, so its DP fold is the prover's
launch-bound hot leaf — thousands of microscopic per-layer transition matmuls. It
is wrapped in the name-routed `zorch.jagged_bp` composite so a vendor emitter fuses
the whole `fori_loop` DP (a 4-vector through `num_vars` soft `[4, 4]` transitions)
into one register-resident kernel, the way `hash_frx.poseidon2` fuses a permutation.
The marker carries a byte-identical decomposition, so an emitter that does not
recognize it inlines the fold unchanged — the fusion is a lowering property, never
a behavior change.

## Gotchas

The eval routes around three [field-dtype limits](poly.md#field-dtype-gotchas)
— `jnp.tile`, extension `reduce_sum`, and `lax.shift` — with the standard
workarounds (a `vmap`'d matmul, host-side `numpy` bit decomposition). One has a
jagged-specific cost worth flagging: the `reduce_sum` workaround unrolls the
column sum at trace time, which grows the trace **linearly in `l_max`** — fine for
the small-`l_max` verifier case here, and the reason `l_max` stays a tight
compiled bound.
