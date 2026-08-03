# Field dtypes: writing finite-field arithmetic

zorch computes over native finite-field dtypes from `zk_dtypes`
(`goldilocks_mont`, `babybear`, the `*x3`/`*x4` extensions, …), lowered by
Fractalyze XLA. Every op reduces mod p in the canonical representation. Arrays
are FRX arrays; use `frx.numpy` (imported as `fnp` throughout zorch) the way
you would `jax.numpy`.

## The one rule

**Reach for the dtype, never hand-rolled modular arithmetic.** Hand-rolls
duplicate the dtype's reduction and silently diverge from it:

- No `pow(x, e, p)`, no `x % MODULUS`, no hardcoded prime literals
  (`2**64 - 2**32 + 1`).
- Exponentiation: `fnp.power(x, e)` on a field-typed `x` with a non-negative
  int `e`. (`lax.pow` is float-only and won't lower for field dtypes.)
- Inverse / division: `x ** -1` or `a / b` on field-typed values — the field
  inverse, not `pow(x, p-2, p)`. `fnp.power` rejects negative exponents; the
  scalar `** -1` form is the way.
- Need the prime itself? `zk_dtypes.pfinfo(F).modulus` — never a literal.
- A host-side constant handed to an `int`-typed API still computes in the
  field first, then casts: `int(F(W32) ** -1)`, not `pow(W32, p-2, p)`.

## Ops that work on ints but fail on field dtypes

Measured on the toolchain `pyzorch 0.2.0` installs, CPU and CUDA tiers:

| Broken on field dtypes | Error | Workaround |
| --- | --- | --- |
| `field >> int` (bit shift) | `ValueError: Cannot lower jaxpr with verifier errors: 'stablehlo.shift_right_arithmetic' op operand #0 must be … integer …` | Bit-decompose host-side in plain `numpy` before entering traced code. |
| `fnp.power(field, exponent_array)` | `TypeError: jnp.power: field/EF base … requires a Python-int exponent; use lax.integer_pow(x, int_exponent).` | A **Python-int** exponent (`fnp.power(x, 5)` / `lax.integer_pow`) is fine; for a geometric ramp `[1, g, g², …]` use `fnp.cumprod`, not power-by-index. |

Older toolchain pins also rejected iota (`fnp.arange` with an extension
dtype), `fnp.tile`, extension-field `fnp.sum`, and iterating a field array —
those all pass on the current wheels. If you are pinned to an older build and
hit an MLIR assertion on one of them, the historical workarounds are in
[poly.md — Field-dtype gotchas](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/poly.md#field-dtype-gotchas).

## Sampling field values in tests

Don't build random field arrays from raw ints by hand:

```python
from zorch.testkit.random_field import rand_field, rand_ext_field
x = rand_field(seed=0, shape=(1 << 10,), dtype=F)
```
