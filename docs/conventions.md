# Coding conventions

> Code, symbols, and file paths are English.

## `@jit`

`@jit` the leaf numeric kernels; compose them in plain Python.

**Apply `@jit`** to a pure field/array function whose output shape is static
(a function of input *shapes*, not values). Static-trip-count Python loops
inside are fine — `jit` unrolls them.

**Do not `@jit`** when the function:
- returns a Python value from structure — e.g. `zorch.utils.bits.log2_strict_usize`
  returns an `int` from a length; `jit` would trace it away.
- composes other work in a static loop with host-side steps between — e.g.
  `zorch.prove` loops the round, and `SumcheckRound.__call__` wraps the
  round-poly/fold arithmetic around the host-side transcript `commit` /
  `challenge`. Decorating these would inline everything into one trace and pull
  the transcript ops in with it.

`SumcheckRound.round_poly` / `fold` are pure numeric and *could* be `@jit`'d,
but are deliberately left undecorated: they are the bodies a future marked
fused region (`stablehlo.composite`) + zkx emitter will lower to one kernel
(see `sumcheck.md`), not blanket-`@jit` candidates.
