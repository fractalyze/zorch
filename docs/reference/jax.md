# Thinking in JAX

[`conventions.md`](conventions.md) gives zorch's rules. This page gives the model
they follow from, so they read as consequences rather than edicts. JAX is a
tracing compiler, not numpy on a GPU, and nearly every mistake a strong
numpy/PyTorch engineer makes here is one of four constraints surfacing late.

1. **A traced function is a pure function of its array inputs.** Python runs once
   with abstract tracers to record a graph. Side effects happen at trace time
   only; branching on an array *value* cannot be recorded. Hence validation on
   shapes in `__post_init__` (it reruns on tracers during `unflatten`), and hence
   a Python `int` pulled from a traced quantity forcing an eager island.
1. **Trace once, run many — shapes are static, values are not.** The graph is
   keyed on shapes, dtypes, and static args. A new shape is a new compile, which
   is why a halving `scan` carry rides a fixed-width buffer with a masked tail.
   When something should be fast and isn't, suspect a recompile before the
   kernel.
1. **State is data.** Threaded state is a registered frozen dataclass, not a
   closure variable or a bag of positional arrays. `data_fields` are the array
   leaves a transform traces over; `meta_fields` are static config, and they
   **compare by value** — an identity-compared meta field silently re-traces its
   whole `jit` zone per freshly built instance.
1. **The output shape picks the loop tool.** `vmap` for an independent batch,
   `lax.scan` for a homogeneous carry, Python `for` for static straight-line or
   heterogeneous work. Wrong choice either inflates the graph past the PTX cliff
   or drops a `stablehlo.while` boundary into code that wanted to fuse.

A class is justified only by configuration or threaded state, never to group
methods — and then it is a frozen registered dataclass of pure methods, with no
mutable `self`. Register what a transform threads (a `Round` as a `scan` carry),
not what is merely invoked (`ReedSolomon.encode`). The exact rules live in
[`conventions.md`](conventions.md#pytree-registration).

## Traps

- **Host↔device sync is the silent tax.** `.item()`, `float(x)`, `np.asarray(x)`,
  or printing a value inside a loop stalls the device. Pull scalars out once.
- **PRNG is explicit.** Thread a key and `split` it; reuse silently correlates
  draws.
- **Updates are functional.** `x.at[i].set(v)` returns a new array;
  `donate_argnums` is the only "in place" there is.
- **`jit` a stable callable, never a fresh `lambda` or inner `def`.** The cache is
  keyed on the callable's `id()`. Bind with `functools.partial`, or hoist.
- **Field dtypes have extra edges** — see
  [`poly.md`](../blocks/poly.md#field-dtype-gotchas).

## When `jit` is slow

It is re-tracing, graph explosion, or recompiling on a new shape — rarely the
kernel. Turn on the flags before theorizing:

```bash
JAX_LOG_COMPILES=1          # trace / lower / compile time, per function
JAX_EXPLAIN_CACHE_MISSES=1  # why JAX retraced, naming function and line
```

A re-trace every call means a per-call callable or an identity-compared meta
field. A slow first call means an unrolled Python `for` — dump
`JAX_DUMP_IR_MODES=eqn_count_pprof` and `pprof -top` names the line. A recompile
on a "different" input means a shape changed.

## Reading list

[🔪 The Sharp Bits 🔪](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
and [How to Think in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)
first — the two pages every JAX engineer is expected to have read. Then
[JIT mechanics](https://docs.jax.dev/en/latest/jit-compilation.html) so "it's
slow" becomes "it's recompiling, here's why", and
[All of Equinox](https://docs.kidger.site/equinox/all-of-equinox/) so state stops
being an arg bag. [Autodidax](https://docs.jax.dev/en/latest/autodidax.html)
builds JAX core from scratch and is the best argument for why purity and flat
pytrees are load-bearing rather than stylistic. The
[slow-tracing guide](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html)
and [JAX errors](https://docs.jax.dev/en/latest/errors.html) are the two to reach
for when stuck.
