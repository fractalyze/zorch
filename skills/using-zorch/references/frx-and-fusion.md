# Writing FRX code that fuses

zorch's performance model assumes consumer code keeps its compute inside a few
fused device kernels. FRX is JAX with the same API surface under the `frx`
name — `@frx.jit`, `frx.vmap`, `frx.lax`, `frx.numpy` (imported as `fnp`) —
so all JAX discipline applies. This page is the condensed authoring rules; the
full mental models are in
[jax.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/reference/jax.md)
and the exact conventions in
[conventions.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/reference/conventions.md).

## The four constraints everything follows from

1. **A traced function is pure.** Python runs once with tracers; side effects
   and value-branching don't record. Validate shapes, not values.
2. **Trace once, run many.** The graph is keyed on shapes/dtypes/statics — a
   new shape is a new compile. When something is slow, suspect a re-trace or
   recompile before the kernel.
3. **State is data.** Threaded state is a registered frozen dataclass. Static
   (`meta`) fields must compare **by value** — an identity-compared meta field
   silently re-traces per instance.
4. **The per-iteration output shape picks the loop tool** (table below).

## `@jit` discipline — one boundary per round

`@jit` the leaf numeric kernels; compose them in plain Python. A round body
gets **exactly one** `@jit` boundary — zero decomposes the would-be-fused
region into eager per-op dispatches; nesting lowers to a call that blocks
single-kernel rewriting. Two valid shapes, forced by host ops, not style:

- **Single-zone**: one `@jit` over the whole prove/open body; the driver loop
  and round bodies stay plain Python inside it. Works only when the loop is
  host-op-free — zorch's `DuplexTranscript` `observe`/`sample` are device ops
  and trace through.
- **Per-island**: an eager driver loop, each maximal host-op-free span its own
  `@jit`. Required when a host op sits inside the loop (a PoW grind reading
  its witness, an `int(...)` query index). Costs a recompile per island
  shape — hoist shape-stable heavy composites (a permutation, an NTT) into
  their own island.

Never `@jit` a heterogeneous round driver (shapes vary per round — it would
unroll the whole composition into one giant trace), a function returning a
Python value from structure, or a fresh `lambda`/inner `def` (the jit cache is
keyed on callable identity — bind with `functools.partial` or hoist).

## Loop tool by output shape

| Situation | Tool | Why |
| --- | --- | --- |
| Independent items, no carry (N queries, N row hashes) | `frx.vmap` | batches into one kernel |
| Compile-time count of pure field ops (Horner, a permutation's rounds) | Python `for`, straight-line | traces to element-wise IR that fuses; `lax.scan` would insert a `while` boundary that breaks fusion |
| Homogeneous carry, round-invariant shapes, many rounds | `lax.scan` | unrolling inflates the graph past the PTX cliff; a shrinking carry rides a fixed-width buffer with a masked tail |
| Per-round shapes change (fold phases, GKR layers) | Python `for` (eager driver) | host-orchestrated separate dispatches — the round *bodies* are the fusion target, not the driver |

A `lax.scan` reached from eager code needs a **stable body callable** — one
built per call recompiles an identical graph every time, silently.

**Tiebreaker for a shrinking fold** (rows 3 and 4 both plausibly apply):
default to the eager driver (row 4) and accept one recompile per shape — a
halving table costs log₂(n) compiles once, then caches. Reach for the
masked fixed-width-buffer scan only when the round count drives compile time
or dispatch latency past the cost of the masking.

## Fusion-ready round bodies

Inside a round body: element-wise field ops plus the one inherent `Σ`. No
gratuitous `reduce`/`gather`, and **no host round-trips** — `.item()`,
`float(x)`, `np.asarray(x)`, printing a traced value all stall the device and
split the region. Keep the transcript device-side (pass it through; never pull
a challenge to Python mid-round). What "one replayable device unit" means and
what the bodies measure out to today:
[fusion north star](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/README.md#fusion-north-star).

## Verifying it actually fused

Don't assume — measure, then pin:

```bash
JAX_LOG_COMPILES=1           # per-function trace/lower/compile time
JAX_EXPLAIN_CACHE_MISSES=1   # names the function and line that re-traced
```

The compile log includes toolchain-internal compiles (`jit(convert_element_type)`,
`jit(dynamic_slice)`, …) — filter to your own function names. A *flood* of
`jit(<op-name>)` lines is itself a smell: your compute is running eagerly,
one dispatch per op, with no `@jit` boundary around it.

- Re-trace on every call → a per-call callable or an identity-compared meta
  field.
- Recompile on a "different" input → a shape changed.
- Slow first call → an unrolled Python `for` that wanted a `scan`
  (`JAX_DUMP_IR_MODES=eqn_count_pprof` + `pprof -top` names the line).
- For kernel-level ground truth, dump HLO (`XLA_FLAGS=--xla_dump_to=<dir>`).
  Per module the dump has many files; read
  `*after_optimizations.txt` for the fused module and `*thunk_sequence.txt`
  for the cleanest answer to "how many kernel launches" — an element-wise
  round body should show one `kLoop` fusion kernel.

zorch's own tests pin compile count, runtime, and peak memory per stage so
regressions fail loudly; do the same for your prover's hot path.
