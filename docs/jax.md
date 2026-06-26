# Thinking in JAX

[`conventions.md`](conventions.md) gives the **rules** zorch follows — when to
`@jit`, `scan` vs `vmap` vs `for`, what to register as a pytree. This page gives
the **mental model those rules come from**, and the canonical places the JAX
authors explain it. Read this first if JAX is new to you; then `conventions.md`
reads as consequences instead of edicts.

The thesis: JAX is not "numpy on a GPU." It is a *tracing compiler* with a small
number of load-bearing constraints, and almost every mistake a strong
numpy/PyTorch engineer makes here is one of those constraints surfacing late
(as a recompile, a silent re-trace, a host-sync stall, or a `Tracer
BoolConversionError`) instead of early. Learn the four ideas below and the rest
is detail.

## Four ideas everything else follows from

### 1. A traced function is a *pure function of its array inputs*

`jit`/`scan`/`vmap` run your Python **once** with abstract tracers to record an
IR graph, then execute that graph. So Python-visible side effects happen at
trace time only, and control flow that branches on an array *value* can't be
recorded — the tracer has no value yet. This is why zorch validates on shapes in
`__post_init__` (it reruns on tracers during `unflatten`) and never branches on a
field element there; why a `.item()` / `int(x)` / `np.asarray(x)` on a traced
value either errors or forces a host round-trip; why a Python-`int` pulled from a
traced quantity (`sample_bits → int(...)`, a PoW `grind`) forces an **eager**
island ([`conventions.md` → one `@jit` boundary per round](conventions.md#jit)).

> Read: **Sharp Bits** and **Thinking in JAX** (below). They are the two pages
> every JAX engineer is expected to have read.

### 2. Trace once, run many — *shapes are static, values are not*

The graph is keyed on input **shapes and dtypes** (and any `static_argnums`),
not values. A new shape ⇒ a new trace ⇒ a fresh XLA compile. This is the single
biggest perf footgun: a helper whose `static_argnames` tuple varies per call
recompiles *every call* — the cost looks like "JAX is slow" but is one-compile-
per-shape. Stable shapes are a design goal, not an accident: it's why a halving
`scan` carry rides a fixed-width buffer with a masked tail rather than an actually
shrinking array, and why TOTAL_MAX padding exists. When something "should be fast
but isn't," suspect a recompile first — `jax.jit(...).lower(...).compile()` and
log shapes before blaming the kernel.

> Read: **JIT mechanics** and the **FAQ** "why is my function recompiling?".

### 3. State is *data* — a flat, registered pytree, never hidden or an arg bag

JAX transforms move pytrees (nested tuples/lists/dicts/registered dataclasses of
arrays) across their boundaries. Two consequences a newcomer underweights:

- **Threaded state is a registered dataclass**, not a closure variable and not a
  loose bag of positional arrays. A round/carry with more than ~2 arrays wants a
  `@register_dataclass` — `data_fields` for the array leaves, `meta_fields` for
  static config. A 12-or-17-arg numeric function is the un-generalized form of a
  one-arg-pytree function; it can't `vmap`/`scan` cleanly and no one can reorder
  it safely.
- **Meta fields compare by value.** Identity `__eq__` on a meta field silently
  re-traces the whole `jit` zone on every freshly built instance (minutes, on a
  replay whose kernels run in ms) — the nastiest "why is it slow" in the repo.

> Read: **Pytrees** (registration), and **Equinox** for the fullest published
> treatment of "model/state *is* a pytree, code is pure functions."

### 4. Repeat work with the tool the *output shape* picks

`vmap` (independent batch, no carry) / `lax.scan` (homogeneous carry, one traced
region) / Python `for` (static straight-line, or heterogeneous host-orchestrated)
are not interchangeable styles — the per-iteration output shape forces the choice,
and getting it wrong either inflates the graph past the PTX cliff (unrolling a big
loop) or drops a `stablehlo.while` boundary into code that wanted to fuse
(scanning a tiny one). zorch's full decision table is in
[`conventions.md` → Loops](conventions.md#loops-for-vs-laxscan-vs-vmap); the JAX
**Control flow** page is the upstream primer.

## Designing a class in JAX

The first question is whether you need a class at all. JAX is functions over
pytrees; reach for a class only when there's **configuration or threaded state to
carry**, not to group methods. A stateless transform is a function. A *configured
operator* (a code, a hash, a sumcheck round) is a class — the same call where you
fix its parameters once.

When a class is justified, it's not an OO object with mutable fields — it's a
**frozen, registered-dataclass pytree of pure methods**. Grade any class you write
against these six; `sumcheck.SumcheckRound` is the worked example that passes all
of them:

1. **Earns its class-hood.** It carries config (`degree`) or state, or it's a
   polymorphic seam consumers dispatch on. If it's a bare stateless function
   dressed as a class, make it a function. (`coding.md`'s "[a code is an object,
   not a call](coding.md)" is the same call made for `ReedSolomon`.)
2. **Frozen + registered if it crosses a transform.** Anything passed to / returned
   from / carried through `jit`/`vmap`/`scan` is
   `@partial(jax.tree_util.register_dataclass, data_fields=[...], meta_fields=[...])`
   + `@dataclass(frozen=True)`. Frozen because a pytree leaf is immutable and a
   meta field must be hashable.
3. **No hidden mutable state.** No `self.x = ...` across calls, no counter, no
   cache-on-`self`. State (the transcript, the MLE carry) threads *in and out* of
   the method, functionally. If a method mutates `self`, the transform can't see
   it and it's a bug, not an optimization.
4. **The meta/data split is the real design decision.** `data_fields` = the array
   leaves a transform should trace over (a challenge `lam`); `meta_fields` = static
   config baked into the trace (`degree`, a permutation). Get this wrong and you
   either trace a constant (a static value as data) or recompile on every value (an
   array as meta). Meta fields **compare by value** — an identity-compared meta
   silently re-traces the whole `jit` zone per freshly built instance.
5. **`__post_init__` validates only static/shape, never a value.** It reruns on
   *tracers* during `unflatten`, so `if some_array > 0: raise` there breaks under
   `jit`/`vmap`. Check `degree >= 1`, shapes, dtypes — not field contents.
6. **Operator vs carry — register only what's threaded.** A class built host-side
   and only *invoked* (`ReedSolomon.encode`, `Poseidon2`) captures its arrays as
   closure constants and is **not** a pytree. A class *threaded through* a transform
   (a `Round` as a `scan` carry) **must** be registered. Registering an operator
   buys nothing (noise); failing to register a carry is an error. zorch draws this
   line explicitly — match it.

The two published schools of "class = pytree module," worth reading for the full
argument:

- **Equinox** ([All of Equinox](https://docs.kidger.site/equinox/all-of-equinox/),
  [`Module`](https://docs.kidger.site/equinox/api/module/module/)) — the minimal,
  closest-to-this-codebase take: a module is literally a registered dataclass; this
  is the rubric above, formalized.
- **Flax** ([flax.readthedocs.io](https://flax.readthedocs.io/)) — the heavier,
  more OO-feeling framework (NNX today, Linen classically); still a pytree
  underneath, but with managed parameter/state collections. Read it to see the
  trade-off zorch deliberately *doesn't* take (no framework, just dataclasses).
- **JAX itself** —
  [Stateful computations](https://docs.jax.dev/en/latest/stateful-computations.html)
  is the from-scratch version: how to carry "state" without a mutable object at all.

The zorch-specific rules — exactly which classes are registered, the
`*PytreeTest` every registered class carries, the `meta_fields` value-equality
gotcha — live in [`conventions.md` → Pytree registration](conventions.md#pytree-registration).

## Traps the vets repeat (short list)

- **Host↔device sync is the silent tax.** `.item()`, `float(x)`, `np.asarray(x)`,
  `.block_until_ready()`, or printing a value *inside* a loop serializes on the
  host and stalls the device. Keep data on-device; pull scalars out once, at the
  end. (The `_round_metadata` per-layer `jnp.asarray` transfers flagged in the
  logup_gkr review are this trap.)
- **PRNG is explicit and functional.** No global seed. Thread a `key`, `split` it
  to fork, never reuse one — reuse silently correlates draws. The design and the
  why are a whole JEP.
- **Updates are functional.** `x.at[i].set(v)` returns a new array; there is no
  in-place write. `donate_argnums` is how you tell XLA it may reuse the input
  buffer when the shapes match — the only "in place" that exists.
- **Don't pull a Python value out of a tracer.** `if x > 0` / `int(x)` /
  `len(traced)` on a traced value is `TracerBoolConversionError`. Use
  `lax.cond`/`jnp.where`, or hoist the decision to a static arg.
- **`jit` a stable callable, never a fresh `lambda`/inner `def`.** The tracing
  cache is keyed on the callable's `id()`; `jax.jit(lambda a, b: f(a, b, cfg))`
  built *inside* a hot function gets a new `id` every call → re-traces the
  identical graph every call. Bind runtime values with
  `functools.partial(f, cfg=...)` (JAX unwraps it and keys on `f`'s id + the
  bound args) or hoist the `jax.jit(...)` so it's created once. Same trap as a
  per-call inner `def`.
- **Field dtypes have extra sharp edges.** `jnp.arange`/`jnp.power`/`np.iinfo`
  over a zk field dtype fail; build ramps with `cumprod`, gather with explicit
  bounds modes. See [`poly.md` → ZKX field-dtype gotchas](poly.md#zkx-field-dtype-gotchas).

## When `jit` is slow: diagnose, don't guess

"`jit` is slow" is almost never the kernel — it's **re-tracing**, **graph
explosion**, or **recompiling on a new shape**. There are three distinct phases,
and they fail differently:

```
your Python  --trace-->  jaxpr  --lower-->  StableHLO/MLIR  --compile-->  XLA binary
              (Python)            (jaxpr→IR)                  (XLA, the slow one)
```

Turn on the two flags before theorizing:

```bash
JAX_LOG_COMPILES=1          # elapsed time for trace / lower / compile, per fn
JAX_EXPLAIN_CACHE_MISSES=1  # WHY JAX retraced — names the function + line
```

Read the symptom:

- **A re-trace every call** (`TRACING CACHE MISS ... never seen function ... but
  seen another function defined on the same line`) — a per-call `lambda`/inner
  `def`, or a freshly built round whose `meta_fields` compare by identity. Fix:
  stable callable / `functools.partial` / value-comparing meta.
- **First call takes tens of seconds, later calls fast** — graph explosion from
  an unrolled Python `for`. Dump `JAX_DUMP_IR_TO=/tmp/ir
  JAX_DUMP_IR_MODES=eqn_count_pprof` and `pprof -top` points at the exact line.
  Fix: `lax.scan` (idea #4).
- **Recompiles on a "different" input** — a shape/dtype changed (idea #2). Fix:
  pad to a fixed bucket, or stop threading a value as static.
- **A barrage of tiny `jit_iota` / `jit_add` compiles at startup** — eager
  op-by-op work (e.g. init code) outside any `@jit`. Fix: wrap it.

The [slow-tracing debugging guide](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html)
is the full version of this, with the log excerpts to match against.

## Reading list

Verified live at time of writing. The JAX pages track `latest`; if a slug 404s,
the page was renamed — search the title.

### Start here (read both, in order)

| Page | Read it for |
| --- | --- |
| [🔪 The Sharp Bits 🔪](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) | Purity, no in-place, no value-branching, the gotcha catalog. The canonical "things that bite you." |
| [How to Think in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) | The trace/transform mental model vs numpy — idea #1 and #4 in one sitting. |

### Reference (reach for the relevant one)

| Page | Read it for | Underpins |
| --- | --- | --- |
| [JIT mechanics: tracing & static args](https://docs.jax.dev/en/latest/jit-compilation.html) | Why a trace is shape-keyed; what recompiles | idea #2, `conventions.md` `@jit` |
| [Control flow & logical operators](https://docs.jax.dev/en/latest/control-flow.html) | `scan`/`cond`/`fori_loop` vs Python loops | idea #4, `conventions.md` Loops |
| [Pytrees](https://docs.jax.dev/en/latest/pytrees.html) | Registering custom nodes; flatten/unflatten | idea #3, `conventions.md` Pytree registration |
| [Pseudorandom numbers](https://docs.jax.dev/en/latest/random-numbers.html) + [PRNG design JEP](https://docs.jax.dev/en/latest/jep/263-prng.html) | Explicit keys, splitting, *why* | the PRNG trap |
| [FAQ](https://docs.jax.dev/en/latest/faq.html) | "Why does it recompile?", donation, sync | idea #2 |
| [JEP index](https://docs.jax.dev/en/latest/jep/index.html) | Design rationale straight from the team | the "why" behind the API |

### `jit`, in depth (the trace → compile pipeline)

Read in this order to actually *understand* `jit` rather than cargo-cult it.

| Page | Read it for |
| --- | --- |
| [Key concepts](https://docs.jax.dev/en/latest/key-concepts.html) | The vocabulary: tracer, jaxpr, transformation, pytree. Everything below assumes it. |
| [Tracing](https://docs.jax.dev/en/latest/tracing.html) | What "trace once with abstract values" actually does — the core mechanism. |
| [How JAX primitives lower (jaxprs)](https://docs.jax.dev/en/latest/jaxpr.html) | The IR `jit` produces; reading a jaxpr is how you *see* what got traced (and what got unrolled). |
| [Just-in-time compilation](https://docs.jax.dev/en/latest/jit-compilation.html) | `jit`, `static_argnums`, what's cached on what key. The primer. |
| [Ahead-of-time compilation](https://docs.jax.dev/en/latest/aot.html) | `jit(f).lower(...).compile()` — splits trace/lower/compile so you can inspect each and pre-compile. |
| [Stateful computations](https://docs.jax.dev/en/latest/stateful-computations.html) | How to carry "state" under `jit` without side effects — idea #3 applied. |
| [Debugging slow tracing & compilation](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html) | The diagnostic playbook above, in full: the flags, the log signatures, the `lambda`-vs-`partial` / unrolling / varying-shape gotchas. |
| [Config options](https://docs.jax.dev/en/latest/config_options.html) | Every diagnostic flag (`jax_log_compiles`, `jax_explain_cache_misses`, IR dumps). |
| [Persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html) | Caching compiles across processes — and why a poisoned cache dir surfaces as a deserialization crash (see [`dev-env.md`](dev-env.md)). |
| [Compiling ML programs via high-level tracing](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf) (Frostig, Johnson, Leary — MLSys 2018) | The founding paper. `jit` *is* this paper; read it once for the model the whole system is built on. |

### Debugging (at runtime)

Runtime debugging is *also* shaped by idea #1: a traced value has no concrete
contents, so `print(x)` prints a tracer (at trace time) and `if x == 0: raise`
is a `TracerBoolConversionError`. The fixes are functional equivalents — print
and breakpoint via `jax.debug.*`, asserts via `checkify` — that survive `jit`.

| Page | Read it for |
| --- | --- |
| [Introduction to debugging](https://docs.jax.dev/en/latest/debugging.html) | The runtime toolkit overview — what works under `jit` and what doesn't. |
| [JAX errors](https://docs.jax.dev/en/latest/errors.html) | The exception glossary: paste your `TracerBoolConversionError` / `ConcretizationTypeError` / leaked-tracer message here for the cause + fix. The fastest way out of a stuck error. |
| [`jax.debug.print` & `jax.debug.breakpoint`](https://docs.jax.dev/en/latest/debugging/print_breakpoint.html) | Printing a *value* (not a tracer) and dropping a breakpoint from inside a traced/`scan`'d body. |
| [`checkify`](https://docs.jax.dev/en/latest/debugging/checkify_guide.html) | Functional runtime asserts under `jit` — the right way to guard a div-by-zero or a length mismatch without an un-`jit`-able `if … raise`. (Exactly the guards the logup_gkr review asks for.) |
| [Debugging flags](https://docs.jax.dev/en/latest/debugging/flags.html) | `jax_debug_nans` (trap the first NaN at its source), `jax_disable_jit` (run eagerly with real values to bisect a bug), `jax_debug_infs`. |
| [`jax.debug` API](https://docs.jax.dev/en/latest/jax.debug.html) | `debug.print` / `debug.callback` / `debug.breakpoint` signatures. |

### Go deep

| Resource | Read it for |
| --- | --- |
| [Autodidax](https://docs.jax.dev/en/latest/autodidax.html) | JAX core (trace → transform → pytree) built from scratch. The single best doc for *why* purity + flat pytrees are load-bearing, not stylistic. |
| [Equinox — All of Equinox](https://docs.kidger.site/equinox/all-of-equinox/) + [tricks](https://docs.kidger.site/equinox/tricks/) | Patrick Kidger's "everything is a pytree, code is pure functions" — the strongest published statement of idea #3. Maps 1:1 onto zorch's registered-dataclass rounds. |
| [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (DeepMind) | The performance mental model — what actually costs (host-sync, recompiles, layout, communication), not folklore. |
| [awesome-jax](https://github.com/n2cholas/awesome-jax) | Curated index of further talks, posts, libraries. |

## If you read three things

1. **Sharp Bits** — the traps, so you stop hitting them.
2. **JIT mechanics** — so "it's slow" becomes "it's recompiling, here's why."
3. **Equinox — All of Equinox** — so state stops being an arg bag and becomes a
   pytree.

Then `conventions.md` is no longer a list of rules to memorize — it's the
obvious application of these four ideas to a proving stack.

To go deep specifically on `jit` — the phase that dominates compile time and
causes the subtlest "why is it slow" bugs — work the [`jit`, in depth](#jit-in-depth-the-trace--compile-pipeline)
group in order, starting from Key concepts and ending at the founding paper.
