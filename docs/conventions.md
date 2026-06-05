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
- composes sub-rounds in a static Python loop over heterogeneous shapes — e.g.
  `fold_rounds` and the GKR `ProveChain` / `VerifyChain` thread the carry +
  transcript through rounds whose message shapes vary round to round. `@jit`
  would unroll the composition into one trace; the per-round numeric bodies are
  the fusion target, not the driver. (`prove` / `verify` are the deliberate
  exception — their per-variable loop is homogeneous, so it *is* one `lax.scan`.)

A round's `_round_poly` / `_fold` are pure numeric and *could* be `@jit`'d,
but are deliberately left undecorated: they are the bodies a future marked
fused region (`stablehlo.composite`) + zkx emitter will lower to one kernel
(see `sumcheck.md`), not blanket-`@jit` candidates.

## Loops: `for` vs `lax.scan` vs `vmap`

Three ways to repeat work; the **shape of the per-iteration output** picks one.

- **`jax.vmap` — independent items, no carry.** A batch whose elements are
  independent (open N FRI queries' Merkle paths, hash N rows) is one `vmap` over
  the batch axis — on device, no Python loop and no `scan`. What is mapped must be
  a registered pytree (see [Pytree registration](#pytree-registration)): `Opening`
  is registered so `tree.open` / `reconstruct_root` `vmap` over a query batch.

- **Python `for` — static-size straight-line arithmetic.** A small,
  compile-time-known count of pure field ops (Horner, synthetic division, a
  permutation's rounds, a Merkle fold's layers) is unrolled with a Python `for`.
  It traces to straight-line element-wise IR that fuses; `lax.scan` would lower to
  `stablehlo.while`, a GPU control-flow boundary that breaks fusion and adds loop
  overhead for a tiny trip count.

- **`lax.scan` — homogeneous per-round loop inside one traced region.** When a
  round repeats with a **round-invariant output shape** and the loop is (or will
  be) one `jit`'d region, scan it: unrolling many rounds inflates the graph past
  the ZKX PTX cliff (#58). `prove` / `verify` are the case — every variable's
  round poly has the same shape. A `scan` carry must keep a **fixed shape**, so a
  halving MLE state rides in a full-width buffer with the live prefix packed at the
  front and the dead tail masked (see [`prove.py`](../zorch/prove.py)). The round
  is the carry, so it must be a registered pytree. `prove` is **generic over the
  round's summand**: it reads only `degree` + `_combine` (the `SumcheckSummand`
  Protocol in `prove.py`) and owns the buffer / mask / fold / scan, so the product
  `SumcheckRound` and the LogUp `LogupSumcheckRound` share one scan — a new
  sumcheck rides it by supplying a `_combine`, not by re-deriving the scan
  machinery. Only this per-variable inner loop scans; the heterogeneous chain over
  it (`fold_rounds`, the GKR `ProveChain`) stays a Python `for` (next bullet).

- **Python `for` — heterogeneous / non-round-invariant per-round loop.** When the
  per-round message or committed artifact changes shape across rounds it is not
  `scan`-shaped — keep it a Python loop (`fold_rounds`, the FRI fold phase, the GKR
  `ProveChain`). FRI's fold halves the codeword and commits a half-size Merkle
  layer each round; the GKR pyramid halves each layer. This is safe as a
  **host-orchestrated** loop (separate dispatches, not one giant traced graph) —
  which is also why it stays a Python loop while the transcript is host-side (the
  device-side transcript is [#3](https://github.com/fractalyze/zorch/issues/3)).

The per-round Fiat-Shamir `observe` / `sample` is wrapped in a `Round` (the
composable unit) by design, so a round loop is one of the two `Round` forms above:
`fold_rounds` (heterogeneous → Python `for`) or `prove` / `verify` (homogeneous →
`lax.scan`).

Decision, in order: independent with no carry → `vmap`; static small straight-line
arithmetic → `for`; sequential carry with a round-invariant shape in one traced
region → `lax.scan`; sequential carry whose per-round shape varies, or that is
host-orchestrated → `for`.

## Pytree registration

A `Round` — or any object — that crosses a `jax` transform boundary (passed to or
returned from `jit` / `vmap` / `scan`, or threaded as a `scan` carry) must be a
registered JAX **pytree**. Register the concrete class as a frozen dataclass:

```python
@partial(jax.tree_util.register_dataclass, data_fields=["lam"], meta_fields=[])
@dataclass(frozen=True)
class LogupSumcheckRound(Round):
    lam: Array
```

- **`data_fields`** are the `Array` leaves the transform traces over (a round's
  challenge `lam`). **`meta_fields`** are static config the trace bakes in
  (`degree`, a permutation instance) — they must be hashable and compare by value.
- Validate in `__post_init__`, never `__init__`, and only on shapes / static
  fields — `__post_init__` reruns on **tracers** during `unflatten`, so branching
  on an `Array` *value* there breaks under `jit`/`vmap`.
- Every registered class gets a `*PytreeTest`: a `tree_flatten`/`unflatten`
  round-trip that asserts the **leaf count** (so a field misclassified as
  meta-vs-data is caught), plus a "threads through `jit` as an argument" check.
  The `vmap`-over-a-leaf case (one `vmap` over a batch of `lam`s) is the
  capability registration buys that closing the object into a constant cannot —
  test it where it applies.

**Which classes.** The per-variable sumcheck rounds —
`sumcheck.prover.SumcheckRound`, `sumcheck.verifier.SumcheckRound`,
`logup_gkr.prover.LogupSumcheckRound` — are registered: `fold_rounds` loops them,
`prove` / `verify` carry them through a `lax.scan` (the per-variable loop is one
traced region), and they are `vmap`-able over their config. Both transcripts —
`DuplexTranscript` and the test `StubTranscript` — are registered for the same
reason: the `scan` threads the transcript as part of its carry. `prove.RoundMsg`
(round poly + challenge) is registered too — `prove` returns it as the `lax.scan`'s
stacked per-round output.

Do **not** pre-register what no transform threads yet — registration that buys no
capability is noise:

- **Heterogeneous-chain rounds** — `logup_gkr`'s `GkrLayerRound` and the
  `ProveChain` / `VerifyChain` wrappers. The GKR pyramid halves every layer, so
  the layers carry different shapes; the chain cannot be `vmap`/`scan`-ed and is
  composed in plain Python. Register only if a transform later threads one.
- **Plain data records** — `LayerProof`, `GkrLayer`, `LogUpGkrOutput`. They pass
  between un-`jit`-ed calls today. Register the moment one becomes `jit`/`scan`
  I/O, not before — as `prove.RoundMsg` now is.

## Comments & documentation

Comments and `docs/` prose carry only what the code can't: **WHY** — the rationale
for a non-obvious choice, a hidden constraint or invariant, a principle, an
external reference. They never restate **WHAT** the code does. The *what* already
has two drift-proof homes: the code itself (names, types) and the tests, which are
the executable usage and run on every commit. A prose "usage guide" duplicates the
tests and goes stale the first time the API moves — so we don't write one.

- **WHY, not WHAT.** `# loop over factors` above a `for` is noise; `# direct
  Lagrange, not barycentric, so a challenge on a node doesn't divide by zero`
  earns its line. If a name already says it, the comment is redundant.
- **Temporally neutral.** State the current permanent rule, not the journey. No
  "used to", "before this commit", "lands in a follow-up" — `git log` / `git
  blame` carry the chronology; in-tree narration rots within a commit or two.
- **Self-contained.** A reader has only the source tree and history. No
  session/spec labels (`Q1:A`, `Approach D`), no references to uncommitted files,
  no home/scratch paths. Link rationale in a tracked file by its repo-relative
  path.
- **A `docs/` page is design notes**, not an API tour — the *why* behind the
  shape, the principles, the non-obvious gotchas. [`sumcheck.md`](sumcheck.md) is
  the intended shape.

### Subsystem doc skeleton

"Design notes" still has a spine. Every block in zorch has to answer the two
non-negotiables, so its doc must make both explicit:

- **Why this shape** — the concept the block factors, and how it stays
  proving-scheme- and zkVM-agnostic.
- **Fusion by construction** — the load-bearing rule that keeps the block one
  fused unit by design, not by a compiler pattern-match.

Everything else is optional, added only when the block has it — design decisions
and their rationale, gotchas, what's deliberately out of scope. Don't pad a doc to
fill a template. [`sumcheck.md`](sumcheck.md) and [`coding.md`](coding.md) are the
two worked shapes — copy whichever fits.

## Naming

A leading underscore marks non-public surface. A `Round`'s only public entry is
`__call__` (plus `__init__`); its internal steps are `_`-prefixed (`_split`,
`_round_poly`, `_fold`, `_combine`). Same-package tests may still reach in and
exercise them by name — the prefix marks intent, it doesn't lock the door.
`_combine` (a per-variable round's summand) is the one such step the `prove` scan
driver also reads — the `_` marks it internal to the sumcheck machinery, not that
only the class itself may call it.

## Type annotations

Every `def` carries a full signature — annotate each parameter and the return,
on tests and nested closures too, `-> None` included (`__init__`,
`__post_init__`, `test_*`). mypy's `disallow_untyped_defs` is the gate (it runs
in pre-commit); a bare parameter or a missing return fails the hook.

Vocabulary:

- `jax.Array` for any field element or array — a scalar challenge is a 0-d
  `Array`, not a Python number.
- The per-factor MLE state threaded through a `Round` is `Sequence[Array]` where
  it is only read, `list[Array]` where it is owned or returned.
- `Transcript` (the Protocol in `transcript.py`) for the threaded transcript,
  never a concrete implementation. A test that reaches a stub-only field narrows
  first: `assert isinstance(t, StubTranscript)`.
- `Any` for a field dtype — the core names no field and treats `dtype` as
  opaque, so `dtype: Any` is the honest type, not a placeholder.

`Round.__call__(self, *args: Any) -> Any` is the one deliberately-loose
signature. A prover round is `(state, transcript) -> (state, transcript, msg)`
and its verifier dual `(claim, msg, transcript) -> (claim, transcript, r, ok)`,
so the base can't name a single shape — the subclasses give the precise ones.

mypy can't see through the ZKX `jax` fork (its shipped stubs don't parse) or
`zk_dtypes`/`zkbench`, so to the checker `Array` collapses to `Any`. The value
is catching a missing or malformed annotation, not deep array-shape checking —
write the precise type regardless; it is documentation that outlives the stubs.

## Seam conformance pins

A `Protocol` seam's conformance is mypy-enforced, not conventional: every
instance module (`testkit` included) ends with a one-line pin

```python
if TYPE_CHECKING:
    # mypy-enforced seam conformance — docs/conventions.md "Seam conformance pins".
    _: type[Permutation] = Poseidon2
```

so signature drift — a renamed method, wrong arity, an added parameter — fails
pre-commit mypy at the instance, instead of surfacing at a consumer call site
(or never, while the instance still has no in-tree consumer). A generic seam
parameterizes the pin with the instance's concrete types — the PCS pins are the
worked example ([pcs.md "Instance anatomy"](pcs.md#instance-anatomy)).

A `@runtime_checkable` `assertIsInstance` test is not a substitute: it checks
member *presence* on a live object, never signatures. The two are complementary
— the pin for instance modules, the runtime check for a test-local duck-typed
fixture that lives and dies inside its test.

With `jax.Array` collapsed to `Any` ([Type annotations](#type-annotations)), a
pin checks names, arity, and parameter count — not coordinate types. It bites
fully only where the signature carries zorch-owned nominal types, an argument
for named-dataclass wire types.

## Testing

A module's tests live in its package's `testing/` dir, named `<module>_test.py`
for the `<module>.py` they exercise — `zorch/sumcheck/prover.py` →
`zorch/sumcheck/testing/prover_test.py`. The mirrored name makes a module's test
findable by name; the `testing/` split keeps the source dir to shippable
surface. Each `testing/` dir carries a `BUILD.bazel` registering every test as a
`py_test`, so `bazel test //...` — not the `pytest` job alone — is the single
source of truth for "all tests pass" (`pytest` adds only the jax-fork coverage
for the `manual`-tagged tests).

`__init__.py` is header-only — the copyright line, nothing more. A consumer
imports from the submodule (`from zorch.sumcheck.prover import ...`), never the
package, so the package surface is whatever the submodules export and never
drifts from a hand-maintained re-export list. The one exception is a
`testing/__init__.py`, which may host shared fixtures the package's tests import
— e.g. `zorch/sumcheck/testing`'s `product` / `eval_mle_oracle` reference
oracles.

Tests subclass `absltest.TestCase` and assert through `self.assert*` /
`self.assertRaises` — never a bare `def test_*` + `assert`. A bare `assert` is a
statement Python drops under `-O`, so under optimization a positive check or an
`assert False` negative silently no-ops and a broken invariant passes unnoticed;
`self.assert*` are method calls that always run. CI invokes the suite via
`pytest`; each file stays runnable standalone through `absltest.main()`.
