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
  `zorch.prove` loops the round, and `prover.SumcheckRound.__call__` wraps the
  round-poly/fold arithmetic around the host-side transcript `observe` /
  `sample`. Decorating these would inline everything into one trace and pull
  the transcript ops in with it.

A round's `_round_poly` / `_fold` are pure numeric and *could* be `@jit`'d,
but are deliberately left undecorated: they are the bodies a future marked
fused region (`stablehlo.composite`) + zkx emitter will lower to one kernel
(see `sumcheck.md`), not blanket-`@jit` candidates.

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
`logup_gkr.prover.LogupSumcheckRound` — are registered: the `prove` / `verify` /
`fold_rounds` drivers loop them, a future `lax.scan` carries them (issue #58), and
they are `vmap`-able over their config. A device-side transcript threaded as a
`scan` carry falls under the same rule when it lands (issue #58).

Do **not** pre-register what no transform threads yet — registration that buys no
capability is noise:

- **Heterogeneous-chain rounds** — `logup_gkr`'s `GkrLayerRound` and the
  `ProveChain` / `VerifyChain` wrappers. The GKR pyramid halves every layer, so
  the layers carry different shapes; the chain cannot be `vmap`/`scan`-ed and is
  composed in plain Python. Register only if a transform later threads one.
- **Plain data records** — `RoundMsg`, `LayerProof`, `GkrLayer`,
  `LogUpGkrOutput`. They pass between un-`jit`-ed calls today. Register the moment
  one becomes `jit`/`scan` I/O, not before.

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
