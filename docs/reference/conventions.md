# Coding conventions

> Code, symbols, and file paths are English.

New to JAX? Read [`jax.md`](jax.md) first — the mental models these rules follow
from, plus the canonical external references. This page is the rules; that one is
the why.

## Device-first, and what "host" is allowed to mean

Every numeric path is designed to trace — device-resident, capturable as one
CUDA graph — by default. Host execution is the exception, and it comes in
exactly two shapes:

- **Inexpressible math** stays host as plain Python/bigints: arithmetic whose
  values exceed every fixed-width lane (balanced-lift reconstructions and norm
  bounds over the full RNS range, the LNP challenge gate's ~300-bit unreduced
  products). No array layer helps here — a field dtype would reduce away the
  very magnitude being measured — so the performance lever is algorithmic, not
  codegen, and numpy object arrays are acceptable as containers for exact
  ints. This is the same boundary lattice-frx pins for its norms and lifts.
- **Array-expressible work that merely *runs* on host** is still written in
  `frx`, never numpy. A numpy path is invisible to XLA codegen — no fusion, no
  emitter, no backend retarget — so its performance can never be fixed, only
  rewritten; the same function in `frx` runs on the CPU backend today and
  moves on-device by changing nothing but the platform. Bare numpy is reserved
  for the bigint containers above and for test oracles, where independence
  from the traced stack is the point.

When a protocol needs a host decision inside a traced pipeline (a rejection
branch, a repeat loop), shape it as a host loop around a traced core with a
fixed budget, and keep the traced core's inputs device-resident — the host
touches verdicts, not arrays.

## `@jit`

`@jit` the leaf numeric kernels; compose them in plain Python.

**Apply `@jit`** to a pure field/array function whose output shape is static
(a function of input *shapes*, not values). Static-trip-count Python loops
inside are fine — `jit` unrolls them.

**Do not `@jit`** when the function:

- returns a Python value from structure — e.g. `zorch.utils.bits.log2_strict_usize`
  returns an `int` from a length; `jit` would trace it away.
- composes sub-rounds in a static Python loop over heterogeneous shapes — e.g.
  `fold_rounds` and the GKR `prove_rounds` / `verify_rounds` thread the carry +
  transcript through rounds whose message shapes vary round to round. `@jit`
  would unroll the composition into one trace; the per-round numeric bodies are
  the fusion target, not the driver. (`prove` / `verify` are the deliberate
  exception — their per-variable loop is homogeneous, so it *is* one `lax.scan`.)
- builds a constant from static (non-`Array`) arguments alone — e.g.
  `zorch.poly.univariate.compute_inv_vandermonde` assembles a matrix from
  `degree` + `dtype`; with no array input there is nothing for `jit` to
  specialize over.

A round's `_round_poly` / `_fold` are pure numeric and *could* be `@jit`'d,
but are deliberately left undecorated: they are the bodies a future marked
fused region (`stablehlo.composite`) + XLA emitter will lower to one kernel
(see `sumcheck.md`), not blanket-`@jit` candidates.

### One `@jit` boundary per round — single-zone or per-island

A round body needs **exactly one** `@jit` boundary around it — never zero (eager
dispatch *decomposes* the fused composite, so the boundary is the perf lever, not
just a cache) and never two (a nested `@jit` lowers to a *call* the single-kernel
rewriter rejects,
[`fusion.py`](https://github.com/fractalyze/hash-frx/blob/main/hash_frx/fusion.py)).
Two shapes satisfy this,
and which one a scheme uses is **forced by whether a host-side op interleaves the
round loop**, not chosen for style:

- **Single-zone** — one `@jit` over the whole `open` / `commit` body; the driver
  loop and the round bodies stay plain Python inside it (`basefold`, `fri`:
  `_open_*_body` wraps `fold_rounds`). Available only when the loop is
  **host-op-free** — the `DuplexTranscript` `observe` / `sample` are device ops
  and trace straight through.

- **Per-island** — an **eager** Python driver loop, with each maximal
  host-op-free compute span its own `@jit` (`jagged`'s per-layer rounds; `whir`'s
  eager `_open_body` over `_island_*`). Required when a host op sits *inside* the
  loop and so cannot be traced: a PoW `grind` (`pow_bits > 0` host-reads the
  witness to validate it — `TracerBoolConversionError` under an outer `@jit`;
  `pow_bits == 0` skips the read and traces), or a `sample_bits → int(...)` query
  index. `whir`'s prover grinds between every fold, so it is per-island; its
  **verifier** has no grind and stays single-zone.

Both keep the *driver* out of `@jit` (the heterogeneous round loop is the
host-orchestrator, never the fusion target) and treat the *round body* as the
fusion unit — single-zone leaves the bodies undecorated inside the one enclosing
`@jit`, per-island makes each its own `@jit`. A grind-bearing scheme **cannot** be
single-zone; a grind-free one need not fragment into islands. Per-island costs a
per-shape recompile of each island (WHIR's tables halve each round → recompiles
per distinct shape), so hoist a shape-stable heavy composite (Poseidon2, an NTT)
into its own island — it then lowers once instead of re-lowering inside every
enclosing shape.

## Loops: `for` vs `lax.scan` vs `vmap`

The **shape of the per-iteration output** picks the tool.

- **`frx.vmap`** — independent items, no carry (N FRI queries' Merkle paths, N
  row hashes). What is mapped must be a registered pytree.
- **Python `for`, straight-line** — a compile-time-known count of pure field ops
  (Horner, synthetic division, a permutation's rounds). It traces to element-wise
  IR that fuses, where `lax.scan` would lower to a `stablehlo.while` boundary
  that breaks fusion for a tiny trip count.
- **`lax.scan`** — a homogeneous carry with a round-invariant output shape inside
  one traced region; unrolling many rounds inflates the graph past the XLA PTX
  cliff. A carry must keep a fixed shape, so a halving MLE state rides a
  full-width buffer with the live prefix packed at the front and the dead tail
  masked (see [`prove.py`](../../zorch/prove.py)). `prove` is generic over the
  round's summand — it reads `degree` + `_combine` and owns the buffer, mask,
  fold and scan, so a new sumcheck rides it by supplying a `_combine`.
- **Python `for`, heterogeneous** — when the per-round message or committed
  artifact changes shape (`fold_rounds`, the FRI fold phase, the GKR
  `prove_rounds`). Safe as a host-orchestrated loop of separate dispatches.

A `lax.scan` reached from eager code needs a **stable body**: the trace cache is
keyed on the body's identity, so one built per call recompiles an identical
graph every time — orders of magnitude over the work being scanned, and silent.
`zorch/scan_body.py` memoizes a body factory for that; a scan inside a `@jit`
zone needs nothing, since the jit cache absorbs it.

The **fixed-width-mask exception** reaches one level up when the shrink is
*predictable*: pad every layer to the max width with the fold-neutral fraction,
carry the live count as a traced threshold (`poly.geq.VirtualGeq`), and on a
padding round select the unchanged carry — sponge leaves included — so
Fiat-Shamir never over-advances. `logup_gkr.circuit.build_jagged_pyramid` rolls
this way, O(1) in the layer count. Reach for it only when the step count drives
compile time or dispatch latency past the cost of the masking; the jagged prove
does not, and stays an unrolled `prove_rounds` on a host-FS transcript.

## Pytree registration

Any object crossing a `frx` transform boundary — passed to or returned from
`jit` / `vmap` / `scan`, or threaded as a `scan` carry — must be a registered
FRX pytree, as a frozen dataclass:

```python
@partial(frx.tree_util.register_dataclass, data_fields=["lam"], meta_fields=[])
@dataclass(frozen=True)
class LogupSumcheckRound(Round):
    lam: Array
```

- **`data_fields`** are the `Array` leaves the transform traces over;
  **`meta_fields`** are static config baked into the trace, and must be hashable
  and **compare by value**. Identity equality does not error — it silently
  re-traces the enclosing jit zone per freshly built instance (~2 min/call on a
  replay whose kernels run in 20 ms). An object-typed meta field needs explicit
  `__eq__`/`__hash__`.
- Validate in `__post_init__`, never `__init__`, and only on shapes and static
  fields — it reruns on **tracers** during `unflatten`, so branching on an
  `Array` value there breaks under `jit`/`vmap`.
- Every registered class gets a `*PytreeTest`: a flatten/unflatten round-trip
  asserting the **leaf count** (catching a meta-vs-data misclassification), plus
  a threads-through-`jit` check, plus the `vmap`-over-a-leaf case where it
  applies — the capability registration buys that a closed-over constant cannot.

**Register what a transform threads, nothing else.** The per-variable sumcheck
rounds and `DuplexTranscript` are registered because `prove` / `verify` carry
them through a `lax.scan`; `prove.RoundMsg` because it is that scan's stacked
output. Heterogeneous-chain rounds (`GkrLayerRound`, the `prove_rounds` drivers)
and plain data records (`LayerProof`, `GkrLayer`) are not: the pyramid roll
threads planes as `Array`s rather than layer objects. Registration that buys no
capability is noise; register the moment one becomes `jit`/`scan` I/O.

## Comments & documentation

Comments and `docs/` prose carry only what the code can't: the rationale for a
non-obvious choice, a hidden constraint or invariant, an external reference.
Never **WHAT** the code does — that has two drift-proof homes already, the code
and the tests. A prose usage guide duplicates the tests and rots at the first API
move, so we don't write one.

- **WHY, not WHAT.** `# loop over factors` is noise; `# direct Lagrange, not
  barycentric, so a challenge on a node doesn't divide by zero` earns its line.
- **Temporally neutral.** No "used to", "before this commit", "lands in a
  follow-up" — `git blame` carries the chronology and in-tree narration rots
  within a commit or two.
- **Self-contained.** A reader has only the source tree and history: no
  session/spec labels, no uncommitted files, no scratch paths.
- **No bare external symbol names.** Another project's symbol rots silently when
  it renames — the reader can't grep it here to notice. Name the durable concept
  and permalink the pinned line. The project name alone is fine.
- **A `docs/` page is design notes**, not an API tour.

### Subsystem doc skeleton

Every block page answers three things, and everything else is optional — don't
pad to fill a template. [`sumcheck.md`](../blocks/sumcheck.md) and
[`coding.md`](../blocks/coding.md) are the worked shapes.

- **Why this shape** — the concept the block factors, and how it stays
  proving-scheme- and implementation-agnostic.
- **Fusion by construction** — the rule keeping the block one unit by design,
  not by a compiler pattern-match.
- **Where it sits in the composition vocabulary** — which components are stage
  roles, which are rounds, the claim each reduces and to what, and the module
  path. A block with no stage role says so; silence is not an answer. Two pages
  describing one protocol in two vocabularies is how a reader concludes they are
  two protocols. Contracts live in
  [`stage-composition.md`](../composition/stage-composition.md); a page names its
  own instances rather than re-deriving the model.

## Naming

A leading underscore marks non-public surface. A `Round`'s only public entry is
`__call__`; its steps are `_`-prefixed (`_split`, `_round_poly`, `_fold`,
`_combine`). The prefix marks intent, not access — same-package tests reach in by
name, and the `prove` scan driver reads `_combine`.

## Type annotations

Every `def` carries a full signature, `-> None` included, on tests and closures
too; mypy's `disallow_untyped_defs` is the gate. Vocabulary: `frx.Array` for any
field element (a scalar challenge is a 0-d `Array`, not a Python number);
`Sequence[Array]` where MLE state is read and `list[Array]` where it is owned;
the `Transcript` Protocol rather than an implementation; `Any` for a field dtype,
which is honest rather than a placeholder — the core names no field.

`Carry` is a type variable; a concrete carry is named for what it holds
(`RunningClaim`, `LayerClaim`, `FoldState`), never `Carry`.

mypy cannot see through FRX (its stubs don't parse) or `zk_dtypes`/`zkbench`, so
`Array` collapses to `Any`. The value is catching a missing or malformed
annotation; write the precise type anyway, as documentation outliving the stubs.

## Seam conformance pins

A `Protocol` seam's conformance is mypy-enforced, not conventional: every
instance module ends with a one-line pin

```python
if TYPE_CHECKING:
    _: type[Permutation] = Poseidon2
```

so signature drift fails pre-commit at the instance rather than at a consumer
call site — or never, while the instance has no in-tree consumer. A generic seam
parameterizes the pin with concrete types
([pcs.md](../blocks/pcs.md#instance-anatomy) is the worked example); a module
pinning two seams names each, since mypy rejects re-annotating `_`.

A `@runtime_checkable` `assertIsInstance` is complementary, not a substitute: it
checks member presence on a live object, never signatures. With `Array` collapsed
to `Any` a pin checks names and arity, biting fully only where the signature
carries zorch-owned nominal types — an argument for named-dataclass wire types.

## Testing

A module's tests live in its package's `testing/` dir, named `<module>_test.py`
for the `<module>.py` they exercise — `zorch/sumcheck/prover.py` →
`zorch/sumcheck/testing/prover_test.py`. The mirrored name makes a module's test
findable by name; the `testing/` split keeps the source dir to shippable
surface. Each `testing/` dir carries a `BUILD.bazel` registering every test as a
`py_test`, so `bazel test //...` — not the `pytest` job alone — is the single
source of truth for "all tests pass" (`pytest` adds only the FRX coverage
for the `manual`-tagged tests).

`__init__.py` is header-only — the copyright line, nothing more. A consumer
imports from the submodule (`from zorch.sumcheck.prover import ...`), never the
package, so the package surface is whatever the submodules export and never
drifts from a hand-maintained re-export list. The one exception is a
`testing/__init__.py`, which may host shared fixtures the package's tests import
— e.g. `zorch/sumcheck/testing`'s `product` / `eval_mle_oracle` reference
oracles.

Tests draw field and curve-point elements as the Montgomery-form dtypes —
every `zk_dtypes` family that ships a `_mont` sibling (`koalabear_mont`,
`koalabearx4_mont`, the babybear/goldilocks families, `bn254_sf_mont`, and
the bn254 G1/G2 `affine`/`jacobian`/`xyzz` point types): Montgomery is the
production encoding the GPU kernels compute in, so tests exercise the
arithmetic path the prover ships. Reach for the bare canonical dtypes only
when a test is *about* the canonical integer encoding itself, and mark that
line `# canonical-encoding test` — the `mont-test-dtypes` pre-commit hook
rejects any other bare-canonical use in a `*_test.py`.

Tests subclass `absltest.TestCase` and assert through `self.assert*` /
`self.assertRaises` — never a bare `def test_*` + `assert`. A bare `assert` is a
statement Python drops under `-O`, so under optimization a positive check or an
`assert False` negative silently no-ops and a broken invariant passes unnoticed;
`self.assert*` are method calls that always run. CI invokes the suite via
`pytest`; each file stays runnable standalone through `absltest.main()`.

### Agnostic tests and goldens

zorch is proving-scheme- and implementation-agnostic (the repo's defining non-negotiable,
[`../README.md`](../../README.md)), so its tests must read as the library's own —
no consumer identity leaks in.

- **Name no consumer or application.** Test names, helper names, and comments
  don't mention a downstream (`_sp1_input_hash`, "SP1's
  reference fixture"). A future reader neither knows nor needs to know what the
  consumer is, and the name rots when that consumer moves.
- **Self-anchor the golden** to zorch's own deterministic output. Don't drag in
  machinery whose sole purpose is to chase an external impl — a `SplitMix64`
  ported only to match a consumer's RNG stream, a config-fingerprint hash zorch
  never computes. Once an external anchor is shown not to reproduce, it is dead
  weight; drop it.
- **The cross-impl check is one-time, not a fixture.** Validate zorch output
  against an independent prover of the same circuit once, record it on the issue,
  and don't bake that external anchor into the suite. In the committed golden,
  one neutral provenance line ("checked once against an independent prover") is
  the whole comment.
- **Reuse fixtures.** Grep the package before writing a helper — a near-duplicate
  of an existing `random_first_layer` is churn.
