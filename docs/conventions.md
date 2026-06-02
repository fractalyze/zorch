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
  `zorch.prove` loops the round, and `prover.SumcheckRoundBase.__call__` wraps the
  round-poly/fold arithmetic around the host-side transcript `observe` /
  `sample`. Decorating these would inline everything into one trace and pull
  the transcript ops in with it.

A round's `_round_poly` / `_fold` are pure numeric and *could* be `@jit`'d,
but are deliberately left undecorated: they are the bodies a future marked
fused region (`stablehlo.composite`) + zkx emitter will lower to one kernel
(see `sumcheck.md`), not blanket-`@jit` candidates.

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
