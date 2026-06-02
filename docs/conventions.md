# Conventions

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
  round-poly/fold arithmetic around the host-side transcript `observe` /
  `challenge`. Decorating these would inline everything into one trace and pull
  the transcript ops in with it.

A round's `round_poly` / `fold` are pure numeric and *could* be `@jit`'d,
but are deliberately left undecorated: they are the bodies a future marked
fused region (`stablehlo.composite`) + zkx emitter will lower to one kernel
(see `sumcheck.md`), not blanket-`@jit` candidates.

## Subsystem docs

A block, seam, or primitive gets one doc under `docs/`, indexed from
[`README.md`](README.md). [`coding.md`](coding.md) is the canonical example —
copy its shape rather than growing a new one. Every section answers *why*
(background, rationale, the rule that follows); the API surface and a usage
walkthrough are *what*, and they stay in the code and its tests.

The shape, in order:

1. `# <name> — <one-line scope>`.
2. **`## Why this module exists`** — the shared concept the block factors and why
   it is a seam, including how it stays proving-scheme- and zkVM-agnostic. This
   is non-negotiable #1 in [`../CLAUDE.md`](../CLAUDE.md). *Required.*
3. **`## The one rule: …`** — the single load-bearing constraint that keeps the
   block fusion-by-construction (non-negotiable #2): the one thing a contributor
   must not violate. *Required.*
4. **`## Design rules`** — the non-obvious decisions, each with its reason.
5. **`## Gotcha(s)`** — the sharp edges a reader will hit.
6. **`## Deliberately out of scope`** — what is not built yet, and why carrying it
   early would be surface to maintain, not value.
7. **`## Tests`** — how to run them, and what the *independent* oracle proves.
   *Required.*

Sections 4–6 appear only when the block has such content; an empty section is
noise. Sections 2 and 3 are not optional because every block must show how it
honors the two non-negotiables — a doc that drops either has hidden a design
question, not just a doc.
