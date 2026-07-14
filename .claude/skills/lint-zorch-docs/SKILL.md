---
name: lint-zorch-docs
description: |
  zorch-specific documentation lint. Wraps /workflow:lint-docs for the
  universal checks, then layers zorch's two non-negotiables on top:
  subsystem doc-coverage parity (every zorch/<subsystem>/ has a
  docs/**/<subsystem>.md and a hub row), the two-section subsystem skeleton
  (why-the-shape/agnostic + fusion-by-construction), ZKX field-dtype gotcha
  dedup (stated canonically only in poly.md), the CLAUDE.md = pointer +
  two-non-negotiables + dev-env rule, and agnostic naming (a scheme/zkVM is
  named only as the consumer boundary). The subsystem list is auto-detected
  from zorch/, so a new block inherits every check the moment its directory
  lands.
  TRIGGER when: (1) /lint-zorch-docs, (2) after touching CLAUDE.md /
  README.md / docs/, (3) after adding a new subsystem, (4) before a docs PR.
  SKIP when: outside the zorch repo; the universal pass already ran clean and
  only zorch-specific scope changed.
user_invocable: true
allowed_tools: Read, Write, Edit, Glob, Grep, Bash, Skill
---

# Lint zorch Docs

Two-stage lint on top of `/workflow:lint-docs`:

1. Universal pass — invoke `/workflow:lint-docs`, surface its report.
1. zorch-specific pass — checks Z1–Z5 below, merged into the same report.

The conventions enforced here live in `docs/reference/conventions.md` (the "subsystem doc
skeleton" + WHY-not-WHAT rules) and the **two non-negotiables** in `CLAUDE.md` /
`README.md` (proving-scheme- & implementation-agnostic; fusion by construction). When a
rule drifts, fix the convention doc and re-run.

## Bail-out

If `$(basename $(git rev-parse --show-toplevel))` does not start with `zorch`,
stop and point the user at `/workflow:lint-docs`.

## Auto-detect the subsystem list

A *subsystem* is a top-level directory under `zorch/` that ships a design block —
it holds a non-`__init__`, non-`_test`, non-`bench_*` `.py` with a module
docstring (a benchmark script is not a primitive). Exclude the support dirs
`testing/`, `testkit/`, `utils/`, and every `__pycache__`. Walk
`zorch/*/` — never hard-code that set, so a new block is checked the moment it
lands. Map a subsystem to its doc by replacing `_` with `-`, then resolving it
under the layout subdirs with a `docs/**/<name>.md` glob (never assume a flat
`docs/`): building blocks live in `docs/blocks/` (`logup_gkr` →
`docs/blocks/logup-gkr.md`), full SNARKs in `docs/schemes/` (`spartan` →
`docs/schemes/spartan.md`).

A *nested* block earns its own page only when its concern stands alone — an
editorial call, not a tree fact. Today that is just `commit/jagged`;
`hash/poseidon2`, by contrast, is documented inside `hash.md`. A fully-recursive
walk would over-detect these sub-components, so this short own-page list stays
explicit with its rationale, separate from the auto-walked top-level set.

## Checks

### Z1. Subsystem doc-coverage parity (High)

For every detected subsystem, verify both:

- a doc exists at `docs/**/<subsystem>.md` (a building block under
  `docs/blocks/`, a full SNARK under `docs/schemes/`).
- `docs/README.md`'s hub table has a row linking it.

Flag a subsystem missing either. Softer (Medium): a non-`bench_*` source module
that grew a module docstring but is named in no `docs/**/*.md` — a new primitive
landed without a doc home (e.g. a fold helper added beside an encoder). Name it;
the user decides the home.

### Z2. Subsystem skeleton (High)

Every block doc must answer the two non-negotiables `conventions.md` mandates: a
section on **why this shape** (the concept it factors, how it stays
proving-scheme- and implementation-agnostic) and a section on **fusion by construction**.
Heuristic: the doc has a `## …why…` heading and a `## …fusion…` heading
(case-insensitive). `coding.md` is the conventions-blessed alternate shape — its
"Why this module exists" + "don't hand-roll the NTT" cover both ideas under
different titles — so treat a near-miss as a prompt to confirm both ideas are
present, not an automatic failure.

### Z3. ZKX field-dtype gotcha dedup (Medium)

The ZKX field-dtype limits are stated canonically in
`blocks/poly.md#field-dtype-gotchas`. Any other doc that re-explains a limit instead
of linking that anchor is a duplicate. Detection: grep the other docs for the limit keywords poly.md's gotcha
bullets use (`iota`, `reduce_sum`, `lax.shift`, `jnp.tile`, `arange`, …) and check
the surrounding lines link `poly.md` — seed the set from those bullets so a new
limit is covered when poly.md gains it. A block-specific *consequence* (jagged's
trace growing with `l_max`) is fine; a restated *rule* is the dup.

### Z4. CLAUDE.md shape (Medium)

zorch's CLAUDE.md is a pointer to README.md + docs/, plus exactly two allowed
content sections: **Two non-negotiables** (agnostic + fusion) and **Development
environment**. Flag any other content section, or non-negotiables prose that has
drifted from `README.md`'s Design Philosophy. (Looser than the universal
pointer-only rule — the two-non-negotiables section is deliberate.)

### Z5. Agnostic naming in docs (Low)

A zkVM name (`sp1`, `openvm`, `zisk`, `risc0`, `jolt`) or the consumer
(`whir-zorch`) may appear only as the **consumer boundary** ("lives in the
consumer", "the consumer's concern", "adapted from whir-zorch"). Generic scheme
names (FRI, Basefold, WHIR, STARK, GKR, sumcheck) are fine anywhere. Flag a
mention that reads as a zorch dependency or feature rather than a boundary, for
review.

## Procedure

1. Bail-out check.
1. Auto-detect subsystems from `zorch/`.
1. Invoke `/workflow:lint-docs`; capture its report.
1. Run Z1–Z5 against the detected set.
1. Merge into one report:

   ```md
   # /lint-zorch-docs report

   ## Universal (from /workflow:lint-docs)
   ...

   ## zorch-specific
   ### High / Medium / Low
   - Z<n> — <finding>. Suggested fix: <action>.

   ## Summary
   <combined counts; --fix pointer if applicable>
   ```

1. With `--fix`: defer mechanical fixes to the upstream skill. For a Z1 missing
   doc, offer to scaffold `docs/blocks/<subsystem>.md` (or `docs/schemes/` for a
   full SNARK) from the skeleton (copy the
   nearest worked shape, fill the headings) and add the hub row — but draft the
   WHY prose for user review, never invent it. For Z2 / Z4 prompt per finding via
   `AskUserQuestion`; the rest report-only.
