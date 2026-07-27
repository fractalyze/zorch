---
name: lint-zorch-srcs
description: |
  zorch-specific source-tree lint enforcing the two non-negotiables.
  Detects consumer/zkVM leakage into zorch/ code (the proving-scheme- and
  zkVM-agnostic rule), consumer/zkVM names in tests or fixtures,
  kernel-splitting ops inside a fused_region body or a Round's
  _round_poly/_fold (the fusion-by-construction rule), and bare asserts in
  tests. Companion to /lint-zorch-docs — the same agnostic + fusion
  philosophy applied to code. Auto-detects subsystems from zorch/, so a new
  block inherits every check the moment its directory lands.
  TRIGGER when: (1) /lint-zorch-srcs, (2) after touching zorch/ source,
  (3) after adding a subsystem, (4) before a src PR.
  SKIP when: outside the zorch repo; nothing under zorch/ changed since the
  last clean run.
user_invocable: true
allowed_tools: Read, Glob, Grep, Bash
---

# Lint zorch Sources

zorch-specific structural lint for the source tree. Companion to
`/lint-zorch-docs`; the two non-negotiables from `README.md` / `CLAUDE.md`
applied to code instead of docs. Type annotations are already gated by mypy in
pre-commit (`disallow_untyped_defs`) — out of scope here.

## Bail-out

If `$(basename $(git rev-parse --show-toplevel))` does not start with `zorch`,
stop.

## Auto-detect the subsystem list

`ls -d zorch/*/` minus the support set `testing/`, `testkit/`, `utils/`, and
`__pycache__` — never hard-code the set. The source checks (S1–S3) scan the whole
`zorch/` tree regardless of nesting; only S5 test-parity walks this top-level set.

## Forbidden tokens

- **zkVM names** — always forbidden in zorch source, code or test:
  `sp1`, `openvm`, `zisk`, `risc0`, `jolt`.
- **The consumer** (`whir-zorch` / `whir_zorch`) — forbidden as a *code
  dependency*; allowed only in a docstring/comment that frames the agnostic
  boundary.
- **NOT forbidden** — generic scheme names the library serves: FRI, Basefold,
  WHIR, STARK, GKR, sumcheck. Naming the scheme a block factors is the point of
  an agnostic block, not a leak.

## Checks

### S1. Agnostic code — no consumer/zkVM dependency (Critical)

Source under `zorch/` (excluding `*_test.py` and `*/testing/`) must not *depend*
on a consumer or zkVM. Two passes:

1. **Imports (hard fail).**
   `grep -rnE '^[[:space:]]*(import|from)[[:space:]].*(whir|sp1|openvm|zisk|risc0|jolt)' zorch/ --include='*.py'`
   — any hit is a leak.
1. **Identifiers (classify).** Token-grep the forbidden set across `zorch/`; a
   hit in *code* (a name, call, or assignment) is a leak, a hit in a
   docstring/comment that frames the boundary ("lives in the consumer", "the
   consumer's …", "adapted from whir-zorch", "an SP1-trace concern") is allowed.
   List the allowed mentions so the boundary stays visible; flag the rest.

### S2. Agnostic tests — no consumer/zkVM names (Critical)

`*_test.py` and `*/testing/` must name no consumer or zkVM at all — stricter than
S1, no docstring exception. Tests reuse the agnostic fixtures and self-anchor
goldens; a cross-impl check against a specific consumer is a one-time thing
recorded on the issue, not a committed fixture. Grep the forbidden set across the
test surface; any hit is a finding.

### S3. Fusion by construction — straight-line fused regions (High)

The body that must stay one kernel — the `decomposition` passed to
`fused_region` (today only `poseidon2`'s `permutation`), and each Round's
`_round_poly` / `_fold` — may use only element-wise field ops plus the one
inherent `Σ`. `fusion.py`'s contract is "no loops, reductions, or gathers", so
flag any of: a control-flow loop (`lax.fori_loop` / `lax.while_loop` / `lax.scan`)
— the body must be unrolled straight-line — `x.at[i].set(...)` (scatter),
`jnp.dot` / `jnp.matmul`, `jnp.tile`, `jnp.take` / gather, or a *second* reduction
(`jnp.sum` / `lax.reduce`) beyond the single trailing `jnp.sum(..., axis=-1)` that
is the round poly's own `Σ`. Locate the bodies
(`fused_region(` call sites; `def _round_poly` / `def _fold` / `def fold`), grep
the op set within them, and ignore matches inside comments. (`.at[].set`
*outside* a fused region — a sponge overwriting its rate lanes, a compression
building its pre-image — is fine; it is not in the marked body.)

### S4. No bare assert in tests (Medium)

`conventions.md`: tests subclass `absltest.TestCase` and assert through
`self.assert*`; a bare `assert` is dropped under `python -O`, so a broken
invariant passes unnoticed. `grep -rnE '^[[:space:]]*assert[[:space:]]' zorch
--include='*_test.py'` — any hit is a finding.

### S5. Test presence parity (Low)

Every detected subsystem has tests — a `testing/` dir or sibling `*_test.py`.
Flag a subsystem that ships source but no tests.

## Procedure

1. Bail-out check.
1. Auto-detect subsystems from `zorch/`.
1. Run S1–S5 against the detected set.
1. Report by severity; each finding names the file (and subsystem) so the user
   can act block-by-block:

   ```md
   # /lint-zorch-srcs report

   ## Critical / High / Medium / Low
   - S<n> — <file> — <finding>. Suggested fix: <action>.

   ## Summary
   <counts; --fix note if applicable>
   ```

1. With `--fix`: report-only by default. S1 / S2 leaks and S3 splits are
   judgment calls (is the dependency real, is the op genuinely splitting) —
   prompt per finding via `AskUserQuestion` rather than rewriting.

## Failure modes

- **False positive on boundary-framing prose (S1).** A docstring naming the
  consumer to say what is *not* here is correct, not a leak. Classify by context;
  list allowed mentions instead of flagging them.
- **Macro / aliased identifiers escape the grep (S1).** A token built by string
  concat or re-exported under a neutral alias won't match. Documented limitation;
  a periodic manual full-text grep is the backstop.
- **`jnp.sum` ambiguity (S3).** One trailing `axis=-1` sum is the inherent `Σ`;
  the check flags a *second* reduction or a gather. A genuinely batched single
  reduction is not a split.
- **Helper-hidden ops escape the body grep (S3).** The op scan is lexical — the
  fused body only, not the helpers it calls (`apply_matrix`, `_combine`,
  `summand_evals`). A split factored into a helper won't match. Helpers
  self-enforce element-wise via `linear.py`'s no-`dot`/`reduce`/`gather`
  contract; a periodic manual check of them is the backstop.
