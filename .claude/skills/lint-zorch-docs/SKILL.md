---
name: lint-zorch-docs
description: |
  zorch-specific documentation review. `tools/lint_docs.py` runs as a
  pre-commit hook and already decides everything the tree can answer —
  subsystem/doc parity, module reachability, canonical-claim duplication,
  dangling symbols and paths, link targets. This skill covers only what
  needs judgment: whether a page answers why-this-shape, fusion-by-
  construction, and where it sits in the composition vocabulary; whether
  CLAUDE.md has stayed a pointer; and whether a scheme or zkVM is named as
  anything but the consumer boundary. Wraps /workflow:lint-docs for the
  universal checks.
  TRIGGER when: (1) /lint-zorch-docs, (2) after touching CLAUDE.md /
  README.md / docs/, (3) after adding a new subsystem, (4) before a docs PR.
  SKIP when: outside the zorch repo; the universal pass already ran clean and
  only zorch-specific scope changed.
user_invocable: true
allowed_tools: Read, Write, Edit, Glob, Grep, Bash, Skill
---

# Lint zorch Docs

The mechanical half is not here. `tools/lint_docs.py` owns every check the
filesystem can settle, because a rule written as prose drifts silently while a
rule that runs fails the commit that breaks it — this file previously named
`commit/jagged` as the one nested block earning its own page, a path that never
existed. Nothing below restates a tree fact.

Three stages:

1. `python3 tools/lint_docs.py` — mechanical. Fix what it reports first; the
   judgment checks read a moving target otherwise.
1. `/workflow:lint-docs` — universal prose checks. Surface its report.
1. J1–J3 below, merged into the same report.

The conventions these enforce live in `docs/reference/conventions.md` (the
subsystem doc skeleton and the WHY-not-WHAT rule) and the **two non-negotiables**
in `CLAUDE.md` / `README.md`. When a rule drifts, fix the convention doc and
re-run.

## Bail-out

If `$(basename $(git rev-parse --show-toplevel))` does not start with `zorch`,
stop and point the user at `/workflow:lint-docs`.

## Checks

### J1. Subsystem skeleton (High)

Every block page must answer the three things `conventions.md` mandates: **why
this shape** (the concept it factors, and how it stays proving-scheme- and
implementation-agnostic), **fusion by construction**, and **where it sits in the
composition vocabulary** (which components are stage roles, which are rounds, the
claim each role reduces and to what).

Headings are a hint, not the test — `coding.md` covers the first two under
different titles and is conventions-blessed. Read for the ideas. A block with no
stage role passes the third by saying so; silence does not.

### J2. CLAUDE.md shape (Medium)

zorch's CLAUDE.md is a pointer to README.md and docs/, plus exactly two content
sections: **Two non-negotiables** and **Development environment**. Flag any other
content section. The non-negotiables must agree with README.md's Design
Philosophy — but agreement by *linking the same canonical statement*, not by
carrying a second copy of it, which is how the fusion claim drifted into
contradiction with what the north star measured.

### J3. Agnostic naming (Low)

A zkVM name (`sp1`, `openvm`, `zisk`, `risc0`, `jolt`) or a consumer repository
may appear only as the **consumer boundary** ("lives in the consumer", "adapted
from …"). Generic scheme names (FRI, Basefold, WHIR, STARK, GKR, sumcheck) are
fine anywhere. Flag a mention that reads as a zorch dependency or feature.

## Procedure

1. Bail-out check.
1. Run the three stages above.
1. Merge into one report:

   ```md
   # /lint-zorch-docs report

   ## Mechanical (tools/lint_docs.py)
   ...

   ## Universal (from /workflow:lint-docs)
   ...

   ## zorch-specific judgment
   ### High / Medium / Low
   - J<n> — <finding>. Suggested fix: <action>.

   ## Summary
   <combined counts; --fix pointer if applicable>
   ```

1. With `--fix`: defer mechanical fixes to `tools/lint_docs.py`'s own messages
   and to the upstream skill. For a J1 gap, offer to draft the missing section
   from the nearest worked shape — but put the WHY prose up for review, never
   invent it. Prompt per finding via `AskUserQuestion`; the rest report-only.
