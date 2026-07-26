# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Scheme-agnostic jagged-sumcheck plumbing.

The reusable machinery for a fixed-width, size-invariant jagged sumcheck round
engine, factored out of the LogUp-GKR prover so any jagged sumcheck consumer
(e.g. a jagged zerocheck) can share it:

- `buffers` — fixed-width cap buffers + the donated layer-entry pool.
- `fs` — the per-round Fiat-Shamir hop (observe/squeeze/fold).
- `layout` — the numpy segment-gather / even-prepad recurrences.
- `schedule` — the `row_counts` -> gather/live/re-pad round schedule.
- `types` — `RoundWidthCaps`, the interpolation constants, the round schedule.
- `rounds` — the generic round-coefficient helpers (bind, Gruen coeffs).

The LogUp-specific combine (the four n0/n1/d0/d1 planes + `lam` + `LogupSummand`)
stays in `zorch.logup_gkr`, which imports this package one-way.
"""
