# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file is just the map plus the two rules every change must respect.

- **Project overview & quick start:** [`README.md`](README.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Dev environment — venv pinning, Fractal XLA plugin, compile caches:**
  [`docs/reference/development.md`](docs/reference/development.md)
- **Detailed design & open decisions:** tracked on GitHub — epic issue
  [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Two non-negotiables

- **Proving-scheme- and implementation-agnostic.** No block may assume a
  particular proving scheme or any one downstream implementation (a zkVM, zkML,
  or zkTLS prover). If scheme- or implementation-specific knowledge is creeping
  in, it belongs in the consumer, not in `zorch`.
- **Fusion is a correctness-of-design property.** A `Round`, an
  `absorb`/`squeeze`, a `commit`/`open`, a fold step, and a hash permutation
  must each lower to one fused kernel — by construction, never by a
  per-primitive compiler pattern-match. The findings and the open
  fusion-direction decision live on the epic issue
  [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).
