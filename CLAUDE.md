# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file only carries Claude-Code-specific environment setup that has no other
home.

- **Project overview & quick start:** [`README.md`](README.md)
- **Architecture & vocabulary:** [`docs/architecture.md`](docs/architecture.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)

## Two non-negotiables

- **Proving-scheme- and zkVM-agnostic.** No block may assume a particular
  proving scheme or any zkVM. If scheme/zkVM-specific knowledge is creeping in,
  it belongs in the consumer (e.g. `whir-zorch`), not in `zorch`.
- **Fusion is a correctness-of-design property.** A `Round`, an
  `absorb`/`squeeze`, a `commit`/`open`, a fold step, and a hash permutation
  must each lower to one fused kernel — by construction, never by a
  per-primitive compiler pattern-match. See `docs/architecture.md`.

## Development environment

`zorch` is pure Python on JAX + the ZKX PJRT plugin. The pinned dev venv lives
under `$DEVENV_ENVS_DIR/zorch/.venv` (default `~/Workspace/envs/zorch/.venv`):

```sh
. "$HOME/Workspace/envs/zorch/.venv/bin/activate"
export PYTHONPATH="$PWD"           # import top-level modules directly
export ZKX_REPO_ROOT="$HOME/Workspace/zkx"   # dev against a local ZKX checkout
```

Without `ZKX_REPO_ROOT`, the pinned `zkx-cuda-pjrt` wheel is used.
