# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file only carries Claude-Code-specific environment setup that has no other
home.

- **Project overview & quick start:** [`README.md`](README.md)
- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md)
- **Detailed design & open decisions:** tracked on GitHub — milestone
  `spine: core + poseidon2 v1`, epic issue
  [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Two non-negotiables

- **Proving-scheme- and zkVM-agnostic.** No block may assume a particular
  proving scheme or any zkVM. If scheme/zkVM-specific knowledge is creeping in,
  it belongs in the consumer (e.g. `whir-zorch`), not in `zorch`.
- **Fusion is a correctness-of-design property.** A `Round`, an
  `absorb`/`squeeze`, a `commit`/`open`, a fold step, and a hash permutation
  must each lower to one fused kernel — by construction, never by a
  per-primitive compiler pattern-match. The findings and the open
  fusion-direction decision live on the epic issue
  [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Development environment

`zorch` is pure Python on JAX + the ZKX PJRT plugin. Each workspace pins its
own jax checkout + venv under `$DEVENV_ENVS_DIR/<workspace>/` (default
`~/Workspace/envs/<workspace>/`); never point a workspace at another
checkout's editable jax. The venv must carry the **opt** selfbuilt jaxlib —
an assertion-enabled build makes XLA compilation an order of magnitude
slower, which shows up as multi-minute test "hangs":

```sh
. "$HOME/Workspace/envs/<workspace>/.venv/bin/activate"
export PYTHONPATH="$PWD"           # import top-level modules directly
export ZKX_REPO_ROOT="$HOME/Workspace/zkx"   # dev against a local ZKX checkout
```

Without `ZKX_REPO_ROOT`, the pinned `zkx-cuda-pjrt` wheel is used.
