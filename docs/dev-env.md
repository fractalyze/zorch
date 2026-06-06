# Development environment

`zorch` is pure Python on JAX + the ZKX PJRT plugin. This page is the dev-loop
reference; for a first install see the quick start in
[`../README.md`](../README.md).

## Per-workspace venvs

Each workspace pins its own jax checkout + venv under
`$DEVENV_ENVS_DIR/<workspace>/` (default `~/Workspace/envs/<workspace>/`);
never point a workspace at another checkout's editable jax. The venv must
carry the **opt** selfbuilt jaxlib — an assertion-enabled build makes XLA
compilation an order of magnitude slower, which shows up as multi-minute test
"hangs":

```sh
. "$HOME/Workspace/envs/<workspace>/.venv/bin/activate"
export PYTHONPATH="$PWD"           # import top-level modules directly
```

## Developing against a local ZKX checkout

To develop against a **local ZKX checkout** instead of the pinned
`zkx-cuda-pjrt` wheel, point the plugin resolver at it:

```sh
export ZKX_REPO_ROOT="$HOME/Workspace/zkx"
```

Without `ZKX_REPO_ROOT`, the pinned `zkx-cuda-pjrt` wheel is used.

## JAX compile-cache rule

A persistent `JAX_COMPILATION_CACHE_DIR` skips recompiles of the heavy
`local_only` tests across venv runs — worthwhile whenever you iterate against
one jaxlib for a while. Keep **one cache directory per jaxlib build** and
treat a rebuilt wheel as a new toolchain: self-built wheels share a version
string, so a shared directory replays the *other* build's executables
([#120](https://github.com/fractalyze/zorch/issues/120)).

`bazel test` strips the variable (hermetic), so caching only applies to the
venv loop.
