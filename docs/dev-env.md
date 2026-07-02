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

## Running the test suite

The local loop is pytest (`pip install -r requirements-dev.in` once per venv).
Run it parallel by default — the suites are dominated by per-test fixed cost
(jit tracing + XLA compile), not data size, so workers scale them down to the
slowest single test:

```sh
pytest -n <physical cores>     # e.g. -n 16 on a 32-thread box
```

Drop `-n` only when chasing one test's output interactively. `bazel test //...`
remains the single source of truth for "all tests pass"
([`conventions.md`](conventions.md)); it parallelizes per target on its own.

## Developing against a local ZKX checkout

Plugin resolution order is `ZKX_GPU_PLUGIN_PATH` > the wheel's bundled `.so` >
`$ZKX_REPO_ROOT/bazel-bin` — so when the venv has `jax-cuda12-plugin` installed
(every pinned venv does), `ZKX_REPO_ROOT` is silently ignored and only
`ZKX_GPU_PLUGIN_PATH` reaches a local zkx build:

```sh
export ZKX_GPU_PLUGIN_PATH="$HOME/Workspace/envs/<workspace>/zkx/bazel-bin/zkx/pjrt/c/pjrt_c_api_gpu_plugin.so"
```

A stale wheel surfaces as `custom op 'stablehlo.composite' is unknown` on any
fresh compile.

## JAX compile-cache rule

A persistent `JAX_COMPILATION_CACHE_DIR` skips recompiles of the heavy
`local_only` tests across venv runs — worthwhile whenever you iterate against
one jaxlib for a while. Keep **one cache directory per jaxlib build** and
treat a rebuilt wheel as a new toolchain: self-built wheels share a version
string, so a shared directory replays the *other* build's executables
([#120](https://github.com/fractalyze/zorch/issues/120)).

`bazel test` strips the variable (hermetic), so caching only applies to the
venv loop.

## Exported round-binary cache (`ZORCH_EXPORT_CACHE_DIR`)

The jagged LogUp-GKR round loop dispatches shape-polymorphic `jax.export`
binaries (`zorch/logup_gkr/round_export.py`). Their symbolic StableHLO build
re-runs every process and is NOT covered by the XLA compile cache; setting
`ZORCH_EXPORT_CACHE_DIR` persists the serialized binaries across processes.
The directory is namespaced by jax version plus a hash of the whole `zorch`
source tree, so any source edit invalidates it — never prune it by hand to
"save a rebuild". The two caches compose: `ZORCH_EXPORT_CACHE_DIR` skips the
export build, `JAX_COMPILATION_CACHE_DIR` skips the per-concrete-shape XLA
codegen that `exported.call` re-runs; a deep pyramid needs both to warm across
proves.
