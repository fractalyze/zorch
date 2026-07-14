# Development environment

`zorch` is pure Python on JAX + the Fractal XLA PJRT plugin. This page is the dev-loop
reference; for a first install see the quick start in
[`../README.md`](../../README.md).

## Per-workspace venvs

Each workspace pins its own frx checkout + venv under
`$DEVENV_ENVS_DIR/<workspace>/` (default `~/Workspace/envs/<workspace>/`);
never point a workspace at another checkout's editable frx. The venv must
carry the **opt** selfbuilt frxlib — an assertion-enabled build makes XLA
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

## Developing against a local Fractal XLA build

The pinned venv installs the four `jax` / `frxlib` / `frx-cuda12-pjrt` /
`frx-cuda12-plugin` wheels from the Fractalyze package index (rebuilt and
published per merged Fractal XLA PR):

```sh
pip install -r requirements.in \
  --extra-index-url https://fractalyze.github.io/pypi/simple/
```

Plugin resolution has **no env-var indirection**: `jax_plugins/xla_cuda12`
loads the `xla_cuda_plugin.so` installed in the venv's site-packages (there is
no `*_GPU_PLUGIN_PATH` override). So to run against a local Fractal XLA build
with unmerged changes, build the `frx-cuda12-plugin` + `frx-cuda12-pjrt` wheels
from the checkout (see the XLA repo's `docs/build_from_source.md`) and
force-reinstall them over the pinned ones:

```sh
pip install --force-reinstall --no-deps \
  <dist>/jax_cuda12_plugin-*.whl <dist>/jax_cuda12_pjrt-*.whl
```

Once the change merges and the index publishes a fresh dev build, bump the four
pins in `requirements.in` to that version instead.

A stale plugin surfaces as `custom op 'stablehlo.composite' is unknown` on any
fresh compile.

## JAX compile-cache rule

A persistent `JAX_COMPILATION_CACHE_DIR` skips recompiles of the heavy
`local_only` tests across venv runs — worthwhile whenever you iterate against
one frxlib for a while. Keep **one cache directory per frxlib build** and
treat a rebuilt wheel as a new toolchain: self-built wheels share a version
string, so a shared directory replays the *other* build's executables
([#120](https://github.com/fractalyze/zorch/issues/120)).

`bazel test` strips the variable (hermetic), so caching only applies to the
venv loop.
