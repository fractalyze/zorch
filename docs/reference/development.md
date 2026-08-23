# Development environment

`zorch` is pure Python on FRX + the Fractalyze XLA PJRT plugin. For a first
install see the quick start in [`../README.md`](../../README.md).

## Per-workspace venvs

Each workspace pins its own FRX checkout and venv under
`$DEVENV_ENVS_DIR/<workspace>/` (default `~/Workspace/envs/<workspace>/`); never
point one workspace at another checkout's editable FRX. The venv must carry the
**opt** selfbuilt frxlib — an assertion-enabled build makes XLA compilation an
order of magnitude slower, which surfaces as multi-minute test "hangs".

```sh
. "$HOME/Workspace/envs/<workspace>/.venv/bin/activate"
export PYTHONPATH="$PWD"
```

## Running the test suite

The local loop is pytest (`pip install -r requirements-dev.in` once per venv),
run parallel by default — the suites are dominated by per-test fixed cost (jit
tracing plus XLA compile) rather than data size, so workers scale them down to
the slowest single test. Drop `-n` only when chasing one test interactively.

```sh
pytest -n <physical cores>
```

`bazel test //...` remains the single source of truth for "all tests pass"
([`conventions.md`](conventions.md)); it parallelizes per target on its own.

### A `py_test` that touches an frx backend needs `GPU_PLUGIN_DEPS`

Any `py_test` whose imports *initialize* an frx backend — directly, or
transitively through `zorch.byte_transcript`, `hash_frx`, or a test fixture
that pulls either in — must append `+ GPU_PLUGIN_DEPS` to its `deps`:

```starlark
load("//:defs.bzl", "GPU_PLUGIN_DEPS")

py_test(
    name = "my_test",
    srcs = ["my_test.py"],
    deps = [
        "//zorch:byte_transcript",
        requirement("numpy"),
    ] + GPU_PLUGIN_DEPS,
)
```

Without it the GPU leg dies with `Backend 'cuda' is not in the list of known
backends` **before reaching any assertion**.

Two things make this cost a red run rather than a local failure:

- **`bazel test` locally cannot reproduce it.** The local invocation does not
  set `FRX_PLATFORMS=cuda`, so the target is green on your machine and red on
  CI. Green locally is not evidence here.
- **"The neighbouring target omits it" is not a valid inference.** A sibling
  test may import only pure-numpy libraries (`lattice_frx` alone, say) and
  legitimately need no plugin. Decide from *this* target's own import closure.

## Developing against a local Fractalyze XLA build

The pinned venv installs the `jax` / `frxlib` / `frx-cuda12-pjrt` /
`frx-cuda12-plugin` wheels from the Fractalyze index, republished per merged
Fractalyze XLA PR:

```sh
pip install -r requirements.in \
  --extra-index-url https://fractalyze.github.io/pypi/simple/
```

Plugin resolution has **no env-var indirection**: `jax_plugins/xla_cuda12` loads
the `xla_cuda_plugin.so` in the venv's site-packages, with no `*_GPU_PLUGIN_PATH`
override. To run against a local build with unmerged changes, build the plugin
and pjrt wheels from the checkout (the XLA repo's `docs/build_from_source.md`)
and force-reinstall them over the pinned ones; once the change merges, bump the
pins in `requirements.in` instead.

```sh
pip install --force-reinstall --no-deps \
  <dist>/jax_cuda12_plugin-*.whl <dist>/jax_cuda12_pjrt-*.whl
```

A stale plugin surfaces as `custom op 'stablehlo.composite' is unknown` on any
fresh compile.

## FRX compile-cache rule

A persistent `JAX_COMPILATION_CACHE_DIR` skips recompiles of the heavy
`local_only` tests across venv runs. Keep **one cache directory per frxlib
build** and treat a rebuilt wheel as a new toolchain: self-built wheels share a
version string, so a shared directory replays the *other* build's executables.
`bazel test` strips the variable, so this applies only to the venv loop.
