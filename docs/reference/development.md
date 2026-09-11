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

### `frx` and `frxlib` move together

The **CPU** runtime — every host FFI handler an XLA PR registers — ships in
`frxlib`, not `frx`. `pip install --upgrade frx` leaves an already-satisfied
`frxlib` alone, so a CPU custom call added upstream still fails with
`NOT_FOUND: No FFI handler registered for <target> on a platform Host` even
though `frx` reports the new version. Upgrade both to the same dev stamp:

```sh
pip install --upgrade --extra-index-url https://fractalyze.github.io/pypi/simple/ \
  frx==0.10.2.devYYYYMMDDHHMMSS frxlib==0.10.2.devYYYYMMDDHHMMSS
```

An XLA merge does not publish a wheel by itself, either: fractalyze/jax has to
bump its pinned XLA revision, its CI has to pass, and only then does the Dev
Release workflow build. Budget on the order of an hour from merge to wheel,
and confirm the pin actually contains the commit rather than the timestamp
merely being newer.

## Benchmarking a backend-dispatched function

Functions that pick a lowering by `frx.default_backend()` — the bit-select
reductions, the NTT paths — do that **inside their own `jit`**, so the chosen
arm is baked into the traced jaxpr, not re-decided per call. Two consequences
for anyone measuring or forcing an arm:

- Patching `default_backend` does nothing on its own. A same-shape call from
  anywhere earlier in the process already cached its arm, and a cache hit never
  re-runs the dispatch. Call `<fn>.clear_cache()` before, **and after**, or the
  forced trace leaks into whatever runs next.
- Clearing per repetition to keep two arms honest means every timing includes a
  compile. That constant dominates and reads out as a flat ~1.2-1.9x whatever
  the kernels actually do. Warm each arm once under its patch, then time with no
  patch and no clear — the cache hit is the executable alone. On the bit-select
  handlers the two methods reported 1.2-1.9x and 2.0-8.0x for the same code.

Absolute times are the tell: tens of milliseconds for a reduction that takes
hundreds of microseconds means the compile is in the measurement.

## FRX compile-cache rule

A persistent `JAX_COMPILATION_CACHE_DIR` skips recompiles of the heavy
`local_only` tests across venv runs. Keep **one cache directory per frxlib
build** and treat a rebuilt wheel as a new toolchain: self-built wheels share a
version string, so a shared directory replays the *other* build's executables.
`bazel test` strips the variable, so this applies only to the venv loop.
