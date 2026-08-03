# Setup: installing pyzorch

## Platform constraints (check these first)

- **Python 3.11 only.** `frxlib` ships cp311 wheels — not 3.12/3.13.
- **Linux x86_64 or macOS Apple Silicon.** No Intel Mac wheels, no Windows.
- Install name is **`pyzorch`**; import name is **`zorch`**. `pip install
  zorch` fetches an unrelated PyPI project — uninstall it if present.

## CPU tier

```sh
pip install pyzorch
```

Pulls `frx` + `frxlib` (Fractalyze's JAX fork) from PyPI. Everything traces and
runs, at CPU speed — sufficient for development and tests.

## GPU tier (CUDA 12)

```sh
pip install pyzorch 'frx[cuda12]' \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The extra index carries the CUDA plugin wheels (`frx-cuda12-pjrt`,
`frx-cuda12-plugin`), which exceed PyPI's per-file size limit. The extra index
is **only** needed for the GPU tier.

## Verify

```sh
python -c "import frx, zorch; print(frx.devices()); print(zorch.__version__)"
```

- `[CpuDevice(id=0)]` — CPU tier active.
- CUDA device list — GPU tier active.

## Symptoms → causes

| Symptom | Cause / fix |
| --- | --- |
| `pip` finds no `frxlib` wheel | Python ≠ 3.11, or unsupported platform (Intel Mac, Windows). Create a 3.11 venv. |
| `import zorch` fails after `pip install zorch` | Wrong package — that's an unrelated project. `pip uninstall zorch && pip install pyzorch`. |
| `An NVIDIA GPU may be present … but a CUDA-enabled jaxlib is not installed. Falling back to cpu.` | `frx[cuda12]` extras missing or the `--extra-index-url` was omitted, so the CUDA plugin wheels never installed. Re-run the GPU command. (The message says "jaxlib" — upstream JAX's name — but it means the frx CUDA plugin.) |
| `CUDA_ERROR_OUT_OF_MEMORY` storm at startup on a shared GPU | frx preallocates ~75% of VRAM by default. `export XLA_PYTHON_CLIENT_PREALLOCATE=false`, and pick an idle device with `CUDA_VISIBLE_DEVICES=<n>`. |
| `custom op 'stablehlo.composite' is unknown` on compile | The installed `frx`/plugin wheels are older than what this zorch release emits. Upgrade both together: `pip install -U pyzorch 'frx[cuda12]' --extra-index-url https://fractalyze.github.io/pypi/simple/`. |
| Tests/compiles are order-of-magnitude slow, look "hung" | An assertion-enabled (debug) toolchain build. Use the released wheels, not a self-built debug `frxlib`. |
| Recompiles repeat across runs | Set a persistent `JAX_COMPILATION_CACHE_DIR` — but keep **one cache dir per toolchain build**; a shared dir replays another build's executables. |

## Keeping versions in lockstep

`pyzorch` pins the `frx` build it was released against. When upgrading, upgrade
`pyzorch` and `frx[cuda12]` in the **same** pip invocation so the resolver moves
them together; mixing a new zorch with an old plugin (or vice versa) is what
produces the `stablehlo.composite` error above. To reproduce the exact
toolchain this guide was verified against, pin the release:
`pip install pyzorch==0.2.0` (plus `'frx[cuda12]'` and the extra index for
GPU).

Contributor-mode setup (editable checkout, self-built toolchain, compile
caches) is different and lives in
[docs/reference/development.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/reference/development.md).
