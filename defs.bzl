"""Shared Starlark helpers for zorch BUILD files."""

load("@zorch_pip//:requirements.bzl", "requirement")

# GPU runtime plugins (ZKX PJRT + the CUDA libs it loads). Carried by every
# jax-using py_test — CI's GPU leg (JAX_PLATFORMS=cuda) initializes the device;
# the CPU leg never initializes the plugin, so they are inert there — and by the
# GPU bench.
# Ungated (no select): two self-contained wheels already in the build graph via
# the bench, so the CPU leg pays no extra download. Revisit if a heavier CUDA
# wheel set ever lands in the lock.
GPU_PLUGIN_DEPS = [
    requirement("jax_cuda12_plugin"),
    requirement("zkx_cuda_pjrt"),
]
