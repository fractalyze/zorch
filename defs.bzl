"""Shared Starlark helpers for zorch BUILD files."""

load("@zorch_pip//:requirements.bzl", "requirement")

# GPU runtime plugins (jax-cuda12 PJRT + plugin). Carried by every
# jax-using py_test — CI's GPU leg (JAX_PLATFORMS=cuda) initializes the device;
# the CPU leg never initializes the plugin, so they are inert there — and by the
# GPU bench.
# Ungated (no select): two self-contained wheels already in the build graph via
# the bench, so the CPU leg pays no extra download. Revisit if a heavier CUDA
# wheel set ever lands in the lock.
GPU_PLUGIN_DEPS = [
    requirement("jax_cuda12_plugin"),
    requirement("jax_cuda12_pjrt"),
]
