"""Shared Starlark helpers for zorch BUILD files."""

load("@zorch_pip//:requirements.bzl", "requirement")

# GPU runtime plugins (frx-cuda12 PJRT + plugin). Carried by every py_test that
# initializes a device: CI's GPU leg pins FRX_PLATFORMS=cuda with no CPU
# fallback, so a test that reaches a backend without these dies at init rather
# than falling back. The CPU leg (FRX_PLATFORMS=cpu) never initializes the
# plugin, so they are inert there — which is why an omission stays invisible
# until the GPU leg runs. The GPU leg's own cuda,cpu pass is not that case: it
# initializes both backends and needs these like any other cuda pass.
#
# Depending on frx is NOT the predicate; touching a device is. A few tests
# depend on frx and never allocate, and they do not carry these.
#
# Ungated (no select): the wheels are already in the build graph via ~90 sibling
# py_tests, so one more consumer pays no extra download, and runfiles link
# rather than copy them. Gating would need a build flag to select on — the
# backend is chosen by a test-time env var, which Bazel configuration cannot
# see — so it is a repo-wide change, not a per-target one.
GPU_PLUGIN_DEPS = [
    requirement("frx_cuda12_plugin"),
    requirement("frx_cuda12_pjrt"),
]
