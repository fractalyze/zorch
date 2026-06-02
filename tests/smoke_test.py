"""Smoke test for the pinned toolchain.

Verifies the bazel + pip plumbing resolves jax and the zkx field dtypes and
that a trivial jit'd computation runs. This is the placeholder target that
keeps `bazel test //...` green until the building-block code lands.
"""

import jax
import jax.numpy as jnp
import zk_dtypes  # noqa: F401  (import-only: confirms the field-dtype wheel resolves)


def test_jax_jit_runs():
    f = jax.jit(lambda x: x.sum())
    assert int(f(jnp.arange(4))) == 6


if __name__ == "__main__":
    test_jax_jit_runs()
    print("ok", jax.__version__)
