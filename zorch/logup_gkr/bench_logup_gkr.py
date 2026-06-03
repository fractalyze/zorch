# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""GPU benchmark — dense LogUp-GKR fused prove.

A ``zkbench`` JaxBenchmark over the whole dense LogUp-GKR prover
(``zorch.logup_gkr.testing.prove_gkr_jitted``): fold the fractional-sum pyramid
from a random first layer (trace gen) and run the per-layer sumcheck chain down
to the interaction floor (proof gen) — the whole prove traced into one ``@jit``
program. Sized by ``(--interaction-variables, --row-variables)``: the first
layer is ``2**(iv + rv)`` wide per MLE and the chain proves ``rv`` layers.

Fusion is by-construction: the whole prove is one program, so the launch-bound
per-op dispatch wall collapses to a single fused compilation. The op carries a
``lower`` thunk, so zkbench's phase-aware measurement tracks the full
compile / runtime × time / memory grid for the fused prove over time:

- runtime: ``latency`` + ``memory`` (device-memory peak) — the warm prove.
- compile: ``compile_time`` + ``compile_memory`` — from ``lowered.compile()``.

The device peak is process-cumulative and the compile number needs a cold
``JAX_COMPILATION_CACHE_DIR``, so for clean per-phase numbers run the two phases
as separate invocations (one ``--row-variables`` per process):

    bazel run //zorch/logup_gkr:bench_logup_gkr -- --phase runtime --row-variables 16
    bazel run //zorch/logup_gkr:bench_logup_gkr -- --phase compile --row-variables 16

``output_hash`` is zorch's own deterministic prove output — a self-anchored,
scheme-agnostic regression guard (not a cross-impl golden).

A benchmark, not a correctness gate, so it is a manual ``bazel run``
(``py_binary``), run on GPU for representative numbers — it runs on the ZKX CPU
backend too, just slowly.
"""

import argparse
import functools
from collections.abc import Iterable

import jax
from jax import Array
from zk_dtypes import koalabear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark, compute_array_hash

from zorch.logup_gkr.testing import prove_gkr_jitted
from zorch.testkit.random_field import rand_field

_SEED = 0


def _first_layer_mles(iv: int, rv: int) -> tuple[Array, ...]:
    """The four random dense first-layer MLEs (n0, n1, d0, d1), ``2**(iv + rv)``
    wide, 4 distinct seeds so they don't alias.

    The Montgomery field (``koalabear_mont``, matching ``bench_merkle_commit``)
    is used rather than the canonical ``koalabear`` of the test fixtures."""
    width = 1 << (iv + rv)
    return tuple(rand_field(_SEED + i, (width,), F) for i in range(4))


def _num_challenges(iv: int, rv: int) -> int:
    """Upper bound on Fiat-Shamir draws: ``iv + 1`` for the output point, then per
    proved layer at most ``lam + (iv + rv) sumcheck rounds + 1`` reduction.
    StubTranscript only reads the prefix it needs, so an over-estimate is free."""
    return (iv + 1) + rv * (iv + rv + 2)


class LogupGkrBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="zorch",
            version="0.1.0",
            default_iterations=20,
            default_warmup=5,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--row-variables",
            type=int,
            nargs="+",
            default=[12, 14, 16],
            help="log2 rows folded by the GKR pyramid (one layer proved each)",
        )
        parser.add_argument(
            "--interaction-variables",
            type=int,
            default=4,
            help="log2 interactions at the floor",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        iv = args.interaction_variables
        for rv in args.row_variables:
            mles = _first_layer_mles(iv, rv)
            challenges = rand_field(_SEED + 99, (_num_challenges(iv, rv),), F)
            args_ = (*mles, challenges, iv)
            yield BenchmarkOp(
                name="logup_gkr_prove",
                fn=functools.partial(prove_gkr_jitted, *args_),
                lower=functools.partial(prove_gkr_jitted.lower, *args_),
                metadata={
                    "degree": str(iv + rv),
                    "interaction_variables": str(iv),
                    "row_variables": str(rv),
                    "field": "koalabear",
                },
                output_hash_fn=lambda a=args_: compute_array_hash(
                    jax.block_until_ready(prove_gkr_jitted(*a))
                ),
                throughput_unit="evals/s",
                throughput_count=1 << (iv + rv),
            )


def main() -> int:
    return LogupGkrBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
