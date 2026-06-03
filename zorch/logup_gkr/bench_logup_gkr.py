# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""GPU benchmark — dense LogUp-GKR prove.

A ``zkbench`` JaxBenchmark over the whole dense LogUp-GKR prover
(``zorch.logup_gkr.testing.prove_gkr``): fold the fractional-sum pyramid from a
random first layer (trace gen), then run the per-layer sumcheck chain down to the
interaction floor (proof gen). Sized by ``(--interaction-variables,
--row-variables)``: the first layer is ``2**(iv + rv)`` wide per MLE and the
chain proves ``rv`` layers.

The prover is eager — the pyramid is folded and proved layer by layer
(``zorch.logup_gkr.prover``), not one program at scale, so this measures the
launch-bound eager prove the future CUDA-graph capture has to beat.

A benchmark, not a correctness gate, so it is a manual ``bazel run``
(``py_binary``), run on GPU for representative numbers — it runs on the ZKX CPU
backend too, just slowly:

    bazel run //zorch/logup_gkr:bench_logup_gkr -- --row-variables 12 14 16
"""

import argparse
from collections.abc import Iterable

from jax import Array
from zk_dtypes import koalabear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from zorch.logup_gkr.circuit import GkrLayer
from zorch.logup_gkr.testing import prove_gkr
from zorch.testkit.random_field import rand_field

_SEED = 0


def _first_layer(iv: int, rv: int) -> GkrLayer:
    """A random dense first layer, ``2**(iv + rv)`` wide per MLE (4 distinct
    seeds so the numerators/denominators don't alias).

    Built here rather than reused from ``testing.random_first_layer`` because the
    benchmark measures the Montgomery field (``koalabear_mont``, matching
    ``bench_merkle_commit``); the test fixture is canonical ``koalabear``."""
    width = 1 << (iv + rv)
    return GkrLayer(
        numerator_0=rand_field(_SEED, (width,), F),
        numerator_1=rand_field(_SEED + 1, (width,), F),
        denominator_0=rand_field(_SEED + 2, (width,), F),
        denominator_1=rand_field(_SEED + 3, (width,), F),
        num_interaction_variables=iv,
    )


def _num_challenges(iv: int, rv: int) -> int:
    """Upper bound on Fiat-Shamir draws: ``iv + 1`` for the output point, then per
    proved layer at most ``lam + (iv + rv) sumcheck rounds + 1`` reduction.
    StubTranscript only reads the prefix it needs, so an over-estimate is free."""
    return (iv + 1) + rv * (iv + rv + 2)


def _prove(first: GkrLayer, challenges: Array) -> Array:
    """Run the full dense prove; return the last-proved layer's round polynomials
    — the end of the sequential carry, so waiting on it waits on the whole prove."""
    _, _, proofs, _ = prove_gkr(first, challenges)
    return proofs[-1].round_polys


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
            first = _first_layer(iv, rv)
            challenges = rand_field(_SEED + 99, (_num_challenges(iv, rv),), F)
            yield BenchmarkOp(
                name="logup_gkr_prove",
                fn=lambda f=first, c=challenges: _prove(f, c),
                metadata={
                    "degree": str(iv + rv),
                    "interaction_variables": str(iv),
                    "row_variables": str(rv),
                    "field": "koalabear",
                },
                throughput_unit="evals/s",
                throughput_count=1 << (iv + rv),
            )


def main() -> int:
    return LogupGkrBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
