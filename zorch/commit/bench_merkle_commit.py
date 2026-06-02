"""GPU benchmark — merkle-commit (Sponge + Compression + MerkleTree).

A ``zkbench`` JaxBenchmark over ``MerkleTree.commit`` for a sweep of power-of-two
heights. Pre-#25-fusion baseline: each layer is its own ``vmap``, so commit
lowers to a host-orchestrated chain of log2(rows) kernels rather than one device
kernel.

GPU only — the blocks segfault under ``@jit`` on the ZKX CPU backend, and CI
runs CPU, so this is a ``py_binary`` (manual ``bazel run``), not a ``py_test``:

    bazel run //zorch/commit:bench_merkle_commit -- --degrees 16 18 20 22
"""

import argparse
from collections.abc import Iterable

import jax
import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from zorch.commit.testing.koalabear16 import koalabear16_merkle


class MerkleCommitBenchmark(JaxBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="zorch",
            version="0.1.0",
            default_iterations=20,
            default_warmup=5,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--degrees",
            type=int,
            nargs="+",
            default=[16, 18, 20, 22],
            help="log2(rows) heights",
        )
        parser.add_argument("--cols", type=int, default=16)
        parser.add_argument("--rate", type=int, default=8)
        parser.add_argument("--out", type=int, default=8)
        parser.add_argument("--arity", type=int, default=2)
        parser.add_argument("--chunk", type=int, default=8)

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        _, _, tree = koalabear16_merkle(
            rate=args.rate, out=args.out, arity=args.arity, chunk=args.chunk
        )
        commit = jax.jit(tree.commit)
        for d in args.degrees:
            rows = 1 << d
            matrix = jnp.arange(rows * args.cols, dtype=F).reshape(rows, args.cols)
            yield BenchmarkOp(
                name="merkle_commit",
                fn=lambda c=commit, m=matrix: c(m),
                metadata={"degree": str(d), "field": "koalabear"},
                throughput_unit="rows/s",
                throughput_count=rows,
            )


def main() -> int:
    return MerkleCommitBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
