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

``--challenge-field`` selects the Fiat-Shamir field: ``ef`` (default,
``koalabearx4``) is the faithful LogUp-GKR workload — a base-field trace under an
extension-field eval-point/``lam``, the EF register/HBM pressure this benchmark
exists to measure — and needs a ZKX plugin carrying prime-ir#332; ``bf`` is a
cheaper upper-bound baseline. Run one field per invocation for an A/B.

``--transcript`` selects the Fiat-Shamir source: ``stub`` (default) feeds a preset
challenge stream — a floor that elides the sponge — while ``duplex`` runs the real
on-device poseidon2 ``DuplexTranscript`` (base field), the honest full e2e. The
duplex path needs a ZKX plugin that fuses the ``poseidon2:`` composite via the
``Poseidon2Fusion`` emitter; without it the unrolled permute hits the
generic-codegen compile cliff.

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
from zk_dtypes import koalabearx4_mont as EF
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark, compute_array_hash

from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.logup_gkr.testing import prove_gkr_jitted, prove_gkr_jitted_with_transcript
from zorch.testkit.random_field import rand_field
from zorch.transcript import DuplexTranscript

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


def _challenges(field: str, seed: int, n: int) -> Array:
    """The Fiat-Shamir challenge stream that drives the prove.

    The faithful LogUp-GKR workload draws challenges from the *extension* field
    (``koalabearx4``) over a base-field (``koalabear``) trace: the EF eval-point
    and batching ``lam`` are exactly the register/HBM pressure this benchmark
    exists to measure, and EF needs a ZKX plugin carrying prime-ir#332 (the EF
    ``.sum()`` reduction SIGABRTed before it). ``bf`` keeps the whole prove in the
    base field -- a cheaper upper-bound baseline that elides the EF expansion."""
    if field == "ef":
        # An EF element is four BF limbs; rand_field emits only base fields.
        return rand_field(seed, (n, 4), F).view(EF).reshape(n)
    return rand_field(seed, (n,), F)


def _hashable(out: Array) -> Array:
    """``compute_array_hash`` casts to u4 limbs; an EF prove output is flattened
    and viewed as its base-field limbs first (a BF output passes through). The
    ``ravel`` (zero-copy) guarantees a contiguous 1-D array for the dtype view."""
    return out.ravel().view(F) if out.dtype == EF else out


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
        parser.add_argument(
            "--challenge-field",
            choices=["bf", "ef"],
            default="ef",
            help="Fiat-Shamir field: ef (koalabearx4, faithful) or bf (upper bound)",
        )
        parser.add_argument(
            "--transcript",
            choices=["stub", "duplex"],
            default="stub",
            help="stub: preset challenge stream (a floor). duplex: real on-device "
            "poseidon2 Fiat-Shamir, the full e2e (base-field; ignores "
            "--challenge-field). Needs a plugin that fuses the poseidon2 composite.",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        iv = args.interaction_variables
        duplex = args.transcript == "duplex"
        prove = prove_gkr_jitted_with_transcript if duplex else prove_gkr_jitted
        # Built once (a poseidon2 permutation can't be constructed under trace) and
        # passed as a traced pytree arg, so the sponge state is a runtime input, not
        # baked into the executable.
        transcript = (
            DuplexTranscript.new(koalabear16_perm(), rate=8) if duplex else None
        )
        for rv in args.row_variables:
            mles = _first_layer_mles(iv, rv)
            # duplex threads the sponge; stub splices a preset challenge stream.
            # Both sit between the MLEs and the static iv, so fn/.lower are uniform.
            middle = (
                transcript
                if duplex
                else _challenges(
                    args.challenge_field, _SEED + 99, _num_challenges(iv, rv)
                )
            )
            op_args = (*mles, middle, iv)
            yield BenchmarkOp(
                name="logup_gkr_prove",
                fn=functools.partial(prove, *op_args),
                lower=functools.partial(prove.lower, *op_args),
                metadata={
                    "degree": str(iv + rv),
                    "interaction_variables": str(iv),
                    "row_variables": str(rv),
                    "field": "koalabear",
                    "transcript": args.transcript,
                    # duplex Fiat-Shamir is base-field; --challenge-field is moot.
                    "challenge_field": "bf" if duplex else args.challenge_field,
                },
                output_hash_fn=lambda a=op_args: compute_array_hash(
                    _hashable(jax.block_until_ready(prove(*a)))
                ),
                throughput_unit="evals/s",
                throughput_count=1 << (iv + rv),
            )


def main() -> int:
    return LogupGkrBenchmark().run()


if __name__ == "__main__":
    raise SystemExit(main())
