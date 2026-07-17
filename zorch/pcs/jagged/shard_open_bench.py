"""Benchmark the stacked jagged PCS (commit + open at a random point) on a
real SP1 shard dump.

Reads a GPU-trace dump's per-chip ``.meta``/``.bin`` matrices, packs them
column-major into the stacked dense buffer the commit expects, builds one
committed ``StackedRound`` (RS-encode + SMCS Merkle commit), then opens
(``stacked_basefold_open``, jit'd) at a random EF point. Reports cold
(compile) + warm wall-time for commit and open separately.

The open's work (one batched FRI over the codeword: ``log_s`` fold rounds,
``num_queries`` openings, grind) is independent of the point/eval *values*,
so a random ``z_final``/``dense_eval`` give representative timing.

Run:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_COMPILATION_CACHE_DIR=/tmp/jaxcc_polybench \
    /tmp/sponge-venv/bin/python zorch/pcs/jagged/shard_open_bench.py \
        --shard-dir=/data/sp1_dumps/rsp_21740136_sp1/shard0
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import frx
import frx.numpy as jnp
import numpy as np
from frx import Array
from zk_dtypes import koalabear_mont as BF
from zk_dtypes import koalabearx4_mont as EF
from zk_dtypes import pfinfo

from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.commit.testing.sp1_koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.pcs.jagged.open import (
    StackedOpenProof,
    StackedRound,
    stacked_basefold_open,
)
from zorch.transcript import DuplexTranscript, GrindingTranscript

MODULUS = pfinfo(BF).modulus
_DEFAULT_SHARD = "/data/sp1_dumps/rsp_21740136_sp1/shard0"
# SP1 koalabear16 BaseFold params (from the gpu_fibonacci fixture meta).
_LOG_S = 21
_BLOWUP = 4
_NUM_QUERIES = 52
_POW_BITS = 0


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _load_main_traces(trace_dir: Path) -> list[Array]:
    chips = []
    for meta_path in sorted(trace_dir.glob("*.meta")):
        meta = dict(
            line.split("=") for line in meta_path.read_text().strip().split("\n")
        )
        height, width = int(meta["num_real"]), int(meta["width"])
        bin_path = trace_dir / f"{meta_path.stem}.bin"
        if height == 0 or width == 0 or not bin_path.exists():
            continue
        raw = np.fromfile(bin_path, dtype=np.uint32)
        chips.append(jnp.array(raw.reshape(height, width)).view(BF))
    return chips


def _pack_dense(chips: list[Array], S: int) -> Array:
    """Column-major concat of every chip's columns, end-padded to a multiple of
    ``S`` (mirrors trace_commit's stacked packing the commit reshapes)."""
    flat = jnp.concatenate([c.T.reshape(-1) for c in chips])
    pad = (-flat.shape[0]) % S
    if pad:
        flat = jnp.concatenate([flat, jnp.zeros(pad, dtype=BF)])
    return flat


def _rand_ef(key: Array, n: int) -> Array:
    return frx.random.randint(key, (n * 4,), 0, MODULUS, dtype=jnp.uint32).view(EF)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=_DEFAULT_SHARD)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--log-stacking-height", type=int, default=_LOG_S)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    S = 1 << args.log_stacking_height

    print(f"loading traces from {args.shard_dir} ...", flush=True)
    chips = _load_main_traces(Path(args.shard_dir) / "gpu_traces")
    n_cols = sum(int(c.shape[1]) for c in chips)
    total = sum(int(c.shape[0]) * int(c.shape[1]) for c in chips)
    print(
        f"  {len(chips)} chips, {n_cols} columns, {total} trace elements "
        f"(~2^{total.bit_length()-1})",
        flush=True,
    )

    smcs = _smcs()
    code = BitReversedReedSolomon(message_len=S, blowup=_BLOWUP, dtype=BF)
    open_jit = frx.jit(
        stacked_basefold_open,
        static_argnums=(0, 1, 5),
        static_argnames=("num_queries", "pow_bits"),
    )

    dense = _pack_dense(chips, S)
    k = dense.shape[0] // S
    dmat = dense.reshape(k, S)  # [K, S] — code rides the leading axis
    frx.block_until_ready(dmat)
    print(
        f"  packed dense {dense.shape} -> stacked [{S}, {k}] "
        f"(codeword [{S * _BLOWUP}, {k}])",
        flush=True,
    )

    encode = frx.jit(code.encode)

    # Commit as ONE jit (encode + transpose + leaf hash + tree compress).
    # NOTE: the XLA sponge_hash *supports* a column-major leaf
    # layout (leaf_stride=1) so the ``.T`` could stay a view, but zorch's
    # smcs.commit currently passes row-major strides (leaf_stride=absorb_len),
    # so XLA materializes the ``.T`` (a wrapped_transpose kernel) and the hash
    # reads row-major. Wiring column-major through smcs would drop the transpose
    # + coalesce the 188-wide leaf reads — follow-up, not done here.
    @frx.jit
    def commit_round(d: Array) -> tuple[Array, list[Array]]:
        codeword = code.encode(d).T  # [S*blowup, K]
        _root, digest_layers = smcs.commit(codeword)
        return codeword, digest_layers

    key = frx.random.PRNGKey(args.seed)

    def _time(label: str, fn: Callable[[], Any], reps: int = 2) -> tuple[Any, float]:
        """Compile-warm then time ``reps`` steady-state calls, freeing each
        output before the next call so peak memory stays one-output (these
        codewords are GBs; holding two OOMs a 32 GB card)."""
        w = fn()
        frx.block_until_ready(frx.tree_util.tree_leaves(w))
        del w
        out, dts = None, []
        for _ in range(reps):
            out = None  # drop the prior timed output before reallocating
            t0 = time.perf_counter()
            out = fn()
            frx.block_until_ready(frx.tree_util.tree_leaves(out))
            dts.append(time.perf_counter() - t0)
        dt = min(dts)
        print(f"  {label:14s} {dt*1e3:9.2f} ms  (min of {reps})", flush=True)
        return out, dt

    # RS-encode alone (transient, freed) first, then the fused commit (which
    # holds the GB codeword) so the two 6 GB encode outputs never co-reside.
    # merkle ~= commit - encode (the leaf hash + tree compress).
    _enc, t_encode = _time("encode (RS NTT)", lambda: encode(dmat))
    del _enc
    (cw_dl), t_commit = _time("commit (fused)", lambda: commit_round(dmat))
    _codeword, digest_layers = cw_dl
    del _codeword, cw_dl  # free the GB codeword now; the open re-encodes it
    t_merkle = t_commit - t_encode
    rd = StackedRound(block=dmat, digest_layers=digest_layers)

    # OPEN at a random point (jit'd; one batched FRI).
    key, kr, kd = frx.random.split(key, 3)
    z_final = _rand_ef(kr, args.log_stacking_height)
    dense_eval = _rand_ef(kd, 1)[0]
    frx.block_until_ready((z_final, dense_eval))

    def _open() -> tuple[StackedOpenProof, GrindingTranscript]:
        return open_jit(
            smcs,
            code,
            [rd],
            z_final,
            dense_eval,
            args.log_stacking_height,
            num_queries=_NUM_QUERIES,
            pow_bits=_POW_BITS,
            transcript=DuplexTranscript.new(Poseidon2(koalabear16_params()), rate=8),
        )

    _proof, t_open = _time("open", _open)
    print(
        f"\n[warm] commit={t_commit*1e3:.1f} ms (encode={t_encode*1e3:.1f} + "
        f"merkle~={t_merkle*1e3:.1f})  open={t_open*1e3:.1f} ms  "
        f"total={(t_commit+t_open)*1e3:.1f} ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
