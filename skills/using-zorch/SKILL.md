---
name: using-zorch
description: >
  Build proof systems with zorch — FRX-native (JAX-fork) building blocks for
  SNARKs: rounds, stages, Fiat-Shamir transcripts, polynomials, hashing, Merkle
  commitment, codes, PCS, sumcheck, LogUp-GKR. Use when writing code that
  imports zorch, assembling or porting a prover on zorch blocks, installing
  pyzorch/frx, or debugging finite-field dtype and FRX toolchain errors in a
  project that consumes zorch.
---

# Using zorch

`zorch` provides reusable, proving-scheme-agnostic building blocks a SNARK is
assembled from. It runs on **FRX**, Fractalyze's fork of JAX, lowered through
Fractalyze XLA with native finite-field dtypes. Install as **`pyzorch`**,
import as **`zorch`** (the `zorch` name on PyPI is an unrelated project).

This guide is verified against **pyzorch 0.2.0** — the imports and links below
pin that release. `main` may be ahead of what pip installs.

## Install

Python **3.11 only**, Linux x86_64 or macOS Apple Silicon:

```sh
pip install pyzorch                    # CPU tier
pip install pyzorch 'frx[cuda12]' \
    --extra-index-url https://fractalyze.github.io/pypi/simple/   # GPU (CUDA 12)
python -c "import frx, zorch; print(frx.devices()); print(zorch.__version__)"
```

Install problems, platform limits, and error symptoms:
[references/setup.md](references/setup.md).

## The two units

A **round** is one step of a repeated recurrence; a **stage** is one claim
reduction with separately deployable prover/verifier roles:

```text
ProverRound:    (carry, transcript)          -> (carry, transcript, message)
VerifierRound:  (carry, transcript, message) -> (carry, transcript, ok)
ProverStage.prove(claim, witness, transcript)     -> ProveResult
VerifierStage.verify(claim, reduction_proof, ...) -> VerifyResult
```

Both roles derive the same **reduced claim**; the reduction proof establishes
the source claim *conditional on* it. Stages chain — each stage's reduced
claim is the next one's source claim — until `TrivialClaim`, which holds by
construction. Which blocks exist and where to import them from:
[references/blocks.md](references/blocks.md).

## Rules that prevent silent breakage

1. **Never hand-roll modular arithmetic.** Compute in the finite-field dtypes
   (`zk_dtypes`); no `pow(x, e, p)`, no `x % MODULUS`, no hardcoded prime
   literals. Field dtypes also reject a few array ops that work on ints (bit
   shifts, power by a traced exponent array) — the exact errors and
   workarounds are in
   [references/field-dtypes.md](references/field-dtypes.md).
2. **Respect the consumer boundary.** Your project owns its protocol schedule,
   claim layout, transcript framing, and serialization; zorch owns anything a
   second, unrelated proof system would reuse unchanged. Don't fork a zorch
   block to specialize it — inject parameters through its seam
   ([references/building-on-zorch.md](references/building-on-zorch.md)).
3. **The transcript is explicit, and challenges are carry, not message.** Pass
   the transcript in and out of every round; never hide it in mutable state.
   Anything both roles can derive from the transcript belongs in the carry —
   only prover→verifier data is a message.
4. **Fusion is a design property, not an optimizer favor.** zorch's contract:
   a `Round`, an `absorb`/`squeeze`, a `commit`/`open`, a fold step, and a
   hash permutation each lower to **one replayable device unit by
   construction** — never by hoping a compiler pattern-match recovers it. In
   your round bodies that means element-wise field ops plus the one inherent
   `Σ`, no gratuitous `reduce`/`gather`, no host round-trips. The `@jit`
   discipline, loop-tool choice, and how to verify:
   [references/frx-and-fusion.md](references/frx-and-fusion.md).

## References

| Read | When |
| --- | --- |
| [references/setup.md](references/setup.md) | installing, wrong-platform/toolchain errors |
| [references/blocks.md](references/blocks.md) | finding the right block and its import path |
| [references/field-dtypes.md](references/field-dtypes.md) | writing field arithmetic; MLIR/dtype crashes |
| [references/frx-and-fusion.md](references/frx-and-fusion.md) | writing FRX/JAX code that fuses; @jit and loop-tool rules; slow-jit debugging |
| [references/building-on-zorch.md](references/building-on-zorch.md) | assembling a prover; what goes in your repo vs upstream |

Design docs (the WHY behind each block):
[docs hub](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/README.md) ·
worked full SNARK, shipped as importable code:
[`zorch.spartan`](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/schemes/spartan.md).
