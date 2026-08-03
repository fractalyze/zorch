# Blocks: what to import, from where

Verified against **pyzorch 0.1.2**. Each block's design doc (the WHY, the
seams, the gotchas) is linked from the
[docs hub](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/README.md).

## Core spine — rounds and chains

```python
from zorch.round import Round, Stage, Bridge, ProverRound, VerifierRound, ProveChain, VerifyChain
from zorch.prove import fold_rounds     # scan driver for homogeneous folding rounds
from zorch.verify import verify         # verifier dual of the fold_rounds scan
```

- `ProverRound` / `VerifierRound` are `Protocol`s with positional-only
  `__call__` — implement the callable, don't subclass.
- `ProveChain(rounds)` threads `(carry, transcript)` through prover rounds and
  collects messages; `VerifyChain` replays them and ANDs each `ok`. Chains are
  themselves `Round`s, so they nest.
- `Stage` and `Bridge` are semantic markers over `Round` (no added behavior):
  a top-level phase vs a transcript-only connective.
- `VerifyChain` raises `ValueError` on a message-count mismatch (malformed
  proof shape); algebraic failure comes back as the `ok` array instead.
Doc: [stage-composition.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/composition/stage-composition.md).

## Fiat-Shamir transcript

```python
from zorch.transcript import Transcript, DuplexTranscript, sample_challenge, GrindError
```

`DuplexTranscript` is a device-side duplex sponge — a pytree threaded
functionally under `@jit`; `observe`/`sample` are device ops. Byte-oriented
host transcripts for matching byte-level references: `zorch.byte_transcript`,
`zorch.sha256_field_transcript`.
Doc: [transcript.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/transcript.md).

## Polynomials

```python
from zorch.poly.multilinear import eval_mle, mle_fold, mle_coeffs_to_evals, mle_evals_to_coeffs
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.univariate import eval_coeffs
```

Leaf numeric helpers — element-wise field ops that fuse into the caller's
kernel; no `@jit` of their own.
Doc: [poly.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/poly.md).

## Hashing

```python
from zorch.hash.permutation import Permutation          # the seam (Protocol)
from zorch.hash.poseidon2.poseidon2 import Poseidon2    # a Permutation instance
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.hash.compression import Compression, CompressionParams
```

Your project supplies the pinned params (width, field, constants) for *its*
permutation; sponge/compression code is shared.
Doc: [hash.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/hash.md).

## Merkle commitment

```python
from zorch.commit.merkle import MerkleTree, Opening
```

Binary tree over Sponge (leaves) + Compression (nodes). Strided and
sparse-multi-column variants: `zorch.commit.strided_merkle`, `zorch.commit.smcs`.
Doc: [commit.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/commit.md).

## Linear codes

```python
from zorch.coding.linear_code import LinearCode          # the seam (Protocol)
from zorch.coding.reed_solomon import ReedSolomon, BitReversedReedSolomon, fri_fold, fri_fold_k
```

RS encode on the native NTT; FRI folds live here (they are code operations,
not PCS-private).
Doc: [coding.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/coding.md).

## PCS — polynomial commitment

```python
from zorch.pcs.protocol import PcsProver, PcsVerifier    # the seam (Protocols)
```

Instances under `zorch.pcs.*`: `fri`, `basefold`, `kzg`, `whir`, `ligero`,
`ligerito`, `ipa`, plus the `jagged` overlay — each a package with
`config` / `prover` / `verifier` modules.
Doc: [pcs.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/pcs.md),
[jagged.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/jagged.md).

## Sumcheck

```python
from zorch.sumcheck.prover import StandardRound, CompressedProductRound, ProductSummand, RoundMsg
from zorch.sumcheck.verifier import SumcheckRound, CoeffsSumcheckRound, UnivariateSkipRound
```

Drive the prover rounds with `zorch.prove.fold_rounds` and replay with
`zorch.verify.verify`. Zerocheck/lincheck/LogUp-GKR configure this block
rather than reimplement it. The call shape, end to end (product of two MLEs
over `n_vars` variables, evaluation tables `f`, `h` of length `2**n_vars`):

```python
claim = fnp.sum(f * h)
state = fnp.stack([f, h])                        # stacked (m, N) factor table
prover_round = StandardRound(ProductSummand(degree=2))   # degree = #factors
state, tp, msgs = fold_rounds(prover_round, state, cheap_transcript(F), n_vars)
proof = fnp.stack(msgs)                          # 2-D: one row per round

point, final_claim, tv, ok = verify(SumcheckRound(degree=2), claim, proof, cheap_transcript(F))
# `ok` covers the round consistency; the caller still owes the terminal
# oracle check: final_claim == f(point) * h(point) — here the prover's fully
# folded state, state[0, 0] * state[1, 0].
```

`verify` raises `ValueError` if the proof is not 2-D one-row-per-round
(malformed shape); algebraic failure comes back in `ok`.
Doc: [sumcheck.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/sumcheck.md).

## LogUp-GKR

```python
from zorch.logup_gkr.prover import LogupSummand, GkrLayerRound, LayerProof
from zorch.logup_gkr.verifier import GkrLayerRound as GkrLayerVerifierRound
```

Fractional-sum GKR circuit (`zorch.logup_gkr.circuit`) reducing a public
output claim layer by layer to an input claim your PCS opening discharges.
Doc: [logup-gkr.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/blocks/logup-gkr.md).

## Worked full SNARK — importable

```python
from zorch.spartan.spartan import prove, verify, SpartanProof
from zorch.spartan.r1cs import R1CS
```

Spartan over R1CS: zerocheck + lincheck stages, an RLC bridge, PCS-open glue —
the reference for what consumer-style assembly looks like.
Doc: [spartan.md](https://github.com/fractalyze/zorch/blob/v0.1.2/docs/schemes/spartan.md).

## Test kit (for your tests, not production)

```python
from zorch.testkit.transcript import cheap_transcript   # real DuplexTranscript, cheap permutation
from zorch.testkit.random_field import rand_field, rand_ext_field
```

`cheap_transcript(F)` (positional field dtype) gives protocol-correct
Fiat-Shamir without Poseidon2 cost; `rand_field(seed=, shape=, dtype=)` builds
valid random field arrays without touching raw ints.
