# Blocks: what to import, from where

Verified against **pyzorch 0.2.0**. Each block's design doc (the WHY, the
seams, the gotchas) is linked from the
[docs hub](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/README.md).

## Core spine — rounds, stages, drivers

```python
from zorch.round import ProverRound, VerifierRound, RunningClaim, prove_rounds, verify_rounds
from zorch.stage import ProverStage, VerifierStage, ProveResult, VerifyResult, TrivialClaim
from zorch.prove import fold_rounds     # folding driver: shape shrinks per round
from zorch.verify import verify         # replay a round sequence against a proof
```

- `ProverRound` / `VerifierRound` are `Protocol`s — implement the callable,
  don't subclass. Only the **message** crosses roles; challenges both sides
  derive go in the carry (`RunningClaim` is the standard verifier carry).
- `ProverStage.prove(claim, witness, transcript)` /
  `VerifierStage.verify(claim, reduction_proof, transcript)` return
  `ProveResult` / `VerifyResult` carrying the same reduced-claim type. Chain
  stages until `TrivialClaim`.
Doc: [stage-composition.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/composition/stage-composition.md).

## Fiat-Shamir transcript

```python
from zorch.transcript import Transcript, DuplexTranscript, sample_challenge, GrindError
from zorch.challenge import ChallengePolicy
```

`DuplexTranscript` is a device-side duplex sponge — a pytree threaded
functionally under `@jit`; `observe`/`sample` are device ops.
`ChallengePolicy(dtype)` pins how squeezed limbs become challenges in your
field; rounds take it as their `challenges=` argument. Byte-oriented host
transcripts for matching byte-level references: `zorch.byte_transcript`,
`zorch.sha256_field_transcript`.
Doc: [transcript.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/transcript.md).

## Polynomials

```python
from zorch.poly.multilinear import eval_mle, mle_fold, mle_coeffs_to_evals, mle_evals_to_coeffs
from zorch.poly.eq import eval_eq, expand_eq_to_hypercube
from zorch.poly.univariate import eval_coeffs
```

Leaf numeric helpers — element-wise field ops that fuse into the caller's
kernel; no `@jit` of their own.
Doc: [poly.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/poly.md).

## Hashing

```python
from hash_frx.permutation import Permutation          # the seam (Protocol)
from hash_frx.poseidon2.poseidon2 import Poseidon2    # a Permutation instance
from hash_frx.sponge import Sponge, SpongeParams
from hash_frx.compression import Compression, CompressionParams
```

Your project supplies the pinned params (width, field, constants) for *its*
permutation; sponge/compression code is shared.
Doc: [hash-frx](https://github.com/fractalyze/hash-frx/blob/main/docs/blocks/hash.md) (the symmetric layer is a separate repo).

## Merkle commitment

```python
from zorch.commit.merkle import MerkleTree, Opening
```

Binary tree over Sponge (leaves) + Compression (nodes). Strided and
sparse-multi-column variants: `zorch.commit.strided_merkle`, `zorch.commit.smcs`.
Doc: [commit.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/commit.md).

## Linear codes

```python
from zorch.coding.linear_code import LinearCode          # the seam (Protocol)
from zorch.coding.reed_solomon import ReedSolomon, BitReversedReedSolomon, fri_fold, fri_fold_k
```

RS encode on the native NTT; FRI folds live here (they are code operations,
not PCS-private).
Doc: [coding.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/coding.md).

## PCS — polynomial commitment

```python
from zorch.pcs.stage import Committer, CommittingOpener, OpeningClaim, OpeningWitness, OpeningProof
```

A PCS is a **committer plus an opening stage**. Instances under `zorch.pcs.*`:
`fri`, `basefold`, `kzg`, `whir`, `ligero`, `ligerito`, `ipa`, plus the
`jagged` overlay — each a package with `config` / `prover` / `verifier`
modules.
Doc: [pcs.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/pcs.md),
[jagged.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/jagged.md).

## Sumcheck

```python
from zorch.sumcheck.stage import SumClaim, SumcheckWitness, EvaluationClaim, SumcheckProver, SumcheckVerifier
from zorch.sumcheck.prover import StandardRound, CompressedProductRound, ProductSummand, RoundMsg
from zorch.sumcheck.verifier import SumcheckRound, CoeffsSumcheckRound, CompressedCoeffsSumcheckRound
```

The smallest complete stage; zerocheck/lincheck/LogUp-GKR configure it rather
than reimplement it. The call shape, end to end (product of two MLEs over
`n_vars` variables, evaluation tables of length `2**n_vars`):

```python
CH = ChallengePolicy(F)
state = fnp.stack([f, h])                 # stacked (m, N) factor table
claim = SumClaim(fnp.sum(f * h), n_vars)

prover = SumcheckProver(StandardRound(ProductSummand(2), challenges=CH))
verifier = SumcheckVerifier(SumcheckRound(2, challenges=CH))

proved = prover.prove(claim, SumcheckWitness(state), cheap_transcript(F))
verified = verifier.verify(claim, proved.reduction_proof, cheap_transcript(F))
# verified.ok covers round consistency; both roles derive the same
# EvaluationClaim (point, value) — the next stage's (or the opening's) input.
```

Doc: [sumcheck.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/sumcheck.md).

## LogUp-GKR

```python
from zorch.logup_gkr.stage import LogUpGkrProver, LogUpGkrVerifier, LogUpOutputClaim, InputLayerClaim, GkrProof
```

Fractional-sum GKR circuit (`zorch.logup_gkr.circuit`) reducing a public
output claim layer by layer to an input claim your PCS opening discharges.
Doc: [logup-gkr.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/blocks/logup-gkr.md).

## Worked full SNARK — importable

```python
from zorch.spartan.spartan import SpartanProver, SpartanVerifier, SpartanClaim, SpartanWitness, SpartanProof
from zorch.spartan.r1cs import R1CS
```

Spartan over R1CS — zerocheck + lincheck stages, transcript-only batching,
PCS-open glue, composed as paired stages — is the reference for what
consumer-style assembly looks like.
Doc: [spartan.md](https://github.com/fractalyze/zorch/blob/v0.2.0/docs/schemes/spartan.md).

## Test kit (for your tests, not production)

```python
from zorch.testkit.transcript import cheap_transcript   # real DuplexTranscript, cheap permutation
from zorch.testkit.random_field import rand_field, rand_ext_field
```

`cheap_transcript(F)` (positional field dtype) gives protocol-correct
Fiat-Shamir without Poseidon2 cost; `rand_field(seed=, shape=, dtype=)` builds
valid random field arrays without touching raw ints.
