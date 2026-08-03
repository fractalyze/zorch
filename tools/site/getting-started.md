# Get started

From zero to a verified sumcheck proof in about two minutes. Everything runs
on the CPU tier — no GPU required.

## 1. Install

Python **3.11**, Linux x86_64 or macOS Apple Silicon:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install pyzorch
python -c "import frx, zorch; print(frx.devices())"   # -> [CpuDevice(id=0)]
```

Anything else printed? See [Setup & troubleshooting](guide/setup.md).

## 2. Prove something

The claim: *two random multilinear polynomials' product sums to `v` over the
3-cube*. Save as `first_proof.py`:

```python
import zk_dtypes
import frx.numpy as fnp
from zorch.challenge import ChallengePolicy
from zorch.sumcheck.stage import (
    SumClaim, SumcheckWitness, SumcheckProver, SumcheckVerifier,
)
from zorch.sumcheck.prover import StandardRound, ProductSummand
from zorch.sumcheck.verifier import SumcheckRound
from zorch.testkit.transcript import cheap_transcript
from zorch.testkit.random_field import rand_field

F = zk_dtypes.koalabear_mont          # a native finite-field dtype
n_vars, N = 3, 8

f = rand_field(0, (N,), F)            # two multilinear polys, as eval tables
h = rand_field(1, (N,), F)

CH = ChallengePolicy(F)               # how challenges are squeezed in F
claim = SumClaim(fnp.sum(f * h), n_vars)

prover = SumcheckProver(StandardRound(ProductSummand(2), challenges=CH))
verifier = SumcheckVerifier(SumcheckRound(2, challenges=CH))

proved = prover.prove(claim, SumcheckWitness(fnp.stack([f, h])), cheap_transcript(F))
verified = verifier.verify(claim, proved.reduction_proof, cheap_transcript(F))

assert bool(verified.ok)
assert bool(proved.reduced_claim.value == verified.reduced_claim.value)
print("verified:", bool(verified.ok))
print("reduced claim: f·h(", verified.reduced_claim.point, ") =",
      verified.reduced_claim.value)
```

```sh
python first_proof.py
# verified: True
# reduced claim: f·h( [...] ) = ...
```

## 3. What just happened

- **A stage reduced a claim.** `SumcheckProver.prove` turned the `SumClaim`
  ("sums to v") into an `EvaluationClaim` ("…provided f·h(r) = u"), plus a
  proof. The verifier replayed the proof and derived the *same* reduced
  claim — that agreement, not the prover's word, is the guarantee.
- **The transcript did Fiat-Shamir.** Both roles threaded a transcript;
  every challenge was squeezed from it after the corresponding message was
  observed, so neither side could adapt.
- **A real prover chains stages** until the claim is trivial — the reduced
  evaluation claim here would feed a PCS opening stage next. Spartan ships
  in the package as the worked full chain.

## 4. Where next

| You want to… | Read |
| --- | --- |
| See every block and its import path | [Blocks & imports](guide/blocks.md) |
| Assemble a full prover | [Assembling a prover](guide/building-on-zorch.md) |
| Use the GPU and keep code fused | [Keeping code fused](guide/frx-and-fusion.md) |
| Avoid the field-dtype traps | [Field-dtype sharp bits](guide/field-dtypes.md) |
| Understand why the blocks look this way | [Design docs](README.md) |
