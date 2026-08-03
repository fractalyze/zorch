# zorch

> **SNARK = Σ IOP Round**

FRX-native building blocks for proof systems. `zorch` sits between **FRX** —
Fractalyze's fork of [JAX](https://github.com/jax-ml/jax) — and the proof systems
that consume it: FRX provides tracing and codegen, lowered through **Fractalyze
XLA**, its fork of stock [XLA](https://github.com/openxla/xla) that adds native
field and elliptic-curve types. `zorch` provides the reusable pieces a proof
system is assembled from.

The banner is a mnemonic for where the work is, not a definition: repeated IOP
rounds are the core interactive computation, and a non-interactive argument adds
its relation, a Fiat–Shamir transform, and commitment machinery around them.
Folding and other families reuse the same pieces without fitting the equation
literally. So zorch's two units are the **round** — one step of a repeated
recurrence — and the **stage** — one claim reduction with paired prover and
verifier roles.

## Installation

**Python 3.11 on Linux x86_64, or macOS on Apple Silicon.** (`frxlib` ships a
cp311 wheel for those two platforms only — not 3.12/3.13, not Intel Macs.)

Install as `pyzorch`, import as `zorch` — the `zorch` name on PyPI belongs to an
unrelated project.

### CPU

```sh
pip install pyzorch
```

### GPU (CUDA 12)

```sh
pip install pyzorch 'frx[cuda12]' \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The extra index carries the CUDA plugin wheels, which are too large for PyPI's
per-file limit. It is not needed for the CPU tier.

### Verify

```sh
python -c "import frx, zorch; print(frx.devices()); print(zorch.__version__)"
```

`[CpuDevice(id=0)]` means the CPU tier; a CUDA install prints the GPU devices.

## Design Philosophy

- **Proving-scheme-agnostic.** The blocks are reusable across proving schemes,
  rather than encoding a single one. `Round`, Fiat-Shamir, `Polynomial`, `PCS`,
  fold, and zero-check compose into FRI, sumcheck, GKR, STARK, Basefold, WHIR,
  …; pairing-based schemes plug in by swapping the `PCS` block (e.g. a
  KZG-style commitment).
- **Implementation-agnostic.** `zorch` targets the proving scheme, not any one
  downstream implementation — a zkVM, a zkML prover, a zkTLS prover. Each plugs
  in as a *consumer*; nothing implementation-specific leaks into a block.
- **Fusion-first.** Each `Round` — and each `commit`/`open`, each
  `absorb`/`squeeze`, each fold step, and a hash permutation's internal rounds —
  must lower to **one replayable device unit**. We get there by *construction*,
  not by a per-primitive pattern-matcher in the compiler. The
  [fusion north star](docs/README.md#fusion-north-star) is where that unit is
  defined and measured.
- **Easy to assemble.** These are building blocks; the API optimizes for snapping
  them together.

## Building blocks

The two units differ in kind, not size. A **round** is *directional and
repeated*: one step of a homogeneous recurrence, driven in a loop. A **stage**
is *paired and whole*: a claim reduction with a prover role and a verifier role
that can be deployed separately. A whole sumcheck stage may be cheaper than one
GKR round — repetition versus pairing decides, never scale.

```text
ProverRound:    (carry, transcript)          -> (carry, transcript, message)
VerifierRound:  (carry, transcript, message) -> (carry, transcript, ok)

ProverStage.prove(claim, witness, transcript)      -> ProveResult
VerifierStage.verify(claim, reduction_proof, ...)  -> VerifyResult
```

The **claim** is the public assertion entering the stage; the **witness** is the
prover's private evidence. Both roles derive the same **reduced claim**, and the
reduction proof does not establish it — it establishes the source claim
*conditional on* it:

```text
reduced claim is true  =>  source claim is true
```

That conditional is what lets stages chain: each one's reduced claim is the
next one's source claim, so a proof system is a sequence of reductions ending at
`TrivialClaim`, which holds by construction and leaves nothing left to prove. An
argument of knowledge is exactly a reduction to the trivial claim, and a PCS
opening is the usual terminal one.

Dense sumcheck is the smallest complete stage, entire:

```python
@dataclass(frozen=True)
class SumClaim:         # public: this polynomial sums to `value` over the cube
    value: Array
    rounds: int


@dataclass(frozen=True)
class EvaluationClaim:  # reduced: ...provided it evaluates to `value` at `point`
    point: Array
    value: Array


class SumcheckProver(ProverStage[SumClaim, SumcheckWitness, EvaluationClaim, Array]):
    def prove(self, claim, witness, transcript):
        pre = transcript
        # the rounds live inside the stage — one per variable, driven in a loop
        _, _, messages = fold_rounds(
            self.prover_round, witness.state, transcript, claim.rounds
        )
        reduction_proof = fnp.stack(messages)
        # The prover's carry is the folded tables: each round squeezes its
        # challenge, folds with it, and drops it. The point and running value
        # accumulate in the verifier's carry instead, so replaying is how the
        # prover obtains a reduced claim it never built. The transcript comes
        # back with it and matches the one fold_rounds discarded.
        point, value, replayed, _ = verify(
            self.verifier_round, claim.value, reduction_proof, pre
        )
        return ProveResult(EvaluationClaim(point, value), reduction_proof, replayed)
```

`SumcheckVerifier.verify` is that same replay against the proof alone. Nothing
here knows where the sum came from: zerocheck, lincheck and LogUp-GKR each
configure this stage and export their own claim type.

A proof system is those reductions chained until the claim is trivial.
`SpartanProver` / `SpartanVerifier` are the worked composite — four stages, a
transcript-only batching step, and prover data that skips to the opening — in
[`docs/schemes/spartan.md`](docs/schemes/spartan.md).

Contracts, ownership rules, the round-vs-stage-vs-committer decision table, and
reuse guidance:
[`docs/composition/stage-composition.md`](docs/composition/stage-composition.md).

## Development

`zorch` is pure Python on FRX, run against its GPU plugin. A virtualenv with
the pinned toolchain:

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

Install the git hooks with both stages named. Plain `pre-commit install` wires
only the `pre-commit` stage, which leaves the commit-message linter inactive —
formatting hooks fire while a malformed commit message sails through to CI:

```sh
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
a valid type, a lowercase summary with no trailing period, a header of at most
80 characters, and a body on everything but `docs`. The same linter runs in CI
over every commit in a pull request and over the PR title, which matters here
because this repo squash-merges with the title as the subject.

The dev loop — per-workspace venvs, developing against a local Fractalyze XLA
build, the FRX compile-cache rule — lives in [`docs/reference/development.md`](docs/reference/development.md).

## Documentation

- **Task-indexed docs hub:** [`docs/README.md`](docs/README.md) — indexes every
  design doc by what you're trying to do.

### Agent skill

[`skills/using-zorch`](skills/using-zorch) packages this guide for AI coding
agents — install commands, block import paths, field-dtype rules, and the
consumer boundary — so an agent writing code *against* zorch in your repo has
them on hand. Install with any agent via [skills.sh](https://skills.sh):

```sh
npx skills add fractalyze/zorch
```

or as a Claude Code plugin:

```text
/plugin marketplace add fractalyze/zorch
/plugin install zorch@zorch
```

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
