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

**Python 3.11 on Linux x86_64 only.**

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

| | **`Round`** | **`Stage`** | **Named protocol operation** |
| --- | --- | --- | --- |
| **Represents** | one step of a repeated recurrence | one conditional claim reduction with separate prover/verifier roles | shared framing, reduction, or security-amplification step without its own proof section |
| **Owns** | the recurrence carry and the message on the wire | shared claim/proof contract; each role owns only its capabilities | no proof section |
| **Examples** | one sumcheck variable, one GKR layer | sumcheck, zerocheck, lincheck, LogUp-GKR, a PCS opening, Spartan | framed observation, domain separation, grinding, claim batching |
| **Composed by** | a recurrence driver inside a stage role | an explicit parent role implementation | the parent whose transcript and soundness accounting require it |

A composite stage writes its own dataflow in ordinary Python, like a PyTorch
module with a custom `forward`; there is no chain driver and no bridge.
`SpartanProver` / `SpartanVerifier` are the reference pair, and a deployed
`SpartanVerifier` is constructible with only a PCS verification key.

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

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
