# pcs — polynomial commitment seam

The *why* behind `zorch/pcs/`. The *what* lives in the code and its tests. Full
design and open decisions: epic issue
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

A Modern SNARK is IOP + PCS, and the PCS is the axis schemes vary on. `pcs` is the
one seam every scheme's commitment plugs into. Three instances anchor the ends of
the design space: [`kzg`](#kzg-pairing-based) (pairing, trusted setup),
[`fri`](#fri-transparent) (transparent, hash-based), and
[`basefold`](#basefold-transparent-multilinear) (transparent, the multilinear
*matrix* commitment the [jagged PCS](#jagged-a-consumer-on-the-seam) builds on).
That schemes at opposite ends of the space satisfy the same two protocols is the
evidence the seam is not shaped after one family; the families beside them live
in `zorch/pcs/` rather than in a list here, which would go stale the next time
one lands.

## Roles and claims

A PCS is **a committer plus a terminal stage**, and the two halves are different
kinds of thing. `commit` runs before any claim exists — it creates the object a
later claim is *about* — so it is a `Committer` (`zorch/pcs/stage.py`), not a
stage role. The opening is a claim reduction, and every family implements the
stage roles for it: `OpeningClaim` (the commitment plus the points) reduces to
`TrivialClaim`, the claim that holds by construction, with `OpeningProof`
carrying the values the prover computed while opening.

Reducing to `TrivialClaim` is what makes an opening *terminal*: nothing remains
for a later stage to prove, which is the shape of a complete argument rather than
one link. Values ride the proof and not the claim, because the prover computes
them during the opening — a claim carrying them would be unconstructible.

## Why the shape

**A committer plus an opening stage, not one `Pcs`.** `commit` / `open` are the
prover's, `verify` the verifier's, split for the two reasons
[sumcheck](sumcheck.md) splits its roles: `open` is an interactive sub-protocol
threading the [Fiat-Shamir transcript](transcript.md), and the sides hold **asymmetric
keys** — a KZG prover key is O(degree), its verifier key three group elements. A
deployed verifier must never carry the prover's, so the boundary is a type. The
Merkle [`commit`](commit.md) has neither property and stays unified; the split
belongs to the PCS layer *using* it.

**Representation is the scheme's business.** The seam takes polynomials in
whatever form the scheme needs — KZG the coefficient basis, the FRI family
evaluations over a domain. No `PolynomialSpace` and no AIR/quotient index lives
on it, so no scheme's shape ossifies into the interface. A scheme is named on its
instance, never on the seam.

### kzg (pairing-based)

`commit` is `C = Σ aᵢ·[τⁱ]₁`, one `lax.msm`; `open` at `z` is the same MSM over
the quotient `(f(x) − f(z))/(x − z)`, with `f(z)` the remainder. No fold rounds —
the single-point opening is non-interactive — so the transcript only feeds a
batching challenge when openings are bundled. `setup` splits the SRS into the
O(degree) `KzgProvingKey` and O(1) `KzgVerifierKey` from one `τ`, and that shared
`τ` is the soundness invariant binding them.

### fri (transparent)

The DEEP quotient trick turns a point opening into a low-degree test: to open `f`
at `z` with claim `v`, show `g(x) = (f(x) − v)/(x − z)` is low degree, which holds
exactly when `v = f(z)`. `g` is never committed — the verifier rebuilds its
codeword from the committed `f` at queried points — so `open` Merkle-commits only
the folded layers. Structurally the opposite of KZG on one seam: interactive,
Merkle-backed ([commit](commit.md) + [coding](coding.md)), no trusted setup, all
field and NTT arithmetic.

### basefold (transparent, multilinear)

`commit` is the RS low-degree extension of each column then a Merkle commit of
the codeword rows. `open` opens the matrix at one shared `z ∈ F^{log S}` by
RLC-batching the columns into a single codeword, running an interleaved sumcheck
that folds MLE and codeword by the same challenge, then a natural-order FRI query
phase whose layer 0 is the RLC of the opened rows; `verify` is the dual. The
query phase carries no proof-of-work grind yet — a soundness gap shared with
[`fri`](#fri-transparent).

Structurally it differs from `kzg`/`fri` by being a **matrix commitment**: the
columns of an MLE `[2^v, w]` share one RS domain and the Merkle leaves are
codeword *rows*, so the whole batch binds under a **single** root where the
others return one per polynomial. The seam permits this because `commitment` is
scheme-defined, and the input convention stays uniform.

## Instance anatomy

Every *instance* follows one shape — shared wire types,
a prover, and a verifier in separate modules — so the instances stay
interchangeable and a new one has a template to fill. (`jagged` is exempt: it is a
[consumer on the seam](#jagged-a-consumer-on-the-seam), not an instance.) One part
of the shape is load-bearing rather than cosmetic: **the verifier module never
imports the prover module.** The prover/verifier asymmetry — a prover holds an
O(degree) key and the retained witness `commit` hands to `open`, a deployed
verifier carries neither — is a *module boundary*, not just a type.

**Conformance is mypy-enforced, not conventional** — the repo-wide seam pin
([conventions.md "Seam conformance pins"](../reference/conventions.md#seam-conformance-pins)).
The seam is generic — the opening stage roles over `OpeningClaim[C]` /
`OpeningWitness[D]` / `OpeningProof[P]` — so each
instance parameterizes its pin with the scheme's wire types:

```python
if TYPE_CHECKING:
    _: type[
        ProverStage[
            OpeningClaim[FriCommitment],
            OpeningWitness[FriProverData],
            TrivialClaim,
            OpeningProof[list[FriProof]],
        ]
    ] = FriProver
```

Because those wire types are zorch-owned nominal types, the PCS pins have full
teeth despite the `frx.Array ≡ Any` caveat: `commit`'s prover data disagreeing
with `open`'s fails the pin. That is why prover data is always a named
dataclass, never a bare list or tuple.

**Commitments are aliases until they grow structure.** Every scheme's commitment
is literally an `Array` (KZG: `[K]` G1 points; FRI: `[K]` roots; BaseFold: one
root) and feeds `Transcript.observe` / the jagged structure bind directly, so
each scheme names it with a `TypeAlias` (`KzgCommitment`, …) rather than a
wrapper dataclass — a wrapper would cost an unwrap at every observe site and a
pytree registration for nothing. Promote an alias to a dataclass only when a
commitment gains real structure.

The seam-level round-trip test (`zorch/pcs/testing/protocol_test.py`) drives the
three design-space anchors through one generic `commit → open → verify` driver
typed against the protocols alone — the behavioral proof that the instances are
interchangeable. (Jagged is deliberately *not* pinned: it is a
[consumer on the seam](#jagged-a-consumer-on-the-seam), not an instance — its
`commit_region` takes blocks plus a stacking height, not a `Sequence` of
polynomials, and `prove_jagged_eval` opens against SP1's shard layout.)

## Jagged: a consumer on the seam

The jagged PCS (`zorch/pcs/jagged/`) is the first `basefold` consumer: it
densifies variable-height columns into one MLE and commits it, then binds the
jagged structure (row/column counts, hashed) into the single root. The layering is
deliberate — `chip → blocks` is the *consumer's* concern (its trace layout), while
`blocks → dense MLE` is zorch's, because the layout must match the `t_c`
prefix-sum convention the jagged indicator (`zorch/pcs/jagged/poly.py`) reads.
The structure bind lives in the jagged layer, not the generic seam — it is why
BaseFold's single-root commitment matters: there is exactly one root to hash the
structure against.

Opening reduces a jagged evaluation to a BaseFold opening of that dense MLE via
two sumchecks (an outer Hadamard `Σ D·J̃` and an inner jagged-assist that collapses
the `O(L)`-column indicator sum to one branching-program leaf) plus the stacked
`z_final` split — see [jagged](jagged.md#opening). Whole-protocol composite fusion
is deferred (gated on `frx.lax.composite` accepting field dtypes).

## Fusion by construction

The PCS seam is agnostic; each instance's `commit`/`open`/`verify` lowers down one
of three tiers, and which tier an op takes is the only thing that varies:

- **GPU normal-form** — element-wise field ops + the inherent `Σ`/NTT (compile-fast,
  portable): KZG's quotient division and Horner, FRI's `fri_fold` and the RS NTT.
- **GPU blessed primitive** — a dedicated `stablehlo` op or custom emitter
  (run-fast): KZG's `lax.msm` (commit and the opening proof), the
  [poseidon2](https://github.com/fractalyze/hash-frx/blob/main/docs/blocks/hash.md) permutation behind FRI's and BaseFold's Merkle layers, the
  NTT (the RS LDE in both FRI and BaseFold commit).
- **CPU-legalized primitive** — `lax.pairing_check` for KZG `verify`, which has no
  GPU kernel; the verifier is O(1), so the host round-trip (MSM on GPU →
  materialize → pairing on CPU) is irrelevant.

This is why "one fused kernel" is a property of an *instance's* lowering, not of the
seam: MSM is a GPU-only kernel, pairing is CPU-only, and the FRI fold/NTT lower on
both. See the hub [fusion north star](../README.md#fusion-north-star).

## AOT and the host/device boundary

Two rules every instance inherits from zorch being AOT, orthogonal to the lowering
tiers above:

- **One device zone, host layout outside.** An instance's `commit`/`open`/`verify`
  is one `@jit` region — a single CUDA-graph-capturable dispatch. Value-independent
  layout (sizes, padding tiers, prefix sums) is computed host-side *before* the
  zone, never as a mid-zone host sync.
- **Shapes are a function of input shapes, not values.** Data-dependent extents are
  padded to a static `2^tier` capacity derived host-side, so a scheme compiles once
  per tier (and per batch shape) rather than per input. This is what lets an
  evaluation PCS over variable-height data stay AOT — it mirrors the jagged eval's
  log-area tiers.
