# pcs — polynomial commitment seam

The *why* behind `zorch/pcs/`. The *what* lives in the code and its tests. Full
design and open decisions: epic issue
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

A Modern SNARK is IOP + PCS, and the PCS is the axis schemes vary on. `pcs` is the
one seam every scheme's commitment plugs into, with three concrete instances
spanning the design space: [`kzg`](#kzg-pairing-based) (pairing, trusted setup),
[`fri`](#fri-transparent) (transparent, hash-based), and
[`basefold`](#basefold-transparent-multilinear) (transparent, the multilinear
*matrix* commitment the [jagged PCS](#jagged-a-consumer-on-the-seam) builds on).
That schemes at opposite ends of the space satisfy the same two protocols is the
evidence the seam is not shaped after one family.

## Why the shape

**Two protocols, `PcsProver` and `PcsVerifier`, not one `Pcs`.** `commit` / `open`
are the prover's; `verify` is the verifier's. They are split for the same two
reasons the [sumcheck](sumcheck.md) block splits prover and verifier: `open` is an
interactive sub-protocol threading the [Fiat-Shamir transcript](hash.md), and the
two sides hold **asymmetric keys** — a KZG prover key is O(degree) (the SRS
powers) while its verifier key is three fixed group elements. A deployed verifier
must never carry the prover's key, so the boundary is a type. A static commitment
primitive like the Merkle [`commit`](commit.md) has neither property and stays a
single unified building block; the split lives in the PCS layer that *uses* it.

**Representation is the scheme's business.** The seam takes polynomials in whatever
form the scheme needs — KZG the coefficient basis (a powers-of-τ MSM), the FRI
family evaluations over a domain. No `PolynomialSpace` and no AIR/quotient
commitment index lives on the seam; those are FRI-implementation or consumer
concerns, kept out so no scheme's shape ossifies into the interface. A scheme is
named only on its instance (`kzg`, `fri`, `basefold`), never on the seam — the
agnostic non-negotiable, the same way `poseidon2` names a `Permutation` instance.

### kzg (pairing-based)

`commit` is `C = Σ aᵢ·[τⁱ]₁`, one `lax.msm`; `open` at `z` is the same MSM over the
quotient `(f(x) − f(z))/(x − z)`, with `f(z)` the division remainder. There are no
fold rounds — KZG's single-point opening is non-interactive — so the transcript
only feeds a batching challenge when more than one opening is bundled. The SRS is
split by `setup` into the O(degree) `KzgProvingKey` and the O(1) `KzgVerifierKey`,
both from one `τ`; that shared `τ` is the soundness invariant binding the two.

### fri (transparent)

The DEEP quotient trick turns a point opening into a low-degree test: to open `f`
at `z` with claim `v`, show `g(x) = (f(x) − v)/(x − z)` is low degree, which holds
exactly when `v = f(z)`. `g` is never committed — the verifier rebuilds its
codeword from the already-committed `f` at queried points — so `open` Merkle-commits
only the folded layers and threads the transcript through the fold challenges.
Structurally the opposite of KZG on the same seam: interactive, Merkle-backed
([commit](commit.md) + [coding](coding.md)'s RS encode and FRI fold), no trusted
setup, all field/NTT arithmetic.

### basefold (transparent, multilinear)

The multilinear-evaluation PCS. `commit` is the RS low-degree extension of each
column followed by a Merkle commit of the codeword rows. `open` opens the matrix
at one shared point `z ∈ F^{log S}` — returning the `K` per-column evals — by
RLC-batching the columns into a single codeword and running an interleaved sumcheck
that folds the MLE and the codeword by the same per-round challenge, then a
natural-order FRI query phase whose layer 0 is the RLC of the opened original rows
(reusing [`fri`](#fri-transparent)'s machinery); `verify` is the dual. Fidelity is
mathematical (prover↔verifier round-trip); the query phase carries no proof-of-work
grind yet — a known soundness gap shared with [`fri`](#fri-transparent).
The one structural difference from `kzg`/`fri` is that BaseFold is a **matrix
commitment** — the columns of an MLE `[2^v, w]` share one RS domain and the Merkle
leaves are codeword *rows* spanning every column, so the whole batch binds under a
**single** root, where `kzg`/`fri` return one root per polynomial. The seam permits
this because `commitment` is scheme-defined; the input convention stays uniform
(`commit` takes a `Sequence` of 1D column MLEs, like the other instances).

## Instance anatomy

Every *instance* (`kzg`, `fri`, `basefold`) follows one shape — shared wire types,
a prover, and a verifier in separate modules — so the instances stay
interchangeable and a new one has a template to fill. (`jagged` is exempt: it is a
[consumer on the seam](#jagged-a-consumer-on-the-seam), not an instance.) One part
of the shape is load-bearing rather than cosmetic: **the verifier module never
imports the prover module.** The prover/verifier asymmetry — a prover holds an
O(degree) key and the retained witness `commit` hands to `open`, a deployed
verifier carries neither — is a *module boundary*, not just a type.

**Conformance is mypy-enforced, not conventional** — the repo-wide seam pin
([conventions.md "Seam conformance pins"](../reference/conventions.md#seam-conformance-pins)).
The seam is generic — `PcsProver[C, D, P]` / `PcsVerifier[C, P]` — so each
instance parameterizes its pin with the scheme's wire types:

```python
if TYPE_CHECKING:
    _: type[PcsProver[FriCommitment, FriProverData, list[FriProof]]] = FriProver
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

The seam-level round-trip test (`zorch/pcs/testing/protocol_test.py`) drives all
three instances through one generic `commit → open → verify` driver typed
against the protocols alone — the behavioral proof that the instances are
interchangeable. (`JaggedPcsProver` is deliberately *not* pinned: it is a
[consumer on the seam](#jagged-a-consumer-on-the-seam), not an instance — its
`commit` takes blocks plus a stacking height, not a `Sequence` of polynomials.)

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
  [poseidon2](hash.md) permutation behind FRI's and BaseFold's Merkle layers, the
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
