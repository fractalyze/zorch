# Polynomial commitment schemes

The `Pcs` seam (`zorch/commit/pcs.py`) is zorch's multilinear-evaluation
commitment interface. `Basefold` (`zorch/commit/basefold.py`) is the first
implementation; the jagged PCS (`zorch/commit/jagged/`) is the first consumer.

## The interface

`commit(mle: [2^v, w]) -> (commitment, prover_data)`. `open` / `verify`
complete the evaluation argument (P3).

- **`commitment`** — the succinct public value (a Merkle root). It enters the
  Fiat-Shamir transcript and is what the verifier receives.
- **`prover_data`** — the retained witness `open` consumes (full Merkle tree /
  codeword + metadata). Never sent to the verifier.

The split is the contract: it makes `commit` transcript-observable and `open`
a pure function of retained prover state. Every PCS family in the ecosystem
(SP1, plonky3, whir) uses this shape.

## Design rules

1. **A `Pcs` receives an MLE, nothing else.** No scheme- or zkVM-specific
   structure leaks in. A consumer that has variable-height columns, chips, or
   any domain layout densifies to an MLE *before* calling `commit`. (Repo
   non-negotiable #1: scheme/zkVM-agnostic.)
2. **One device zone, host layout outside.** The device commit (RS-LDE →
   Merkle → any structure bind) is one `@jit` region — a single
   CUDA-graph-capturable dispatch (see the fusion north star in
   `docs/README.md`). Value-independent layout (sizes, padding tiers, prefix
   sums) is computed host-side before the zone, never as a mid-zone host sync.
3. **AOT: shapes are a function of input shapes, not values.** Data-dependent
   extents (e.g. a jagged region's total area) are padded to a static `2^tier`
   capacity derived host-side, so `commit` compiles once per tier (and per
   structural block count). Mirrors the jagged eval's log-area tiers.

## Layering for jagged

`chip → blocks` is the consumer's (e.g. whir-zorch); `blocks → dense MLE
layout` is zorch's, because the layout must match the `t_c` prefix-sum
convention the jagged indicator (`zorch/commit/jagged/poly.py`) reads. The structure
binding (row/column counts hashed into the commitment) lives in the jagged
layer, not the generic PCS.
