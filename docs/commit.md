# commit — Merkle commitment

The *why* behind `zorch/commit/`. The *what* lives in the code and its tests, the
executable usage that runs on every commit. Full design and open decisions: epic
issue [fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

The sibling [`jagged`](jagged.md) module under `commit/` — the verifier-side
point-eval a jagged opening is checked against — is documented separately.

## Why the shape

**Layer-by-layer binary tree on Sponge + Compression.** `commit` hashes each
matrix row to a leaf digest (the [Sponge](hash.md)), then folds sibling pairs per
layer (the [Compression](hash.md)) down to a single root, returning
`(raw_root, digest_layers)` leaf-first. The two halves agree by contract —
`leaf_hasher.out == compressor.chunk`, and the compressor is 2-to-1 (`arity == 2`)
— so the tree is just "a leaf hasher and a 2-to-1 compressor", nothing about which
hash.

**No domain separator, no proof layout.** `commit` returns raw digests only. The
domain separator, the opening / proof wire layout, and the verify error codes are
scheme-specific and live in the consumer (whir-zorch's SMCS). Keeping them out is
exactly what makes the tree reusable across schemes — the agnostic
non-negotiable. `open` / `verify` here are the structural path-and-rebuild
mechanics, not a scheme's soundness story.

## Fusion by construction

Each Merkle layer is one `vmap` over its nodes — internally one `compress`, at the
leaf one `hash`, i.e. one permute per node-batch — and the fold unrolls (layer
count is static), so no host-driven loop appears. The permute is the fusion unit
([hash.md](hash.md#fusion-by-construction)): once it is captured to a kernel (the
poseidon2 path, [#25](https://github.com/fractalyze/zorch/issues/25)), a whole
layer is one GPU kernel. See the hub [fusion north star](README.md#fusion-north-star).
