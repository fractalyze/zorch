# STATUS — fuse adjacent Merkle parent levels into one GPU dispatch

Tracking: **fractalyze/flock-zorch#191** (port of Layr-Labs/flock-challenge
`fb7e3506c9`, kernel `parent_hash3`). This branch:
`perf/191-merkle-level-fusion`.

## TL;DR

The **zorch-side half** of the port is landed and **byte-gated**: `MerkleTree`
now folds runs of adjacent levels as **one `zorch.merkle_fold` region** instead
of one `vmap` per level, with every level still written to its ordinary
flat-tree slot (so `open` is unchanged). The fused fold is proven
**bit-identical** to the per-level fold.

The **measured launch drop does not land in this branch**, and cannot: the
actual level-fusing kernel is a **vendor XLA/MLIR emitter** in the pinned
`frx-cuda12-plugin` wheel (the fork), which the session isolation forbids
building. Absent that emitter the marker **decomposes to the per-level fold** —
byte-identical, identical kernel count. This is the deliberate seam the emitter
plugs into, exactly as `zorch.sponge_hash` is the seam for `SpongeHashFusion`.

## What the reference does (and why it is not a JAX rewrite)

`parent_hash3` (Metal): a 128-thread group consumes 256 children and emits
128/64/32 parents into their flat-tree slots, keeping the two intermediate read
sets in threadgroup memory — deleting 2 launches + 2 global read passes per
fused triple. In zorch each `compress` lowers to an opaque `zorch.poseidon2`
custom fusion; **XLA will not merge two custom fusions across the compress
boundary** (same class as xla#168 kNtt), so keeping the intermediate layer
resident across levels requires a *custom fused emitter*, not a JAX
restructuring. Confirmed empirically below.

## What landed (this branch, all in `zorch/`)

- `zorch/commit/merkle.py`
  - `MerkleTree(..., fused_levels: int = 1)` — keyword-only knob; `1` = the
    historical per-level fold (default, unchanged, no marker). Joins the value
    identity (`__eq__`/`__hash__`) since it traces a different body (#214).
  - `MERKLE_FOLD_MARKER = "zorch.merkle_fold"` — the whole-region marker,
    name-routed (like `zorch.sponge_hash`), carrying `arity` / `digest_elems` /
    `levels` attrs and the per-level `zorch.poseidon2` markers nested inside.
  - `_fold_levels(layer, num_levels)` — emits one region per run of whole
    levels; its decomposition runs the successive `compress` folds and returns
    **each** level's layer (leaf-to-root), which is what keeps every node in its
    flat-tree slot.
  - `_fusible_levels` / `_fold_one_level` — greedy grouping + the padding-aware
    per-level tail (a k-ary short level or a group shorter than `fused_levels`
    falls back exactly as before, matching the reference's per-level tail).
  - Gated on `compressor.has_dedicated_fusion`: a non-dedicated permutation
    stays per-level (the region only has value when it nests dedicated
    per-permute kernels for the emitter to keep resident).
- `zorch/commit/testing/merkle_test.py` — `FusedLevelsMerkleTest` (9 tests, the
  byte gate).

## Byte gate — PASSING (GPU, RTX 5090, frx 0.10.1.dev20260801051831)

```
$ python -m pytest zorch/commit/testing/merkle_test.py::FusedLevelsMerkleTest -q
.........                                                                 [100%]
9 passed in 123.20s

$ python -m pytest zorch/commit/testing/merkle_test.py -q      # full file, no regression
.................................................                        [100%]
49 passed in 680.17s
```

The gate asserts, for `fused_levels=3` vs `fused_levels=1`: identical root,
identical **every** digest layer, identical openings, `verify` holds, the
Plonky3 golden root still matches on the fused path, and structurally the fused
lowering carries `zorch.merkle_fold` nesting `zorch.poseidon2`. Covered:
binary depth 4 (fused triple + per-level tail), binary depth 6 (two triples),
arity-4 with a padding tail, the non-dedicated-fusion fallback.

## Launch measurement (the primary signal) — no drop yet, by construction

Optimized HLO fusion-kind counts, depth-9 fold (512 leaves):

```
[per-level fused_levels=1]  OPTIMIZED: {Custom: 9, Loop: 9}   merkle_fold in lowered: False
[fused     fused_levels=3]  OPTIMIZED: {Custom: 9, Loop: 9}   merkle_fold in lowered: True
IDENTICAL optimized launch structure (no vendor emitter): True
```

`Custom` = the per-level `zorch.poseidon2` kernels (one per level); `Loop` = the
per-level pad/reshape/truncate data movement. The `zorch.merkle_fold` marker is
present in the **lowered** HLO but XLA **decomposes** it (no recognizer), so the
optimized HLO — hence the launch count — is unchanged. An nsys trace would show
the same: 0 launch drop today. With the emitter, the 9 `Custom` parent kernels
collapse toward ~3 fused-triple dispatches + tail (the `parent_hash3` win).

## What remains — the vendor `MerkleFusion` emitter (NOT buildable here)

The pinned `frx-cuda12-plugin` ships `Poseidon2Fusion` and `SpongeHashFusion`
emitters (+ CPU twins) but **no** merkle/fold emitter (verified via `strings` on
`xla_cuda_plugin.so`). Realizing the launch drop needs a new emitter in the
Fractalyze XLA fork:

- **Where:** `xla/backends/gpu/codegen/` (GPU) + the CPU twin, dispatched by
  composite name in `fusions.cc` on `"zorch.merkle_fold"` — mirroring
  `SpongeHashFusion`, which already parses *nested `zorch.poseidon2`
  composite.attributes* (the exact shape this marker emits).
- **ABI:** one operand = the level-L layer `[n, chunk]`; `levels` outputs
  `[n/arity, chunk], [n/arity^2, chunk], …`, each destination-passed to its
  flat-tree slot. Read `arity`/`digest_elems`/`levels` off `composite.attributes`
  and the nested `zorch.poseidon2` operands for the round constants.
- **Kernel:** one launch per fused group; hold the intermediate layers in
  shared memory across the `levels` compress rounds (the `parent_hash3`
  threadgroup-memory move); each round is a whole number of warps.
- **Then:** bump the `requirements.in` pin to the wheel carrying it; the zorch
  code here needs **no** change — the byte gate already proves the marked and
  decomposed folds agree, so the emitter only has to match the decomposition.

Out of scope for this session by the stated isolation (no touching
`~/Workspace/xla`; the wheel is a prebuilt pin) and because a vendor emitter
ships through a separate XLA-fork PR + wheel republish, not a zorch PR.

## Trivial follow-ups (zorch-side, safe)

- Thread `fused_levels` through `StridedMerkleTree` (its internal top
  `MerkleTree`) and `smcs` so the real commit paths opt in once the emitter
  exists. Left off to keep this change minimal and the blast radius on
  `MerkleTree` only.
