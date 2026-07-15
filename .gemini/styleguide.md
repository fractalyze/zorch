# zorch review style guide

Guidance for automated code review on `zorch` — FRX-native building blocks for
Modern SNARKs (IOP + PCS). Keep comments focused on these repo-specific rules;
generic style nits should stay at or below MEDIUM severity.

## Two non-negotiables (flag at HIGH severity)

1. **Proving-scheme- and implementation-agnostic.** No building block may import,
   name, or special-case a particular downstream implementation — a zkVM (SP1,
   OpenVM, ZisK, …), a zkML or zkTLS prover — or a particular proving scheme. If
   such knowledge appears in a block, it belongs in the consumer, not in `zorch`.
   Flag any such leakage — vendor names, FFI shims, `pure_callback` to a vendor
   lib, scheme-specific constants baked into a generic block.

1. **Fusion by construction.** A `Round`, an `absorb`/`squeeze`, a
   `commit`/`open`, a fold step, and a hash permutation must each lower to a
   single fused kernel — by construction, never by a per-primitive compiler
   pattern-match. In code that is meant to fuse (round bodies, permutations,
   linear layers), flag:

   - `jnp.dot` / `jnp.matmul` / `lax.reduce` / `jnp.sum` / gather-shaped slicing
     inside a fusable body — these become `kInput`/gather fusion boundaries on
     GPU. Linear layers (e.g. MDS) must be expressed as explicit field add/mul.
   - Reliance on a downstream compiler pattern-matcher to recognize a shape.
   - Unrolling many rounds into one `@jit` (risks `LAUNCH_OUT_OF_RESOURCES`);
     prefer the `fused_rounds` primitive / a marked loop.

## FRX conventions

- **Functional, pytree state.** State threads immutably — operations return new
  state, never mutate in place. Register state containers as pytrees/dataclasses.
- **Bounded compile size.** Don't Python-unroll over data length; use
  `lax.scan` / `lax.fori_loop` / the `fused_rounds` primitive so the compiled
  graph size is independent of input length.
- **`@jit` boundaries.** Compilation happens at the caller boundary; don't put
  `@jit` on methods called inside `vmap`/`fori_loop` where it would freeze
  `self`. Keep field dtypes (`zk_dtypes`) end-to-end; don't add canonical↔Mont
  conversions at internal boundaries.

## Tests

- Every behavior change needs a test. New blocks need tests for the intended
  behavior and the edge/boundary cases. Where a reference implementation exists
  (e.g. poseidon2 vs Plonky3), prefer a byte/numeric match against it.
- **Montgomery dtypes only** (flag at HIGH severity): tests draw field AND
  curve-point elements as the `_mont` form of every `zk_dtypes` family that
  ships one — fields (`koalabear(x4)`, `babybear(x4)`, `goldilocks(x3)`,
  `bn254_sf`) and the bn254 G1/G2 `affine`/`jacobian`/`xyzz` point types.
  Flag any bare form in a test, INCLUDING when the surrounding file still
  uses it (file-local style can be the legacy half; the documented rule
  wins). The one exception is a test genuinely about the canonical integer
  encoding itself, marked `# canonical-encoding test` on the same line —
  the `mont-test-dtypes` pre-commit hook enforces the same rule
  mechanically.

## Comments, commits, PRs

- **Terse, why-not-what.** Comments explain non-obvious rationale, not a
  restatement of the code. No historical notes (`// X used to …`), no
  session-local labels (`Q1:A`, `PR-β`), no references to uncommitted files or
  local-only paths.
- **Conventional commit prefixes** (`feat` / `fix` / `refactor` / `chore` /
  `docs` / `test`); breaking user-facing changes get `!`.
- **No `Co-Authored-By:` or "Generated with …" trailers** in commits or PR
  bodies.

## Scope discipline

- Touch only what the change needs. Flag unrelated refactors, dead-code removal
  the PR didn't set out to do, or "drive-by" reformatting that inflates the diff.
