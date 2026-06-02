# sumcheck — the `Round` building block

A product sumcheck prover, built from the core `Round`
abstraction. This is a usage guide; the design rationale lives on the epic issue
[fractalyze/zorch#1](https://github.com/fractalyze/zorch/issues/1).

## Usage

Prove `Σ_x f(x)` for a multilinear `f` — one degree-1 round per variable:

```python
import jax.numpy as jnp
import zk_dtypes
from zorch.prove import prove
from zorch.sumcheck import SumcheckRound
from zorch.transcript import StubTranscript

F = zk_dtypes.koalabear
f = jnp.arange(1, 17, dtype=F)            # 2^4 evals of a multilinear
challenges = jnp.arange(2, 6, dtype=F)    # one per variable (stubbed transcript)

final, transcript, proof = prove(
    SumcheckRound(degree=1), [f], StubTranscript(challenges))
# proof[i] = round i's univariate evaluated on [0..degree];
# final[0] == f(challenges).
```

A product sumcheck `Σ_x Π_k f_k(x)` is the same call with the factors as the
state list and `degree = len(factors)` (the round sums the product of the
state factors):

```python
prove(SumcheckRound(degree=2), [a, b], StubTranscript(challenges))
```

## How it fits together

- `Round` (`round.py`) is the composable unit, like an `nn.Module`: subclass and
  implement `__call__(state, transcript) -> (state, transcript, msg)`, using the
  base `commit` (absorb a message) and `challenge` (squeeze `n` challenges). The
  transcript is an immutable carry — threaded in and out, never mutated.
- `SumcheckRound` is one round: `round_poly` (the prover's message) → `commit` →
  `challenge` → `fold`. `prove` runs it once per variable; a composite protocol
  is itself a `Round`, so there is no per-protocol `.prove()` chain. State is
  `list[Array]` (one MLE per factor); the round sums their product and folds
  each at the challenge — all inline, with no dependency on a separate polynomial
  layer.

## Notes for the next reader

- **The transcript is stubbed.** `StubTranscript` replays preset challenges; the
  real duplex-sponge transcript (#3) implements the same `Transcript` Protocol
  and drops in without touching any round.
- **Fusion is by construction, not a decorator.** `round_poly`/`fold` are kept to
  element-wise field ops plus the one inherent `Σ` (no `reduce`/`gather` beyond
  it), so a round body stays foldable into one kernel. The marker + generic zkx
  emitter that actually emit it are a later, cross-repo step — there is
  deliberately no placeholder marker in the tree now.
- **`jnp.arange(dtype=<extension field>)` raises** (`iota` is unimplemented for
  extension dtypes such as `koalabearx4`). Build index/domain arrays in the base
  field and `.astype(EF)`. Base→extension embedding and extension arithmetic
  otherwise work, so a base-field MLE folds correctly against extension-field
  challenges.

## Tests

`PYTHONPATH=. python zorch/sumcheck/testing/prove_test.py` (and the per-module
`*_test.py` files). Coverage: the sumcheck consistency identity over `koalabear`,
plus a base-MLE / `koalabearx4`-challenge path.
