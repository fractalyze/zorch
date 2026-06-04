"""constraint_eval — mark an α-RLC'd constraint evaluation as one fused op.

Wraps `eval_fn(trace)` (a per-row constraint evaluation producing `[..., K]`)
followed by the random-linear-combination `sum_k alpha_k * C_k` in a
`jax.lax.composite` named `zorch.constraint_eval`. The marker preserves the
high-level "evaluate K constraints, fold them under one challenge vector" shape
so a recognizing compiler can emit a single kernel that accumulates
`alpha_k * C_k` incrementally and never materializes the `[..., K]` constraint
tensor; an unrecognizing compiler inlines the decomposition to the identical
result.

Agnostic: `eval_fn` is opaque — its body belongs to the caller — and the marker
and fold carry no proving-scheme or zkVM knowledge. Sibling of
`zorch.fusion::fused_region`, and shares the `stablehlo.CompositeOp` fallback in
`zorch._composite.composite_or_inline`: absent the composite backport, the
decomposition runs inline instead — numerically identical, only the marker is
dropped.

The RLC is emitted as an unrolled fold (`acc += alpha_k * C_k`), not `jnp.dot`
/ `@`: a reduction in the marked body would split the region under the
single-kernel rewriter, and the unrolled fold keeps the accumulation
register-resident. Field addition is associative and exact, so the fold's order
is byte-equal to the dot it replaces.
"""

from __future__ import annotations

from collections.abc import Callable

from jax import Array

from zorch._composite import composite_or_inline

CONSTRAINT_EVAL_MARKER = "zorch.constraint_eval"


def constraint_eval(
    eval_fn: Callable[[Array], Array],
    trace: Array,
    alpha: Array,
    *,
    name: str = CONSTRAINT_EVAL_MARKER,
) -> Array:
    """Mark `sum_k alpha_k * eval_fn(trace)_k` as one `zorch.constraint_eval`.

    `eval_fn(trace)` must produce constraints in the trailing axis (`[..., K]`),
    matching `alpha`'s trailing length `K`; the result drops that axis. The K
    count and the alpha operand index ride along as composite attributes for the
    recognizing emitter; they are metadata, so the decomposition ignores them (it
    reads K from `alpha`'s static shape). On a jaxlib without
    `stablehlo.CompositeOp` the marker is dropped and the decomposition runs
    inline (see the module docstring).
    """
    num_constraints = alpha.shape[-1]
    if num_constraints < 1:
        raise ValueError(
            f"alpha must carry at least one coefficient, got {num_constraints}"
        )

    def decomposition(trace: Array, alpha: Array, **_attrs: object) -> Array:
        constraints = eval_fn(trace)
        acc = constraints[..., 0] * alpha[..., 0]
        for k in range(1, num_constraints):
            acc = acc + constraints[..., k] * alpha[..., k]
        return acc

    return composite_or_inline(
        decomposition,
        trace,
        alpha,
        name=name,
        num_constraints=num_constraints,
        alpha_operand_idx=1,
    )
