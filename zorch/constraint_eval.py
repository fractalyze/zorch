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
`zorch.fusion::fused_region`, and shares the `lax.composite` emission in
`zorch._composite.composite`.

The RLC is emitted as an unrolled fold (`acc += alpha_k * C_k`), not `jnp.dot`
/ `@`: a reduction in the marked body would split the region under the
single-kernel rewriter, and the unrolled fold keeps the accumulation
register-resident. Field addition is associative and exact, so the fold's order
is byte-equal to the dot it replaces.

An optional `live_width` declares a runtime row bound: rows at leading-axis
index >= the bound read as the field's zero. The bound is a VALUE (an s32
scalar operand), not a shape, so a consumer that keeps round-invariant operand
shapes reuses one compiled kernel across rounds while paying constraint-circuit
work for live rows only — the recognizing emitter computes
`out[x] = x < live ? body(x) : 0` and the decomposition mirrors that masked
form exactly, keeping marked and inlined paths byte-identical lane for lane.
The sumcheck marker's `num_real` expresses the same leading-row liveness as a
static composite ATTRIBUTE; here the bound is an OPERAND because it must vary
per round without changing the marker's fingerprint — static-vs-runtime is the
axis that picks the wire shape.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax

from zorch._composite import composite

CONSTRAINT_EVAL_MARKER = "zorch.constraint_eval"


def constraint_eval(
    eval_fn: Callable[[Array], Array],
    trace: Array,
    alpha: Array,
    *,
    live_width: Array | int | None = None,
    column_weights: Array | None = None,
    name: str = CONSTRAINT_EVAL_MARKER,
) -> Array:
    """Mark `sum_k alpha_k * eval_fn(trace)_k` as one `zorch.constraint_eval`.

    `eval_fn(trace)` must produce constraints in the trailing axis (`[..., K]`),
    matching `alpha`'s trailing length `K`; the result drops that axis. The K
    count and the alpha operand index ride along as composite attributes for the
    recognizing emitter; they are metadata, so the decomposition ignores them (it
    reads K from `alpha`'s static shape). An unrecognizing compiler inlines the
    decomposition to the identical result (see the module docstring).

    `live_width`, when given, bounds the result's leading axis at runtime: rows
    at index >= the bound are the field's zero. It must be a scalar `int32`
    (a Python int is converted) holding a non-negative value — the emitter
    compares indices unsigned, so a negative bound would diverge between the
    marked and inlined paths. It rides as operand 2 with its index declared in
    `live_width_operand_idx`; zkx hard-errors on a malformed declaration rather
    than silently falling back to the unbounded path.

    `column_weights`, when given, adds a per-row weighted column sum
    `sum_c trace[row, c] * column_weights[c]` to each row's accumulated value —
    a rank-1 vector with one weight per trace column. It rides as the trailing
    operand; the recognizing emitter folds the `trace @ column_weights` dot into
    the per-row accumulator (computed thread-locally while the row is already
    loaded), so no separate matmul kernel is launched. The marker keeps the dot
    in its body so the inlined / monolithic paths stay byte-identical. It
    requires `live_width` (the bounded path the emitter folds into) and is added
    AFTER the live mask: dead rows are zero, so a zero row's column term
    vanishes and the masked and unmasked forms agree. The term carries no
    proving-scheme meaning here (a consumer may use it for a column opening
    batch) — `zorch` stays scheme-agnostic.
    """
    num_constraints = alpha.shape[-1]
    if num_constraints < 1:
        raise ValueError(
            f"alpha must carry at least one coefficient, got {num_constraints}"
        )
    if column_weights is not None:
        # Rides as the trailing operand AFTER live_width — so it requires one,
        # keeping the operand order fixed (trace, alpha, live, weights) for the
        # decomposition's positional binding — and carries one weight per trace
        # column. Validate at the entry point so a mismatch fails loud here
        # rather than as a cryptic matmul trace error.
        if live_width is None:
            raise ValueError("column_weights requires live_width")
        num_cols = trace.shape[-1]
        if column_weights.ndim != 1 or column_weights.shape[0] != num_cols:
            raise ValueError(
                "column_weights must be rank-1 with one weight per trace column "
                f"({num_cols}), got shape {column_weights.shape}"
            )

    def decomposition(
        trace: Array,
        alpha: Array,
        live_width: Array | None = None,
        column_weights: Array | None = None,
        **_attrs: object,
    ) -> Array:
        constraints = eval_fn(trace)
        acc = constraints[..., 0] * alpha[..., 0]
        for k in range(1, num_constraints):
            acc = acc + constraints[..., k] * alpha[..., k]
        if live_width is not None:
            if acc.ndim == 0:
                raise ValueError("live_width needs a result with a leading row axis")
            # lax.select, not jnp.where — the single-kernel body rule; see
            # zorch/fusion.py's module docstring.
            rows = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            acc = lax.select(rows < live_width, acc, jnp.zeros_like(acc))
        if column_weights is not None:
            # Added AFTER the live mask, so the body root is
            # add(masked_fold, dot(trace, column_weights)) — the shape a
            # recognizing emitter folds into the per-row accumulator (hand-emitted
            # in-kernel; the inlined path runs the dot directly). A dot is allowed
            # in the bounded body, and dead rows are zero so the column term is
            # byte-neutral under the live mask.
            acc = acc + trace @ column_weights
        return acc

    operands: tuple[Array, ...] = (trace, alpha)
    attrs: dict[str, int] = {
        "num_constraints": num_constraints,
        "alpha_operand_idx": 1,
    }
    if live_width is not None:
        if isinstance(live_width, int):
            if live_width < 0:
                raise ValueError(f"live_width must be non-negative, got {live_width}")
            live: Array = jnp.asarray(live_width, jnp.int32)
        else:
            # asarray funnels any non-Array (float, numpy scalar) into the
            # shape/dtype rejection below instead of an opaque AttributeError.
            live = jnp.asarray(live_width)
            if live.shape != () or live.dtype != jnp.int32:
                raise ValueError(
                    "live_width must be a scalar int32 (the wire type zkx "
                    f"validates), got shape {live.shape} dtype {live.dtype}"
                )
        operands += (live,)
        attrs["live_width_operand_idx"] = 2
    if column_weights is not None:
        # Trailing operand; the emitter recognizes it structurally (the rank-1
        # operand of the body-root dot), so no operand-index attribute is needed.
        operands += (column_weights,)
    return composite(decomposition, *operands, name=name, **attrs)
