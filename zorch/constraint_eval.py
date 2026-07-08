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
The jagged sumcheck marker's `row_counts` attribute expresses the same
leading-row liveness as a static composite ATTRIBUTE; here the bound is an OPERAND
because it must vary per round without changing the marker's fingerprint —
static-vs-runtime is the axis that picks the wire shape.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, lax

from zorch._composite import composite

CONSTRAINT_EVAL_MARKER = "zorch.constraint_eval"


def _scalar_int32_operand(value: Array | int, name: str) -> Array:
    """Normalize a scalar-int32 operand (`live_width`, `start_offset`): a Python
    int is converted and range-checked; anything else must already be a scalar
    int32 (the wire type zkx validates). asarray funnels a non-Array (float,
    numpy scalar) into the shape/dtype rejection instead of an opaque
    AttributeError."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return jnp.asarray(value, jnp.int32)
    arr = jnp.asarray(value)
    if arr.shape != () or arr.dtype != jnp.int32:
        raise ValueError(
            f"{name} must be a scalar int32 (the wire type zkx validates), "
            f"got shape {arr.shape} dtype {arr.dtype}"
        )
    return arr


def constraint_eval(
    eval_fn: Callable[..., Array],
    trace: Array,
    alpha: Array,
    *,
    live_width: Array | int | None = None,
    start_offset: Array | int | None = None,
    window_rows: int | None = None,
    column_weights: Array | None = None,
    aux_operands: tuple[Array, ...] = (),
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

    `start_offset`, when given, treats `trace` as a TALL shared buffer
    `[TOTAL_ROWS, num_cols, ...]` and windows it: the decomposition slices the
    `window_rows`-row window starting at row `start_offset` BEFORE `eval_fn`
    runs, so the constraint, the α-RLC, the `live_width` mask, and the column
    dot below all see the window exactly as if it had been passed as `trace`
    directly. It requires `live_width` (the window's live-height bound) and
    `window_rows` (a Python `int` — the window's static row count, so the
    emitted kernel keeps a fixed shape across calls at different offsets). It
    is a scalar `int32` like `live_width` (a Python int is converted) and
    rides as the operand immediately after `live_width`, with its index
    declared in `start_offset_operand_idx` and the window height in the
    static `window_rows` attribute; a recognizing emitter reads the window in
    place from the shared buffer without materializing a copy, while the
    decomposition's `lax.dynamic_slice` keeps the inlined path
    byte-identical.

    `column_weights`, when given, adds a per-row weighted column sum
    `sum_c trace[row, c] * column_weights[c]` to each row's accumulated value —
    a rank-1 vector with one weight per trace column. It rides as an operand
    after `live_width` (and before `aux_operands`, when given); the emitter
    identifies it structurally (not by index) and folds the
    `trace @ column_weights` dot into
    the per-row accumulator (computed thread-locally while the row is already
    loaded), so no separate matmul kernel is launched. The marker keeps the dot
    in its body so the inlined / monolithic paths stay byte-identical. It
    requires `live_width` (the bounded path the emitter folds into) and is added
    AFTER the live mask: dead rows are zero, so a zero row's column term
    vanishes and the masked and unmasked forms agree. The term carries no
    proving-scheme meaning here (a consumer may use it for a column opening
    batch) — `zorch` stays scheme-agnostic.

    `aux_operands`, when non-empty, are extra inputs the constraint reads beyond
    the trace: `eval_fn` is called as `eval_fn(trace, *aux_operands)` instead of
    1-ary `eval_fn(trace)`. They ride as the trailing operands with their indices
    declared in `aux_operand_idxs`. A constraint that depends on a runtime array
    its trace does not carry passes it here as a DECLARED operand rather than
    closing over it. That distinction is load-bearing under `jax.jit`: a
    closed-over array enters the composite decomposition as a Tracer constant,
    which `lax.composite` rejects (`UnexpectedTracerError`), whereas a declared
    operand traces cleanly. `zorch` reads no meaning from them; the recognizing
    emitter forwards them to the constraint body and the inlined path passes them
    to the same `eval_fn`, so marked and inlined stay byte-identical.
    """
    num_constraints = alpha.shape[-1]
    if num_constraints < 1:
        raise ValueError(
            f"alpha must carry at least one coefficient, got {num_constraints}"
        )
    if start_offset is not None:
        # Rides immediately after live_width (so it requires one — the
        # window's live-height bound), and requires window_rows (the static
        # output height, since it sizes the emitted kernel's fixed shape).
        # Validate here so a mismatch fails loud, not as a cryptic
        # dynamic_slice trace error.
        if live_width is None:
            raise ValueError("start_offset requires live_width")
        if window_rows is None:
            raise ValueError(
                "start_offset requires window_rows (the static window height)"
            )
        # window_rows is a static slice size AND a static attr, so validate it
        # loud at the seam like the other optionals (bool is an int subclass,
        # but a bool window height is a bug, not a 0/1 height).
        if not isinstance(window_rows, int) or isinstance(window_rows, bool):
            raise ValueError(
                f"window_rows must be a Python int, got {type(window_rows).__name__}"
            )
        if not 0 < window_rows <= trace.shape[0]:
            raise ValueError(
                "window_rows must be in 1..the trace height "
                f"({trace.shape[0]}), got {window_rows}"
            )
    elif window_rows is not None:
        # window_rows alone would size a window that never gets sliced — a silent
        # no-op. Fail loud, mirroring the other requires-a-companion checks.
        raise ValueError(
            "window_rows requires start_offset (it sizes the offset window)"
        )
    if column_weights is not None:
        # Rides after live_width (so it requires one), keeping the optional
        # order fixed. Validate here so a mismatch fails loud, not as a cryptic
        # matmul trace error.
        if live_width is None:
            raise ValueError("column_weights requires live_width")
        num_cols = trace.shape[-1]
        if column_weights.ndim != 1 or column_weights.shape[0] != num_cols:
            raise ValueError(
                "column_weights must be rank-1 with one weight per trace column "
                f"({num_cols}), got shape {column_weights.shape}"
            )
    if aux_operands is None or hasattr(aux_operands, "ndim"):
        # None (a `pv=None`-style migration slip) or a bare array (which would
        # splat into per-element scalars) — want a sequence of whole arrays.
        raise ValueError("aux_operands must be a tuple of arrays (use () for none)")
    aux_operands = tuple(aux_operands)  # accept any sequence; normalize to tuple

    # Optional operands are independent, so they don't form a fixed prefix;
    # bind them by presence in a known order (live, weights, then aux) rather
    # than by defaulted params, which would mis-bind aux to the weights slot.
    has_live = live_width is not None
    has_offset = start_offset is not None
    has_weights = column_weights is not None
    n_aux = len(aux_operands)

    def decomposition(
        trace: Array,
        alpha: Array,
        *optional: Array,
        **_attrs: object,
    ) -> Array:
        # *optional silently drops a surplus operand from the inlined path while
        # the marked kernel still carries it (a marked-vs-inlined divergence);
        # guard loud instead.
        n_expected = has_live + has_offset + has_weights + n_aux
        if len(optional) != n_expected:
            raise TypeError(
                f"constraint_eval decomposition expected {n_expected} optional "
                f"operand(s), got {len(optional)} — an appended operand is not "
                "accounted for here"
            )
        tail = iter(optional)
        live_width = next(tail) if has_live else None
        start_offset = next(tail) if has_offset else None
        column_weights = next(tail) if has_weights else None
        aux = tuple(tail)  # the remaining n_aux operands feed the constraint body
        if start_offset is not None:
            # trace is the tall shared buffer [TOTAL_ROWS, num_cols(, ...)];
            # evaluate the constraint on the window_rows-row window (axis 0) at
            # start_offset. window_rows is static (closed over), not an operand.
            # Slicing here, before eval_fn runs, keeps everything below (the
            # constraint, the RLC, the live_width mask, the column dot) identical
            # whether trace arrived pre-windowed or tall.
            assert window_rows is not None  # validated above: start_offset requires it
            trace = lax.dynamic_slice_in_dim(trace, start_offset, window_rows, axis=0)
        constraints = eval_fn(trace, *aux)
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
        operands += (_scalar_int32_operand(live_width, "live_width"),)
        attrs["live_width_operand_idx"] = 2
    if start_offset is not None:
        operands += (_scalar_int32_operand(start_offset, "start_offset"),)
        # Computed from len(operands), not hardcoded: start_offset requires
        # live_width so it lands at 3 today, but this stays correct if the
        # optional-operand order ever grows a slot between them.
        attrs["start_offset_operand_idx"] = len(operands) - 1
        assert window_rows is not None  # validated above: start_offset requires it
        attrs["window_rows"] = window_rows
    if column_weights is not None:
        # The emitter recognizes it structurally (the rank-1 operand of the
        # body-root dot), so no operand-index attribute is needed.
        operands += (column_weights,)
    # Two emit sites, not one: the list-valued aux_operand_idxs can only be
    # passed as a named kwarg (a dict-typed attrs unpack would collide with
    # composite's typed `version` param under mypy), and a named kwarg cannot be
    # conditional within a single call. The no-aux branch keeps the attribute
    # off entirely, which the emitter routes on.
    if not aux_operands:
        return composite(decomposition, *operands, name=name, **attrs)
    # Trailing operands at dynamic indices; the emitter finds them by these.
    aux_operand_idxs = list(range(len(operands), len(operands) + n_aux))
    operands += aux_operands
    return composite(
        decomposition, *operands, name=name, aux_operand_idxs=aux_operand_idxs, **attrs
    )
