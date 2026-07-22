"""constraint_eval — mark an α-RLC'd constraint evaluation as one fused op.

Wraps `eval_fn(trace)` (a per-row constraint evaluation producing `[..., K]`)
followed by the random-linear-combination `sum_k alpha_k * C_k` in a
`frx.lax.composite` named `zorch.constraint_eval`. The marker preserves the
high-level "evaluate K constraints, fold them under one challenge vector" shape
so a recognizing compiler can emit a single kernel that accumulates
`alpha_k * C_k` incrementally and never materializes the `[..., K]` constraint
tensor; an unrecognizing compiler inlines the decomposition to the identical
result.

Agnostic: `eval_fn` is opaque — its body belongs to the caller — and the marker
and fold carry no proving-scheme or zkVM knowledge. Sibling of
`zorch.fusion::fused_region`, and shares the `lax.composite` emission in
`zorch._composite.composite`.

The RLC is emitted as an unrolled fold (`acc += alpha_k * C_k`), not `fnp.dot`
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

import frx.numpy as fnp
from frx import Array, lax

from zorch._composite import composite

CONSTRAINT_EVAL_MARKER = "zorch.constraint_eval"


def _scalar_int32_operand(value: Array | int, name: str) -> Array:
    """Normalize a scalar-int32 operand (`live_width`, `start_offset`): a Python
    int is converted and range-checked; anything else must already be a scalar
    int32 (the wire type XLA validates). asarray funnels a non-Array (float,
    numpy scalar) into the shape/dtype rejection instead of an opaque
    AttributeError."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return fnp.asarray(value, fnp.int32)
    arr = fnp.asarray(value)
    if arr.shape != () or arr.dtype != fnp.int32:
        raise ValueError(
            f"{name} must be a scalar int32 (the wire type XLA validates), "
            f"got shape {arr.shape} dtype {arr.dtype}"
        )
    return arr


def _fold_coeff_operand(value: Array | int, dtype: object) -> Array:
    """Normalize the runtime fold coefficient k to a rank-0 scalar of the
    trace's field. A Python int is cast to that field; an Array must already
    be a rank-0 scalar of it. Anything else fails loud — the decomposition
    and the backend emitter both evaluate `base + k*delta` in the trace's
    field, so a differently-typed k would corrupt silently downstream."""
    if isinstance(value, int):
        return fnp.asarray(value, dtype)
    if not hasattr(value, "ndim"):
        raise TypeError(
            f"fold_coeff must be a field scalar or int, got {type(value).__name__}"
        )
    if value.ndim != 0:
        raise ValueError(f"fold_coeff must be a scalar, got shape {value.shape}")
    if value.dtype != dtype:
        raise ValueError(
            f"fold_coeff must be the trace's field {dtype}, got {value.dtype}"
        )
    return value


def constraint_eval(
    eval_fn: Callable[..., Array] | None,
    trace: Array,
    alpha: Array,
    *,
    live_width: Array | int | None = None,
    start_offset: Array | int | None = None,
    window_rows: int | None = None,
    col_stride: Array | int | None = None,
    num_cols: int | None = None,
    delta: Array | None = None,
    fold_coeff: Array | int | None = None,
    column_weights: Array | None = None,
    max_monomials: int | None = None,
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

    An empty `alpha` (K = 0) is the constraint-free form: `eval_fn` must be
    None and `column_weights` must be given — the per-row value is just the
    masked column term (a consumer's lookup-only evaluation). The RLC fold
    degenerates to the field zero, which is exact, so the result is
    byte-identical to the bare masked dot.

    `live_width`, when given, bounds the result's leading axis at runtime: rows
    at index >= the bound are the field's zero. It must be a scalar `int32`
    (a Python int is converted) holding a non-negative value — the emitter
    compares indices unsigned, so a negative bound would diverge between the
    marked and inlined paths. It rides as operand 2 with its index declared in
    `live_width_operand_idx`; XLA hard-errors on a malformed declaration rather
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

    `col_stride` + `num_cols`, when given, treat the shared buffer as FLAT
    and column-major jagged: column `c` of the window is the rank-1 slice of
    `window_rows` elements starting at `start_offset + c*col_stride`, so
    evaluations of different heights pack into one buffer with no rectangle
    padding. `col_stride` is a runtime scalar `int32` like `start_offset`
    (its operand index rides in `col_stride_operand_idx`); `num_cols` is a
    Python `int` (a static attribute — it shapes the window). Requires
    `start_offset`.

    `delta` + `fold_coeff`, when given, evaluate the window of
    `trace + fold_coeff*delta` instead of `trace`'s: `delta` windows the SAME
    shared buffer at the same offsets (same shape and field as `trace`), and
    `fold_coeff` is a runtime rank-0 scalar of the trace's field, so one
    compiled kernel serves every coefficient value instead of a caller
    materializing the combined buffer per value. Requires `start_offset`;
    both must be given together.

    `column_weights`, when given, adds a per-row weighted column sum
    `sum_c trace[row, c] * column_weights[c]` to each row's accumulated value —
    a rank-1 vector with one weight per trace column. It rides as an operand
    after `live_width` (and before `aux_operands`, when given); the emitter
    identifies it structurally (not by index) and folds the
    `trace @ column_weights` dot into
    the per-row accumulator (computed thread-locally while the row is already
    loaded), so no separate matmul kernel is launched. The marker keeps the dot
    in its body so the inlined / monolithic paths stay byte-identical. It
    requires `live_width` (the bounded path the emitter folds into), and the
    live mask wraps the WHOLE per-row value including this term (mask-last):
    a window into a compact-packed shared buffer straddles the next
    evaluation's live rows, so the dot's dead-row contributions must zero out
    with everything else. The term carries no
    proving-scheme meaning here (a consumer may use it for a column opening
    batch) — `zorch` stays scheme-agnostic.

    `aux_operands`, when non-empty, are extra inputs the constraint reads beyond
    the trace: `eval_fn` is called as `eval_fn(trace, *aux_operands)` instead of
    1-ary `eval_fn(trace)`. They ride as the trailing operands with their indices
    declared in `aux_operand_idxs`. A constraint that depends on a runtime array
    its trace does not carry passes it here as a DECLARED operand rather than
    closing over it. That distinction is load-bearing under `frx.jit`: a
    closed-over array enters the composite decomposition as a Tracer constant,
    which `lax.composite` rejects (`UnexpectedTracerError`), whereas a declared
    operand traces cleanly. `zorch` reads no meaning from them; the recognizing
    emitter forwards them to the constraint body and the inlined path passes them
    to the same `eval_fn`, so marked and inlined stay byte-identical.
    """
    num_constraints = alpha.shape[-1]
    if num_constraints < 1:
        # A constraint-free marker is just the masked column term (a consumer's
        # lookup-only evaluation): the RLC fold degenerates to the field zero
        # and the column dot carries the whole per-row value. Without the dot
        # there is nothing to evaluate, so K = 0 alone stays an error.
        if column_weights is None:
            raise ValueError(
                "alpha must carry at least one coefficient unless "
                f"column_weights is given, got {num_constraints}"
            )
        if eval_fn is not None:
            raise ValueError(
                "eval_fn must be None when alpha is empty (no constraints to "
                "evaluate)"
            )
        if aux_operands:
            raise ValueError(
                "aux_operands require constraints (nothing reads them when "
                "alpha is empty)"
            )
    elif eval_fn is None:
        raise ValueError("eval_fn is required when alpha carries coefficients")
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
    if (col_stride is None) != (num_cols is None):
        raise ValueError("col_stride and num_cols must be given together")
    if col_stride is not None:
        # Jagged flat-trace window: `trace` is a 1-D buffer of column-major
        # per-chip segments; column c's rows live at
        # trace[start_offset + c*col_stride + row]. col_stride (the runtime
        # per-column segment length) and num_cols (the static column count)
        # size the [window_rows, num_cols] window the constraint sees.
        if start_offset is None:
            raise ValueError("col_stride requires start_offset (the window base)")
        if trace.ndim != 1:
            raise ValueError(
                f"col_stride windows a flat 1-D trace, got shape {trace.shape}"
            )
        if not isinstance(num_cols, int) or isinstance(num_cols, bool):
            raise ValueError(
                f"num_cols must be a Python int, got {type(num_cols).__name__}"
            )
        if num_cols < 1:
            raise ValueError(f"num_cols must be at least 1, got {num_cols}")
    if delta is not None or fold_coeff is not None:
        # Fold-inside: the per-row trace is `trace[row] + fold_coeff*delta[row]`
        # at a RUNTIME coefficient, so one compiled kernel serves every
        # coefficient value. Both window the SAME shared buffer, so
        # start_offset is required and delta must match trace's shape and
        # field.
        if delta is None or fold_coeff is None:
            raise ValueError("delta and fold_coeff must be given together")
        if start_offset is None:
            raise ValueError("delta requires start_offset (both window the buffer)")
        if delta.shape != trace.shape:
            raise ValueError(
                f"delta shape {delta.shape} must match trace shape {trace.shape}"
            )
        if delta.dtype != trace.dtype:
            raise ValueError(
                f"delta must be the trace's field {trace.dtype}, got {delta.dtype}"
            )
    if column_weights is not None:
        # Rides after live_width (so it requires one), keeping the optional
        # order fixed. Validate here so a mismatch fails loud, not as a cryptic
        # matmul trace error.
        if live_width is None:
            raise ValueError("column_weights requires live_width")
        want_cols = num_cols if col_stride is not None else trace.shape[-1]
        if column_weights.ndim != 1 or column_weights.shape[0] != want_cols:
            raise ValueError(
                "column_weights must be rank-1 with one weight per trace column "
                f"({want_cols}), got shape {column_weights.shape}"
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
    has_jagged = col_stride is not None and num_cols is not None and num_cols > 1
    has_delta = delta is not None
    has_weights = column_weights is not None
    n_aux = len(aux_operands)

    def decomposition(
        trace: Array,
        *optional: Array,
        **_attrs: object,
    ) -> Array:
        # *optional silently drops a surplus operand from the inlined path while
        # the marked kernel still carries it (a marked-vs-inlined divergence);
        # guard loud instead. alpha leads the tail only when K >= 1: an empty
        # alpha carries no data, and as an operand it would be constant-sunk
        # into the fused body with no user — an HLO-verifier error.
        has_alpha = num_constraints > 0
        n_expected = (
            has_alpha
            + has_live
            + has_offset
            + has_jagged
            + 2 * has_delta
            + has_weights
            + n_aux
        )
        if len(optional) != n_expected:
            raise TypeError(
                f"constraint_eval decomposition expected {n_expected} optional "
                f"operand(s), got {len(optional)} — an appended operand is not "
                "accounted for here"
            )
        tail = iter(optional)
        alpha_op = next(tail) if has_alpha else None
        live_width = next(tail) if has_live else None
        start_offset = next(tail) if has_offset else None
        col_stride = next(tail) if has_jagged else None
        delta = next(tail) if has_delta else None
        fold_coeff = next(tail) if has_delta else None
        column_weights = next(tail) if has_weights else None
        aux = tuple(tail)  # the remaining n_aux operands feed the constraint body
        if start_offset is not None:
            # trace is the tall shared buffer — rank-2 [TOTAL_ROWS, num_cols]
            # for the row-window path, or the flat 1-D jagged buffer for the
            # col_stride path; evaluate the constraint on its
            # [window_rows, num_cols] window at start_offset. window_rows and
            # num_cols are static (closed over), not operands. Slicing here,
            # before eval_fn runs, keeps everything below (the constraint, the
            # RLC, the live_width mask, the column dot) identical whether trace
            # arrived pre-windowed, tall, or flat.
            assert window_rows is not None  # validated above: start_offset requires it
            # allow_negative_indices=False so the offset drives the dynamic-slice
            # start directly. The default wraps it (compare<0 / add-size / select),
            # so the axis-0 start becomes that select — but the emitter binds the
            # window base as "the parameter driving a dynamic-slice start"
            # (ConstraintEvalStartOffsetIdx); a wrapped start hides the base and
            # drops the marker to the unbounded path. start_offset is non-negative
            # by construction, so the wrap is dead semantics — byte-neutral to drop.
            if col_stride is not None:
                assert num_cols is not None  # validated together at entry

                # Jagged: column c is a rank-1 slice at the affine start
                # `start_offset + c*col_stride`. Constant folding leaves
                # column 0 as the bare base and column 1 as add(o, H) by the
                # time the recognizer sees the body — it resolves the base
                # from column 0 and the stride from the add starts.
                def _win(t: Array) -> Array:
                    return fnp.stack(
                        [
                            lax.dynamic_slice_in_dim(
                                t,
                                start_offset + c * col_stride,
                                window_rows,
                                axis=0,
                                allow_negative_indices=False,
                            )
                            for c in range(num_cols)
                        ],
                        axis=1,
                    )

            elif num_cols is not None:
                # Single-column jagged chip: only column 0 exists, so there is
                # no stride evidence for a recognizer to resolve — the marker
                # omits the col_stride operand and the window degenerates to
                # the plain rank-1 base slice, reshaped to the [window_rows, 1]
                # the constraint sees.
                def _win(t: Array) -> Array:
                    return lax.dynamic_slice_in_dim(
                        t,
                        start_offset,
                        window_rows,
                        axis=0,
                        allow_negative_indices=False,
                    ).reshape(window_rows, 1)

            else:

                def _win(t: Array) -> Array:
                    return lax.dynamic_slice_in_dim(
                        t,
                        start_offset,
                        window_rows,
                        axis=0,
                        allow_negative_indices=False,
                    )

            trace = _win(trace)
            if delta is not None:
                # Fold-inside: eff = base_window + fold_coeff * delta_window. One
                # kernel serves every fold coefficient (runtime). Byte-identical
                # to windowing a pre-folded `base + fold_coeff*delta` trace.
                trace = trace + fold_coeff * _win(delta)
        if num_constraints == 0:
            # Constraint-free (lookup-only) form: the RLC fold is the field
            # zero and the column dot below carries the whole per-row value.
            # Field add of zero is exact, so `0 + trace @ w` is byte-identical
            # to the bare dot.
            acc = fnp.zeros(trace.shape[:-1], trace.dtype)
        else:
            assert eval_fn is not None  # validated at entry: K >= 1 requires it
            assert alpha_op is not None
            constraints = eval_fn(trace, *aux)
            acc = constraints[..., 0] * alpha_op[..., 0]
            for k in range(1, num_constraints):
                acc = acc + constraints[..., k] * alpha_op[..., k]
        if column_weights is not None:
            # A dot is allowed in the bounded body; a recognizing emitter folds
            # it into the per-row accumulator (hand-emitted in-kernel; the
            # inlined path runs the dot directly).
            acc = acc + trace @ column_weights
        if live_width is not None:
            if acc.ndim == 0:
                raise ValueError("live_width needs a result with a leading row axis")
            # lax.select, not fnp.where — the single-kernel body rule; see
            # zorch/fusion.py's module docstring.
            # The mask comes LAST — select(rows < live_width, rlc + dot, 0) —
            # so the column term's dead rows zero out too. A window into a
            # compact-packed shared buffer straddles the NEXT chip's live rows,
            # so the dead rows are NOT zero and an unmasked dot would leak them;
            # the live-bounded emitter kernel zeroes whole dead rows, and this
            # order is what matches it byte-for-byte.
            rows = lax.broadcasted_iota(fnp.int32, acc.shape, 0)
            acc = lax.select(rows < live_width, acc, fnp.zeros_like(acc))
        return acc

    # K = 0 omits the alpha operand entirely: an empty array carries no data,
    # and as an operand it would be constant-sunk into the fused body with no
    # user — an HLO-verifier error on the recognizing compiler.
    operands: tuple[Array, ...] = (trace, alpha) if num_constraints > 0 else (trace,)
    attrs: dict[str, int] = {"num_constraints": num_constraints}
    if num_constraints > 0:
        attrs["alpha_operand_idx"] = 1
    if live_width is not None:
        attrs["live_width_operand_idx"] = len(operands)
        operands += (_scalar_int32_operand(live_width, "live_width"),)
    if start_offset is not None:
        operands += (_scalar_int32_operand(start_offset, "start_offset"),)
        # Computed from len(operands), not hardcoded: start_offset requires
        # live_width so it lands at 3 today, but this stays correct if the
        # optional-operand order ever grows a slot between them.
        attrs["start_offset_operand_idx"] = len(operands) - 1
        assert window_rows is not None  # validated above: start_offset requires it
        attrs["window_rows"] = window_rows
    if col_stride is not None and num_cols is not None and num_cols > 1:
        # Jagged flat-trace window: the runtime per-column segment length,
        # riding right after start_offset; num_cols is static like
        # window_rows. Advisory — the emitter resolves the stride structurally
        # (the base from the bare column-0 start, the stride from the add
        # starts). A single-column chip omits the operand entirely: no add
        # start exists to resolve a stride from, and the plain rank-1 window
        # needs none.
        operands += (_scalar_int32_operand(col_stride, "col_stride"),)
        attrs["col_stride_operand_idx"] = len(operands) - 1
        attrs["num_cols"] = num_cols
    if delta is not None:
        # Fold-inside: delta rides right after start_offset, then the runtime
        # coefficient. The emitter resolves both structurally (delta is the second
        # tall trace, fold_coeff the rank-0 field scalar feeding a multiply — see
        # ResolveFoldTrace), so their indices ride only as advisory attributes.
        operands += (delta, _fold_coeff_operand(fold_coeff, trace.dtype))
        attrs["delta_operand_idx"] = len(operands) - 2
        attrs["fold_coeff_operand_idx"] = len(operands) - 1
    if column_weights is not None:
        # The emitter recognizes it structurally (the rank-1 operand of the
        # body-root dot), so no operand-index attribute is needed.
        operands += (column_weights,)
    if max_monomials is not None:
        # Perf hint only: the recognizing emitter distributes each cone into up
        # to this many monomials (a flatter, better-occupancy body, byte-
        # identical to the cone body). The decomposition ignores it; an
        # unrecognizing backend is unaffected. `<= 0` disables (keeps the cone
        # body), matching the emitter's gate.
        if not isinstance(max_monomials, int) or isinstance(max_monomials, bool):
            got = type(max_monomials).__name__
            raise ValueError(f"max_monomials must be a Python int, got {got}")
        attrs["cone_program_max_monomials"] = max_monomials
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
