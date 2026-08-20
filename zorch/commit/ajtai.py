"""Module-lattice commitments beside the Merkle tree — Ajtai, and BDLOP for hiding.

The lattice counterpart of `merkle`: scheme-agnostic, with no domain
separator, no transcript, and no challenge — those belong to the consumer's
commitment scheme. What lives here is the algebra and the opening predicate:

- **Ajtai**: `t = A·s` over `Z_q[X]/(X^d+1)` with a short witness `s` —
  binding under MSIS, and *additively homomorphic*
  (`commit(s1) + commit(s2) == commit(s1 + s2)`), which is the property a
  folding consumer needs and a hash tree cannot give. The homomorphism is
  part of the tested contract, not an accident.
- **BDLOP** (eprint 2016/997): the hiding extension — `t0 = B0·r`,
  `t1 = B1·r + m` — opened by revealing `(m, r)` and re-running the algebra.

Commitment is one `matvec` in the NTT domain per equation, so it traces and
batches exactly like the ring ops it is made of (lattice-frx's `RnsRing`).
Verification is a host-boundary predicate on purpose: the opening bound is an
ℓ∞ norm over the *balanced lift* of the witness, and lattice-frx pins lifts
and norms to the host (`rns.reconstruct_centered` + `norms.linf`) because no
lane holds them. The recomputation inside `verify` is the same traced
`matvec` as `commit`; only the comparison and the norm materialise.

Randomness stance: matrices, witnesses, and randomness arrive as ring
elements the caller built — nothing here samples. A cryptographic seed→matrix
CRS expansion belongs to the consumer until the raw-XOF transcript seam and
lattice-frx's uniform-from-bytes sampler land; the tests use a seeded numpy
`Generator` and say so.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import frx
import numpy as np
from lattice_frx import norms, rns
from lattice_frx.ring import Coeff, Eval, RnsRing


@dataclass(frozen=True)
class AjtaiParams:
    """`rows × cols` module shape over `ring`, with the ℓ∞ opening bound.

    Static configuration, not a pytree: the ring carries the moduli and the
    bound is a plain host integer read only at the verification boundary.
    Binding strength (MSIS) is the consumer's parameter choice; this seam
    only enforces the shapes and the bound it is given.
    """

    ring: RnsRing
    rows: int
    cols: int
    beta_inf: int


class AjtaiCommitment:
    """`commit(A, s) = A·s` and its opening predicate."""

    def __init__(self, params: AjtaiParams) -> None:
        self.params = params

    def commit(self, matrix: Eval, witness: Eval) -> Eval:
        """One `matvec`: `[rows, cols, d] × [cols, d] → [rows, d]`, traced."""
        p = self.params
        _require_lead("commit: matrix", matrix, (p.rows, p.cols))
        _require_lead("commit: witness", witness, (p.cols,))
        return p.ring.matvec(matrix, witness)

    def verify(self, matrix: Eval, commitment: Eval, opening: Coeff) -> bool:
        """The opening predicate: `‖opening‖∞ ≤ β` on the balanced lift, and
        the opening re-commits to `commitment`. Host boundary by design —
        see the module docstring."""
        p = self.params
        if not isinstance(opening, Coeff):
            raise TypeError(
                f"verify: opening must be Coeff (the norm is a coefficient-"
                f"domain notion), got {type(opening).__name__}"
            )
        if not _within_bound(p.ring, opening, p.beta_inf):
            return False
        return _equal(self.commit(matrix, p.ring.ntt(opening)), commitment)


@partial(frx.tree_util.register_dataclass, data_fields=["t0", "t1"], meta_fields=[])
@dataclass(frozen=True)
class BdlopPair:
    """A BDLOP commitment: the binding half `t0` and the message half `t1`.

    A pytree so a consumer can hash it into a transcript, batch it under
    `vmap`, or fold it like any other pair of ring elements."""

    t0: Eval
    t1: Eval


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["message", "randomness"],
    meta_fields=[],
)
@dataclass(frozen=True)
class BdlopOpening:
    """What `verify` consumes: the message and the randomness, both in the
    coefficient domain — the domain norms and "reveal" mean anything in."""

    message: Coeff
    randomness: Coeff


@dataclass(frozen=True)
class BdlopParams:
    """Shapes for `t0 = B0·r` (`rows × randomness_cols`) and
    `t1 = B1·r + m` (`messages × randomness_cols`), plus the ℓ∞ bound on `r`.
    """

    ring: RnsRing
    rows: int
    randomness_cols: int
    messages: int
    beta_inf: int


class BdlopCommitment:
    """BDLOP commit/verify. The bound applies to the randomness only — the
    message is unconstrained, which is what makes the scheme hiding rather
    than merely binding."""

    def __init__(self, params: BdlopParams) -> None:
        self.params = params

    def commit(
        self, b0: Eval, b1: Eval, message: Coeff, randomness: Coeff
    ) -> BdlopPair:
        p = self.params
        _require_lead("commit: b0", b0, (p.rows, p.randomness_cols))
        _require_lead("commit: b1", b1, (p.messages, p.randomness_cols))
        _require_lead("commit: message", message, (p.messages,))
        _require_lead("commit: randomness", randomness, (p.randomness_cols,))
        r = p.ring.ntt(randomness)
        return BdlopPair(
            t0=p.ring.matvec(b0, r),
            t1=p.ring.add(p.ring.matvec(b1, r), p.ring.ntt(message)),
        )

    def verify(
        self, b0: Eval, b1: Eval, commitment: BdlopPair, opening: BdlopOpening
    ) -> bool:
        p = self.params
        if not _within_bound(p.ring, opening.randomness, p.beta_inf):
            return False
        recomputed = self.commit(b0, b1, opening.message, opening.randomness)
        return _equal(recomputed.t0, commitment.t0) and _equal(
            recomputed.t1, commitment.t1
        )


def _require_lead(name: str, element: Coeff | Eval, lead: tuple[int, ...]) -> None:
    """The element's leading (batch/module) axes against the params' shape —
    per-limb `d` and the ring's limb count are the ring ops' own checks."""
    got = tuple(element.limbs[0].shape[:-1])
    if got != lead:
        raise ValueError(f"{name}: leading axes {got}, want {lead}")


def _elements_as_host(ring: RnsRing, batched: Coeff) -> Iterator[np.ndarray]:
    """Each element of a `[k, d]` coefficient batch as a `(limbs, d)` uint64
    host array — the shape `rns`'s reconstructions consume."""
    obj_limbs = [np.asarray(limb).astype(object) for limb in batched.limbs]
    for j in range(obj_limbs[0].shape[0]):
        yield np.array(
            [[int(v) for v in limb[j]] for limb in obj_limbs], dtype=np.uint64
        )


def _within_bound(ring: RnsRing, batched: Coeff, beta_inf: int) -> bool:
    """`‖·‖∞ ≤ β` for every element, over the full centered reconstruction —
    a single-limb lift would accept an opening whose other limbs disagree."""
    return all(
        norms.linf(rns.reconstruct_centered(host, ring.q_moduli)) <= beta_inf
        for host in _elements_as_host(ring, batched)
    )


def _equal(a: Eval, b: Eval) -> bool:
    """Exact per-limb equality, materialised to the host — verification is a
    boundary predicate, not a traced op."""
    return all(
        np.array_equal(np.asarray(x).astype(object), np.asarray(y).astype(object))
        for x, y in zip(a.limbs, b.limbs)
    )
