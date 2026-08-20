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

from dataclasses import dataclass
from functools import partial

import frx
import numpy as np
from lattice_frx import norms, rns
from lattice_frx.ring import Coeff, Eval, RnsRing


class AjtaiCommitment:
    """`commit(A, s) = A·s` and its opening predicate, over a `rows × cols`
    module with an ℓ∞ opening bound.

    Configuration rides the constructor, like the sibling `MerkleTree`; the
    ring carries the moduli and `beta_inf` is a plain host integer read only
    at the verification boundary. Binding strength (MSIS) is the consumer's
    parameter choice — this seam only enforces the shapes and the bound it
    is given.
    """

    def __init__(self, ring: RnsRing, rows: int, cols: int, beta_inf: int) -> None:
        self.ring = ring
        self.rows = rows
        self.cols = cols
        self.beta_inf = beta_inf

    def commit(self, matrix: Eval, witness: Eval) -> Eval:
        """One `matvec`: `[rows, cols, d] × [cols, d] → [rows, d]`, traced."""
        _require_lead("commit: matrix", matrix, (self.rows, self.cols))
        _require_lead("commit: witness", witness, (self.cols,))
        return self.ring.matvec(matrix, witness)

    def verify(self, matrix: Eval, commitment: Eval, opening: Coeff) -> bool:
        """The opening predicate: `‖opening‖∞ ≤ β` on the balanced lift, and
        the opening re-commits to `commitment`. Host boundary by design —
        see the module docstring."""
        if not _within_bound(self.ring, opening, self.beta_inf):
            return False
        return _equal(self.commit(matrix, self.ring.ntt(opening)), commitment)


@partial(frx.tree_util.register_dataclass, data_fields=["t0", "t1"], meta_fields=[])
@dataclass(frozen=True)
class BdlopPair:
    """A BDLOP commitment: the binding half `t0` and the message half `t1`.

    A pytree so a consumer can hash it into a transcript, batch it under
    `vmap`, or fold it like any other pair of ring elements."""

    t0: Eval
    t1: Eval


class BdlopCommitment:
    """BDLOP commit/verify: `t0 = B0·r` over `rows × randomness_cols`,
    `t1 = B1·r + m` over `messages × randomness_cols`.

    The ℓ∞ bound applies to the randomness only — the message is
    unconstrained, which is what makes the scheme hiding rather than merely
    binding. Configuration rides the constructor, as in `AjtaiCommitment`."""

    def __init__(
        self,
        ring: RnsRing,
        rows: int,
        randomness_cols: int,
        messages: int,
        beta_inf: int,
    ) -> None:
        self.ring = ring
        self.rows = rows
        self.randomness_cols = randomness_cols
        self.messages = messages
        self.beta_inf = beta_inf

    def commit(
        self, b0: Eval, b1: Eval, message: Coeff, randomness: Coeff
    ) -> BdlopPair:
        _require_lead("commit: b0", b0, (self.rows, self.randomness_cols))
        _require_lead("commit: b1", b1, (self.messages, self.randomness_cols))
        _require_lead("commit: message", message, (self.messages,))
        _require_lead("commit: randomness", randomness, (self.randomness_cols,))
        r = self.ring.ntt(randomness)
        return BdlopPair(
            t0=self.ring.matvec(b0, r),
            t1=self.ring.add(self.ring.matvec(b1, r), self.ring.ntt(message)),
        )

    def verify(
        self,
        b0: Eval,
        b1: Eval,
        commitment: BdlopPair,
        message: Coeff,
        randomness: Coeff,
    ) -> bool:
        """An opening is the revealed `(message, randomness)` pair, taken
        bare — mirroring `commit`'s own signature and the Ajtai sibling."""
        if not _within_bound(self.ring, randomness, self.beta_inf):
            return False
        recomputed = self.commit(b0, b1, message, randomness)
        return _equal(recomputed.t0, commitment.t0) and _equal(
            recomputed.t1, commitment.t1
        )


def _require_lead(name: str, element: Coeff | Eval, lead: tuple[int, ...]) -> None:
    """The element's leading (batch/module) axes against the scheme's declared
    module shape — `matvec` checks only internal mat/vec consistency, so a
    well-formed matvec of the wrong module shape would sail through it."""
    got = tuple(element.limbs[0].shape[:-1])
    if got != lead:
        raise ValueError(f"{name}: leading axes {got}, want {lead}")


def _within_bound(ring: RnsRing, batched: Coeff, beta_inf: int) -> bool:
    """`‖·‖∞ ≤ β` over the full centered reconstruction of every coefficient.

    `Coeff` only — a norm of NTT values is a bug wearing a plausible shape,
    and this guard is the one domain gate both schemes' verifies share. The
    batch flattens into one reconstruction because ℓ∞ of a batch is the ℓ∞
    of its concatenation, and the reconstruction is per-coefficient; the
    full-chain lift (not limb 0's) is deliberate — a single-limb lift would
    accept an opening whose other limbs disagree. `astype(np.uint64)` is the
    field dtype's own exact canonical conversion, so the host `(limbs, N)`
    contract is composed from the dtype layer rather than re-derived.
    """
    if not isinstance(batched, Coeff):
        raise TypeError(
            f"verify: opening values must be Coeff (the norm is a "
            f"coefficient-domain notion), got {type(batched).__name__}"
        )
    host = np.stack(
        [np.asarray(limb).astype(np.uint64).reshape(-1) for limb in batched.limbs]
    )
    return norms.linf(rns.reconstruct_centered(host, ring.q_moduli)) <= beta_inf


def _equal(a: Eval, b: Eval) -> bool:
    """Exact per-limb equality on the host.

    A different limb count is a different RNS chain, i.e. a different ring:
    never equal, checked first so a shorter commitment with matching prefix
    limbs cannot pass by `zip` truncation. The dtype check has the same
    never-raise role — comparing across two field dtypes is a different-ring
    `False`, not an error — and keeps the comparison on the vectorized field
    arrays instead of boxing every coefficient into a Python int.
    """
    if len(a.limbs) != len(b.limbs):
        return False
    for x, y in zip(a.limbs, b.limbs):
        hx, hy = np.asarray(x), np.asarray(y)
        if hx.dtype != hy.dtype or not np.array_equal(hx, hy):
            return False
    return True
