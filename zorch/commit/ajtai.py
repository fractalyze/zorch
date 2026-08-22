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
- **ABDLOP** (eprint 2022/284, eq. 5): the two halves combined in one scheme —
  `t_A = A1·s1 + A2·s2` binds a small-norm witness `s1` Ajtai-style while
  `t_B = B·s2 + m` carries unrestricted-coefficient messages BDLOP-style,
  both under the same randomness `s2`. This is the commitment LNP proofs
  (`zorch/lnp`) open and prove statements about, and it lives on lattice-frx's
  *partial-split* host ring (`q ≡ 5 (mod 8)`) rather than the NTT ring the
  other two use: LNP soundness extraction divides by challenge differences,
  which Lemma 2.6 grounds in the partial-split factorization — and no modulus
  satisfies both ring modes.

For Ajtai and BDLOP, commitment is one `matvec` in the NTT domain per
equation, so it traces and batches exactly like the ring ops it is made of
(lattice-frx's `RnsRing`). ABDLOP is host-boundary throughout instead: the
partial-split ring pins products to the host, so its commitment is a host
matvec composed from the ring's own `mul`/`add` over the `(limbs, d)` uint64
contract, with module vectors stacked as `(k, limbs, d)`.
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
from lattice_frx.split_ring import HostSplitRing


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


@dataclass(frozen=True)
class AbdlopPair:
    """An ABDLOP commitment: the Ajtai half `t_a` and the message half `t_b`.

    Host `(k, limbs, d)` uint64 stacks, not a pytree — the partial-split
    ring is host-boundary, so there is nothing to trace or batch here."""

    t_a: np.ndarray
    t_b: np.ndarray


class AbdlopCommitment:
    """ABDLOP commit/verify: `t_a = A1·s1 + A2·s2` over `rows × (s1_cols +
    randomness_cols)`, `t_b = B·s2 + m` over `messages × randomness_cols`.

    Two ℓ∞ bounds because the two witness halves are different objects in
    the scheme: `beta1_inf` bounds the committed witness `s1` (the Ajtai
    half's message), `beta2_inf` the randomness `s2`; the message `m` is
    unconstrained, as in BDLOP. Configuration rides the constructor, as in
    the siblings — binding (MSIS) and hiding (MLWE) strength are the
    consumer's parameter choice."""

    def __init__(
        self,
        ring: HostSplitRing,
        rows: int,
        s1_cols: int,
        randomness_cols: int,
        messages: int,
        beta1_inf: int,
        beta2_inf: int,
    ) -> None:
        self.ring = ring
        self.rows = rows
        self.s1_cols = s1_cols
        self.randomness_cols = randomness_cols
        self.messages = messages
        self.beta1_inf = beta1_inf
        self.beta2_inf = beta2_inf

    def commit(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
    ) -> AbdlopPair:
        self._require_stack("commit: a1", a1, (self.rows, self.s1_cols))
        self._require_stack("commit: a2", a2, (self.rows, self.randomness_cols))
        self._require_stack("commit: b", b, (self.messages, self.randomness_cols))
        self._require_stack("commit: s1", s1, (self.s1_cols,))
        self._require_stack("commit: s2", s2, (self.randomness_cols,))
        self._require_stack("commit: message", message, (self.messages,))
        ring = self.ring
        return AbdlopPair(
            t_a=ring.add(ring.matvec(a1, s1), ring.matvec(a2, s2)),
            t_b=ring.add(ring.matvec(b, s2), message),
        )

    def verify(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b: np.ndarray,
        commitment: AbdlopPair,
        s1: np.ndarray,
        s2: np.ndarray,
        message: np.ndarray,
    ) -> bool:
        """The exact opening predicate: `‖s1‖∞ ≤ β1`, `‖s2‖∞ ≤ β2` on the
        balanced lifts, and the triple re-commits to `commitment`. The
        *relaxed* opening (with a challenge factor) is the LNP protocol
        layer's notion, not this seam's — same discipline as the siblings
        keeping challenges out of the commitment algebra."""
        if not _within_bound_host(self.ring, s1, self.beta1_inf):
            return False
        if not _within_bound_host(self.ring, s2, self.beta2_inf):
            return False
        recomputed = self.commit(a1, a2, b, s1, s2, message)
        return _equal_host(recomputed.t_a, commitment.t_a) and _equal_host(
            recomputed.t_b, commitment.t_b
        )

    def _require_stack(self, name: str, arr: np.ndarray, lead: tuple[int, ...]) -> None:
        """The `_require_lead` of the host-array convention: leading
        (module) axes against the scheme's declared shape, with the
        trailing `(limbs, d)` fixed by the ring."""
        want = lead + (len(self.ring.q_moduli), self.ring.d)
        got = arr.shape
        if got != want:
            raise ValueError(f"{name}: shape {got}, want {want}")


def _linf_within(host: np.ndarray, q_moduli: tuple[int, ...], beta_inf: int) -> bool:
    """The one norm policy every opening predicate here shares: ℓ∞ over the
    *centered* reconstruction. Named once because the two ported
    reconstructions differ at exactly `Q/2` (see lattice-frx's `rns.py`),
    so which one the bound reads is a pinned choice, not a detail."""
    return norms.linf(rns.reconstruct_centered(host, q_moduli)) <= beta_inf


def _within_bound_host(ring: HostSplitRing, stacked: np.ndarray, beta_inf: int) -> bool:
    """`‖·‖∞ ≤ β` over the full centered reconstruction, the host-array
    twin of `_within_bound`: the `(k, limbs, d)` batch flattens into one
    `(limbs, k·d)` reconstruction because ℓ∞ of a batch is the ℓ∞ of its
    concatenation. The domain gate has no work to do here — the
    partial-split ring is coefficient-domain by construction."""
    host = np.transpose(stacked, (1, 0, 2)).reshape(len(ring.q_moduli), -1)
    return _linf_within(host, ring.q_moduli, beta_inf)


def _equal_host(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality on host stacks. `np.array_equal` already refuses a
    shape mismatch (a truncated RNS chain included); the dtype check keeps
    the same never-raise role as `_equal`'s — a different dtype is a
    different-ring `False`, not an error."""
    return a.dtype == b.dtype and bool(np.array_equal(a, b))


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
    return _linf_within(host, ring.q_moduli, beta_inf)


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
