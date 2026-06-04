"""koalabear-16 Merkle stack fixture — TEST only.

The Poseidon2(koalabear-16) -> Sponge -> Compression -> MerkleTree wiring used by
merkle_test. Lives under commit/ rather than poseidon2/testing so the poseidon2
fixture stays free of a back-dependency on the commit layer.
"""

from __future__ import annotations

from zorch.commit.merkle import MerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.testing.koalabear16 import koalabear16_perm
from zorch.hash.sponge import Sponge, SpongeParams


def koalabear16_merkle(
    rate: int = 8, out: int = 8, arity: int = 2, chunk: int = 8
) -> tuple[Sponge, Compression, MerkleTree]:
    """`(sponge, compressor, tree)` over the golden koalabear-16 permutation."""
    perm = koalabear16_perm()
    sponge = Sponge(perm, SpongeParams(rate=rate, out=out))
    comp = Compression(perm, CompressionParams(arity=arity, chunk=chunk))
    return sponge, comp, MerkleTree(sponge, comp)
