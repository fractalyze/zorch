"""Permutation Protocol is structural and hash-agnostic."""
import jax.numpy as jnp
from zorch.hash.permutation import Permutation


class _Id:
    width = 3
    dtype = jnp.int32
    def permute(self, state):
        return state


def test_duck_typed_impl_satisfies_protocol():
    assert isinstance(_Id(), Permutation)


def test_consumer_reads_width_and_dtype_without_naming_a_hash():
    p = _Id()
    state = jnp.zeros(p.width, dtype=p.dtype)   # sponge-style allocation
    assert state.shape == (3,)
    assert jnp.array_equal(p.permute(state), state)


if __name__ == "__main__":
    test_duck_typed_impl_satisfies_protocol()
    test_consumer_reads_width_and_dtype_without_naming_a_hash()
    print("ok")
