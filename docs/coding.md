# coding — linear codes

## Why this module exists

PCS, FRI, and Basefold all bottom out in one operation: a linear code's
`encode`. `coding` factors that shared concept the way `hash/` factors symmetric
primitives — a `LinearCode` Protocol seam plus concrete codes — so a proof
system depends on "a linear code", not on Reed-Solomon specifically. Reed-Solomon
is the first code; Brakedown and others drop into the same seam when a consumer
needs one. They are not carried before then: an unused second code is surface to
maintain, not value.

## The one rule: don't hand-roll the NTT

In the ZKX-patched JAX, `jax.lax.fft` **is** the native finite-field NTT — field
dtype in and out, a `generator` argument selecting the root, extension fields
auto-decomposed into prime-field NTTs, and the whole thing lowered to a single
fused kernel. Reed-Solomon `encode` therefore hands its evaluation to `lax.fft`
and adds only a pad and an optional scale around it.

A hand-rolled radix-2 butterfly would be the opposite: `log(n)` separate,
unfused kernels that the compiler cannot recognize as an NTT. It is slower and
it breaks fusion-by-construction. This is the same stance poseidon2 takes —
express the algebra in its natural form and let zkx lower it, rather than
fighting the compiler with a pattern it must re-discover. Any future code that
needs a transform follows this rule.

## The fold rides the encoder's domain

`fri_fold` is the FRI-family codeword fold every FRI-style scheme (FRI, Basefold,
WHIR, STARK) shares. It lives here, not in [`poly`](poly.md#why-the-shape),
because its x-coordinates must be the *same* `lax.fft` evaluation domain the
encoder used: `eval_domain` reads them off the NTT itself, so there is still no
field-generator table — the same stance as `encode`. It is distinct from
`poly.mle_fold` (the additive multilinear combine over the hypercube), folding a
*codeword* over the RS evaluation domain instead. The natural-order conjugate
layout makes `f(x)` / `f(−x)` a slice, not a gather, so the fold stays
fusion-friendly.

## Design rules

- **A code is an object, not a call.** `ReedSolomon` is a class, not a function,
  because the seam is polymorphic and introspectable: a consumer accepts any
  `LinearCode`, reads `block_len` / `message_len` / `dtype` (to size a Merkle
  tree, to reason about rate and soundness), and calls `.encode()` without
  knowing which code it holds. A bare function carries none of those attributes
  and satisfies no `isinstance(x, LinearCode)`. Construction is also where the
  code's parameters are fixed once and the coset ramp is precomputed. This
  mirrors `Poseidon2` (a configured operator), not a free function.

- **Not a pytree.** `ReedSolomon` holds a JAX array (`_coset_powers`) but is a
  configured *operator*, not threaded carry state — so, like `Poseidon2`, it is
  not registered as a pytree. The "register state containers as pytrees" rule is
  about immutably-threaded state (the `Transcript`), not operators constructed
  host-side and invoked. `vmap(rs.encode)` captures the array as a closure
  constant and needs no registration; register only if a consumer ever threads
  an instance through a transform.

- **Coset shift is the caller's, not the code's.** `coset_shift` is a field
  element passed in, defaulting to `None` (the plain two-adic subgroup). The
  code keeps no per-field generator table: `zk_dtypes.pfinfo` exposes
  two-adicity but no multiplicative generator, and it is the scheme (FRI/STARK)
  that knows which coset it needs disjoint from the trace domain. Domain policy
  lives with the consumer that has the context to choose it.

- **`encode` acts on the last axis.** Leading batch axes ride through untouched,
  so many polynomials — or a matrix of rows — encode in one call. This is part
  of the seam contract, and it keeps one `encode` lowering to one fused
  transform.

## Gotcha

The coset power ramp `[1, h, h², …]` is built with `jnp.cumprod` — `jnp.arange` /
`jnp.power` over a field dtype both fail (see the
[ZKX field-dtype gotchas](poly.md#zkx-field-dtype-gotchas)).

## Deliberately out of scope

Inverse/decode, explicit extension-field *message* support (only base field is
verified; extension data should ride the backend decomposer, untested),
`generator` interop conventions (gnark = 5, circom = 7), and every code other
than Reed-Solomon. Each lands with its first real consumer.

## Tests

`PYTHONPATH=. python zorch/coding/testing/reed_solomon_test.py`. The oracle is
kept independent of the encoder on purpose: it recovers the NTT domain from
`lax.fft` of an impulse (`NTT(e₁)_j = ωʲ`) and evaluates the message by Horner on
that domain, so a bug in pad-then-NTT cannot hide behind a round-trip that would
share the same code path.
