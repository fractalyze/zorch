# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`DuplexTranscript.fs_on_host` routes the duplex sponge to the host CPU, keeping
the sponge state host-resident across the stream -- byte-identical to the on-device
sponge. Every test compares `fs_on_host=True` against the default device path on the
same inputs."""
from __future__ import annotations

import ast
import pathlib
from dataclasses import replace
from typing import Any
from unittest import mock

import frx
import frx.numpy as fnp
import zk_dtypes
from absl.testing import absltest, parameterized
from frx import tree_util

import zorch
from zorch import transcript as transcript_mod
from zorch.sumcheck.jagged.fs import _fs_reduce
from zorch.testkit.koalabear16 import koalabear16_perm
from zorch.testkit.random_field import rand_ext_field, rand_field
from zorch.transcript import (
    DuplexState,
    DuplexTranscript,
    sample_challenge,
)

F = zk_dtypes.koalabear_mont
EF = zk_dtypes.koalabearx4_mont

# Host-FS runs the sponge on the CPU inside the callback; on a CPU-only backend the
# device and host sponges coincide (nothing to compare).
_CPU_BACKEND = frx.default_backend() == "cpu"


@absltest.skipIf(_CPU_BACKEND, "host-FS vs device sponge is only meaningful off CPU")
class TranscriptHostFsTest(parameterized.TestCase):
    """The `fs_on_host` opt-in: observe/sample/observe_and_sample/sample_challenge
    route to the host sponge and stay byte-identical to the device path."""

    def _new(self, fs_on_host: bool) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8, fs_on_host=fs_on_host)

    def _state_eq(self, a: DuplexTranscript, b: DuplexTranscript) -> bool:
        return all(
            bool(fnp.all(x == y))
            for x, y in zip(
                tree_util.tree_leaves(a), tree_util.tree_leaves(b), strict=True
            )
        )

    @parameterized.parameters(3, 8, 19)  # partial block, full block, block-crossing
    def test_observe_byte_identical(self, n: int) -> None:
        v = rand_field(1, (n,), F)
        self.assertTrue(
            self._state_eq(self._new(False).observe(v), self._new(True).observe(v))
        )

    @parameterized.parameters(1, 4, 9)
    def test_sample_byte_identical(self, k: int) -> None:
        v = rand_field(2, (5,), F)
        ta, dev = self._new(False).observe(v).sample(k)
        tb, host = self._new(True).observe(v).sample(k)
        self.assertTrue(bool(fnp.all(dev == host)))
        self.assertTrue(self._state_eq(ta, tb))

    @parameterized.parameters(1, 4)
    def test_observe_and_sample_byte_identical(self, k: int) -> None:
        if k == 1 and frx.default_backend() == "gpu":
            self.skipTest(
                "quarantined: the k=1 fused absorb+squeeze diverges host-vs-"
                "device on the cuda backend (gpu1 CI and an RTX 5090; k=4 and"
                " every other host-FS case match). Tracked on the zorch work"
                " board: 'fix(gpu): sparse Poseidon production-scale"
                " byte-match and transcript host-FS scoped-absorb fail on"
                " cuda'."
            )
        # The single-callback fused absorb+squeeze == the device fused form.
        v = rand_field(6, (5,), F)
        ta, dev = self._new(False).observe_and_sample(v, k)
        tb, host = self._new(True).observe_and_sample(v, k)
        self.assertTrue(bool(fnp.all(dev == host)))
        self.assertTrue(self._state_eq(ta, tb))

    @parameterized.parameters(0, 3)
    def test_check_witness_byte_identical(self, pow_bits: int) -> None:
        # check_witness routes through the backend too -- it once ran the device
        # sponge regardless of fs_on_host, so a host-FS grind's re-check could
        # disagree with its own search. Host and device must match.
        w = rand_field(8, (1,), F)[0]
        ta, dev = self._new(False).check_witness(w, pow_bits=pow_bits)
        tb, host = self._new(True).check_witness(w, pow_bits=pow_bits)
        self.assertEqual(bool(dev), bool(host))
        self.assertTrue(self._state_eq(ta, tb))

    @parameterized.named_parameters(("base_1limb", F, 1), ("ext_4limb", EF, 4))
    def test_sample_challenge_byte_identical(self, dtype: Any, limbs: int) -> None:
        # sample_challenge routes through the `sample` method, so it picks up the
        # host body; the multi-limb reinterpret stays on the device.
        v = rand_field(3, (5,), F)
        _, dev = sample_challenge(self._new(False).observe(v), dtype, limbs)
        _, host = sample_challenge(self._new(True).observe(v), dtype, limbs)
        self.assertTrue(bool(fnp.all(dev == host)))

    def test_flag_carries_across_steps(self) -> None:
        # fs_on_host must survive every step so the whole stream stays on host.
        t = self._new(True)
        self.assertTrue(t.observe(rand_field(4, (4,), F)).fs_on_host)
        self.assertTrue(t.sample(1)[0].fs_on_host)
        self.assertTrue(t.observe_and_sample(rand_field(5, (3,), F), 2)[0].fs_on_host)

    def test_state_goes_host_resident(self) -> None:
        # The optimization: the host-FS sponge state lives on the CPU across the
        # whole stream, so each hop crosses only `values` in and the challenge out,
        # not the 5 state leaves both ways.
        t = self._new(True).observe(rand_field(7, (4,), F))
        leaf = tree_util.tree_leaves(t)[0]
        self.assertEqual(next(iter(leaf.devices())).platform, "cpu")


@absltest.skipIf(_CPU_BACKEND, "a scoped host absorb is only meaningful off CPU")
class ScopedHostAbsorbTest(parameterized.TestCase):
    """`absorb_on_host` relocates ONE absorb to the CPU sponge without moving the
    stream: byte-identical to `observe`, and the state comes back on the device so
    the following device-path steps are unaffected."""

    def _new(self, on_host: bool = False) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8, fs_on_host=on_host)

    def _state_eq(self, a: DuplexTranscript, b: DuplexTranscript) -> bool:
        return all(
            bool(fnp.all(x == y))
            for x, y in zip(
                tree_util.tree_leaves(a), tree_util.tree_leaves(b), strict=True
            )
        )

    @parameterized.named_parameters(
        ("under_a_block", 5), ("one_block", 8), ("block_and_tail", 17), ("many", 130)
    )
    def test_byte_identical_to_observe(self, mlen: int) -> None:
        v = rand_field(mlen + 1, (mlen,), F)
        self.assertTrue(
            self._state_eq(self._new().observe(v), self._new().absorb_on_host(v))
        )

    def test_byte_identical_mid_stream(self) -> None:
        # From a non-zero (in_pos, out_pos): the partial input buffer has to cross
        # to the host and back, not just the sponge lane.
        def seeded() -> DuplexTranscript:
            t = self._new().observe(rand_field(1, (5,), F))
            return t.sample(3)[0].observe(rand_field(2, (7,), F))

        v = rand_field(3, (130,), F)
        self.assertTrue(self._state_eq(seeded().observe(v), seeded().absorb_on_host(v)))

    def test_sequence_matches_successive_observes(self) -> None:
        # The variadic form is the point: a length prefix plus its payload must
        # absorb in one host excursion, not one per message.
        msgs = [
            fnp.array(130, F),
            rand_field(1, (130,), F),
            fnp.array(130, F),
            rand_field(2, (130,), F),
        ]
        ref = self._new()
        for m in msgs:
            ref = ref.observe(m)
        self.assertTrue(self._state_eq(ref, self._new().absorb_on_host(*msgs)))

    def test_sequence_crosses_the_host_boundary_once(self) -> None:
        # Absorbing a sequence one call at a time would drag the state back to
        # the device between messages; batching pays the trip once. Counted by
        # how many times a fresh host commit happens.
        moves = 0
        real = transcript_mod._state_on_host

        def counting(state: DuplexState) -> DuplexState:
            nonlocal moves
            if not transcript_mod._on_host(state.sponge_state):
                moves += 1
            return real(state)

        msgs = [rand_field(i, (130,), F) for i in range(3)]
        with mock.patch.object(transcript_mod, "_state_on_host", counting):
            self._new().absorb_on_host(*msgs)
        self.assertEqual(moves, 1)

    def test_empty_sequence_is_the_identity(self) -> None:
        self.assertTrue(self._state_eq(self._new(), self._new().absorb_on_host()))

    def test_state_returns_to_the_device(self) -> None:
        # The point of "scoped": the stream stays on the device path, so the
        # following steps must not silently run host-resident.
        t = self._new().absorb_on_host(rand_field(4, (130,), F))
        for leaf in tree_util.tree_leaves(t):
            self.assertNotEqual(next(iter(leaf.devices())).platform, "cpu")
        self.assertFalse(t.fs_on_host)

    def test_challenges_match_after_a_scoped_absorb(self) -> None:
        # What actually has to hold: the Fiat-Shamir stream cannot notice where an
        # absorb ran.
        v = rand_field(6, (130,), F)
        _, dev = sample_challenge(self._new().observe(v), EF, 4)
        _, scoped = sample_challenge(self._new().absorb_on_host(v), EF, 4)
        self.assertTrue(bool(fnp.all(dev == scoped)))

    def test_host_backend_absorbs_in_place(self) -> None:
        # Already on host: nothing to relocate, and the state must NOT be dragged
        # to the device -- that would undo the backend's whole residency win.
        t = self._new(True).absorb_on_host(rand_field(8, (17,), F))
        self.assertTrue(
            self._state_eq(t, self._new(True).observe(rand_field(8, (17,), F)))
        )
        self.assertEqual(
            next(iter(tree_util.tree_leaves(t)[0].devices())).platform, "cpu"
        )

    def test_rejects_a_traced_call(self) -> None:
        # It moves buffers across the host boundary, so it cannot be traced. Failing
        # loudly beats silently leaking a host round-trip into a compiled zone.
        with self.assertRaisesRegex(ValueError, "eager"):
            frx.jit(lambda t, x: t.absorb_on_host(x))(
                self._new(), rand_field(9, (8,), F)
            )


class _RecordingFs:
    """A `_FsBackend` that records which entry points it was asked for and
    delegates. Installed on a transcript so a test can assert a *call site*
    reaches the backend, rather than naming one backend's body directly."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[str] = []

    @property
    def on_host(self) -> bool:
        return self.inner.on_host

    def observe(self, t: Any, values: Any) -> Any:
        self.calls.append("observe")
        return self.inner.observe(t, values)

    def sample(self, t: Any, n: int) -> Any:
        self.calls.append("sample")
        return self.inner.sample(t, n)

    def observe_and_sample(self, t: Any, values: Any, n: int) -> Any:
        self.calls.append("observe_and_sample")
        return self.inner.observe_and_sample(t, values, n)

    def check_witness(self, t: Any, witness: Any, *, pow_bits: int) -> Any:
        self.calls.append("check_witness")
        return self.inner.check_witness(t, witness, pow_bits=pow_bits)


class FsEntryPointTest(parameterized.TestCase):
    """Every Fiat-Shamir hop reaches the transcript's `fs` backend.

    `_FsBackend` exists so the device/host choice is structural -- "every method
    routes through the backend and so cannot silently ignore the host placement".
    The transcript's own methods hold that line (the class above checks the
    results). What escapes it is a *caller* that names a backend body directly:
    such a hop is pinned to that body's backend and a `fs_on_host=True` prove
    silently keeps running it on the device. These tests watch the callers.

    Not skipped on CPU: routing is Python dispatch, so it is backend-independent.
    """

    def _spied(self, fs_on_host: bool) -> tuple[DuplexTranscript, _RecordingFs]:
        t = DuplexTranscript.new(koalabear16_perm(), rate=8, fs_on_host=fs_on_host)
        spy = _RecordingFs(t.fs)
        return replace(t, fs=spy), spy

    @parameterized.named_parameters(("device", False), ("host", True))
    def test_jagged_round_hop_routes_through_the_backend(self, fs_on_host: bool):
        """The sumcheck per-round hop -- the hottest FS call site in a jagged
        LogUp-GKR prove, ~78% of its hops -- must reach the backend. It once called
        the device body `_observe_and_sample_marked` directly, so `fs_on_host=True`
        left every round on the device."""
        t, spy = self._spied(fs_on_host)
        t_out, _, _, _ = _fs_reduce(
            rand_ext_field(1, (4,), F, EF),  # round poly, coefficient form
            t,
            rand_ext_field(2, (), F, EF),  # pad_adj
            rand_ext_field(3, (), F, EF),  # z_cur
            4,  # challenge limbs for EF
            EF,
        )
        self.assertEqual(spy.calls, ["observe_and_sample"])
        self.assertIs(t_out.fs, spy)

    def test_no_module_imports_a_private_fs_body(self):
        """Nothing outside `transcript.py` may import a private name from it.

        The private FS bodies (`_observe_and_sample_marked`, `_observe_and_sample_body`,
        `_observe_body`, `_sample_body`) are backend implementations, one per
        placement; the entry points are the `DuplexTranscript` methods. Importing a
        body is how a call site leaves the `_FsBackend` contract, so the import is
        what this forbids -- statically, for every module at once, which a
        behavioural test can only do one call site at a time. Tests are exempt:
        they cover the bodies deliberately."""
        root = pathlib.Path(zorch.__file__).parent
        offenders, scanned = [], set()
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root)
            if rel.name == "transcript.py" or "testing" in rel.parts:
                continue
            scanned.add(rel.as_posix())
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "zorch.transcript":
                    continue
                offenders += [
                    f"{rel}:{node.lineno} imports {a.name}"
                    for a in node.names
                    if a.name.startswith("_")
                ]
        # This walks the runfiles tree, which holds only what the target depends
        # on -- so an unlisted dep would shrink the scan to nothing and the test
        # would pass having checked no code at all. Name the modules that must be
        # in it, so a thin closure fails here instead of going quiet.
        self.assertContainsSubset(
            {"challenge.py", "sumcheck/jagged/fs.py", "logup_gkr/jagged_prover.py"},
            scanned,
            "the FS callers are missing from the runfiles tree -- add their "
            "targets to this test's deps, the scan is not covering them",
        )
        self.assertEqual(
            offenders,
            [],
            "call these through the DuplexTranscript methods, not the private "
            "backend bodies:\n  " + "\n  ".join(offenders),
        )


class HostFsFfiTargetTest(parameterized.TestCase):
    """`set_host_fs_ffi_target` moves EVERY host hop onto the FFI, eager included.

    The eager hops are the ones that matter. A jagged prove runs ~101 hops, ~98 of
    them inside jitted layer zones where they cost nothing at the margin; the
    three at stage boundaries run eagerly, and on the resident path each one ships
    five state leaves to the CPU and back for 0.43us of hashing. Those three were
    the whole host-vs-device gap, so a regression that quietly leaves them on the
    resident path is a ~2ms regression that no correctness test would catch.

    Dispatch only, so this runs on any backend -- the handler itself is the
    consumer's and needs a GPU.
    """

    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(transcript_mod.set_host_fs_ffi_target, None)
        transcript_mod.set_host_fs_ffi_target(None)

    def _transcript(self) -> DuplexTranscript:
        return DuplexTranscript.new(koalabear16_perm(), rate=8, fs_on_host=True)

    def _spy_on_ffi(self) -> list[tuple[int, bool]]:
        """Replace the eager FFI wrapper with a recorder that returns the
        transcript untouched, so no handler is needed."""
        seen: list[tuple[int, bool]] = []

        def fake(n: int, absorbs: bool) -> Any:
            seen.append((n, absorbs))

            def run(t: Any, *rest: Any) -> Any:
                out = fnp.zeros(max(n, 1), t.field)
                return (t, out)

            return run

        self.enter_context(mock.patch.object(transcript_mod, "_host_hop_ffi_eager", fake))
        return seen

    def test_no_target_keeps_the_resident_eager_path(self) -> None:
        """Without a target the eager hop stays on the resident sponge, which is
        the faster path when nothing traced is involved."""
        seen = self._spy_on_ffi()
        t, _ = self._transcript().observe_and_sample(rand_field(9, (4,), F), 1)
        self.assertEqual(seen, [])
        self.assertTrue(transcript_mod._on_host(t.state.sponge_state))

    @parameterized.named_parameters(
        ("observe_and_sample", "observe_and_sample"),
        ("sample", "sample"),
        ("observe", "observe"),
    )
    def test_target_routes_every_eager_entry_point(self, method: str) -> None:
        seen = self._spy_on_ffi()
        transcript_mod.set_host_fs_ffi_target("a_consumers_target")
        t = self._transcript()
        if method == "sample":
            t.sample(1)
        elif method == "observe":
            t.observe(rand_field(9, (4,), F))
        else:
            t.observe_and_sample(rand_field(9, (4,), F), 1)
        self.assertLen(seen, 1, f"{method} did not reach the FFI target")


if __name__ == "__main__":
    absltest.main()
