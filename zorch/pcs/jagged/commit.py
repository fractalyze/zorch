# zorch/pcs/jagged/commit.py
# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Jagged commit shim — the commit logic moved into `JaggedPcsProver`
(`zorch.pcs.jagged.prover`) so commit + open live on one prover object. This
module re-exports it under the legacy `JaggedPcs` name for existing importers.
"""

from __future__ import annotations

from zorch.pcs.jagged.prover import JaggedPcsProver as JaggedPcs

__all__ = ["JaggedPcs"]
