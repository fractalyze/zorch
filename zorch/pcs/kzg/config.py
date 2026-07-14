# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""KZG wire types on the `pcs` seam. Both are bare arrays — the commitment is a
batch of G1 points and the proof is the batch of quotient commitments π — so they
are named aliases, not wrapper dataclasses (promoted only if a commitment ever
gains structure; see docs/blocks/pcs.md "Instance anatomy")."""

from __future__ import annotations

from typing import TypeAlias

from frx import Array

KzgCommitment: TypeAlias = Array  # bn254 G1 affine [K] — one commitment per poly
KzgProof: TypeAlias = Array  # bn254 G1 affine [K] — one opening proof π per poly
