# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0

# Stamped to 0.1.0.dev<UTC timestamp> by .github/workflows/dev-release.yml at
# release time; the 0.0.0 placeholder is what local / editable installs report.
# Single source of truth for the packaged version: pyproject.toml derives it via
# `attr =`, and release.yml refuses to publish a tag that disagrees with it.
__version__ = "0.1.0"
