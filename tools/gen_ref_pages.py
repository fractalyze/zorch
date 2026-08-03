# Copyright 2026 The Zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Generate one API-reference page per public zorch module, at docs build.

Runs inside mkdocs via the gen-files plugin (see mkdocs.yml); nothing is
committed. Each page is a single mkdocstrings directive, so the rendered
reference is read straight from the working tree's signatures and docstrings —
the WHAT stays in code (docs/reference/conventions.md), and a new module is
documented the moment its file lands, with no page to remember to add.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

REPO = Path(__file__).resolve().parent.parent

nav = mkdocs_gen_files.Nav()

for path in sorted((REPO / "zorch").rglob("*.py")):
    rel = path.relative_to(REPO)
    parts = list(rel.with_suffix("").parts)

    # Private modules, tests, and test scaffolding are not public surface.
    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue
    if parts[-1].endswith("_test") or "testing" in parts or "__pycache__" in parts:
        continue
    if parts[-1] == "__init__":
        parts = parts[:-1]

    doc_path = Path("api", *parts[1:] or ["index"]).with_suffix(".md")
    identifier = ".".join(parts)
    nav[parts] = doc_path.relative_to("api").as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as page:
        page.write(f"# `{identifier}`\n\n::: {identifier}\n")
    mkdocs_gen_files.set_edit_path(doc_path, rel.as_posix())

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as summary:
    summary.writelines(nav.build_literate_nav())

# The landing page is site presentation, not repo documentation — it lives
# beside this script (outside docs/, which tools/lint_docs.py polices for
# tree-resolvable links; the landing links build-time virtual pages).
with mkdocs_gen_files.open("index.md", "w") as page:
    page.write((REPO / "tools" / "site" / "index.md").read_text(encoding="utf-8"))
mkdocs_gen_files.set_edit_path("index.md", "tools/site/index.md")

# Mirror the agent skill bundle as the site's task-oriented Guide section.
# The skill is the single source (release-lockstep-gated); the site copy is
# produced at build time, so the two can never drift.
SKILL = REPO / "skills" / "using-zorch"

skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
skill_md = skill_md.split("---\n", 2)[-1]  # drop the agent frontmatter
skill_md = skill_md.replace("(references/", "(")
with mkdocs_gen_files.open("guide/index.md", "w") as page:
    page.write(skill_md)
mkdocs_gen_files.set_edit_path("guide/index.md", "skills/using-zorch/SKILL.md")

for ref in sorted((SKILL / "references").glob("*.md")):
    out = f"guide/{ref.name}"
    with mkdocs_gen_files.open(out, "w") as page:
        page.write(ref.read_text(encoding="utf-8"))
    mkdocs_gen_files.set_edit_path(out, ref.relative_to(REPO).as_posix())
