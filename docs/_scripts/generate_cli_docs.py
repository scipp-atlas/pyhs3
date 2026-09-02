#!/usr/bin/env python3
"""Generate the pyhs3 CLI reference page from the live Typer app.

Shells out to the ``typer`` console script's own ``utils docs`` command
(ships unconditionally with the ``typer`` package pyhs3 already depends on,
no new dependency) rather than calling Typer's internal APIs directly:
``typer.cli.docs()`` builds its own Click ``Context`` from the CLI
invocation it's running inside, which is awkward to replicate faithfully
in-process. Regenerating this way also means the reference page can never
drift from the actual command definitions -- it is re-derived from
``pyhs3.cli:app`` on every docs build, the same way ``docs/reference/api.rst``
is kept in sync with the source via ``autosummary_generate``.
"""

# ruff: noqa: T201

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = DOCS_DIR / "reference" / "cli.md"


def main() -> int:
    """Regenerate docs/reference/cli.md from the live pyhs3 CLI app."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "typer",
            "pyhs3.cli",
            "utils",
            "docs",
            "--name",
            "pyhs3",
            "--output",
            str(OUTPUT_FILE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"✗ Failed to generate {OUTPUT_FILE}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    # Diátaxis markers per .claude/rules/diataxis-declaration.md: that file
    # specifies the RST comment form (`.. name:`) for `.rst` units. MyST does
    # not treat a bare `..`-prefixed line as a comment the way docutils does,
    # so the same literal text is wrapped in an HTML comment here instead --
    # hidden from rendered output in both raw Markdown viewers and Sphinx,
    # while `grep -rn "diataxis:" docs/` (dropping the RST-only `^\.\. `
    # anchor) still finds it.
    markers = "<!-- .. diataxis: reference -->\n<!-- .. status: implemented -->\n\n"
    OUTPUT_FILE.write_text(markers + OUTPUT_FILE.read_text())

    print(f"✓ Generated {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
