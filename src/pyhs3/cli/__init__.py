"""pyhs3 command-line interface.

Wires the individual subcommands into a single Typer application exposed as the
``pyhs3`` console script (see ``[project.scripts]`` in ``pyproject.toml``).
Commands are split by concern, mirroring pyhf's CLI layout: :mod:`pyhs3.cli.spec`
holds spec-level operations (``validate``, ``inspect``) and :mod:`pyhs3.cli.infer`
holds inference operations (``nll``), with shared I/O helpers in
:mod:`pyhs3.cli._shared`.
"""

from __future__ import annotations

import typer

from pyhs3.cli import graph, infer, plot, spec

app = typer.Typer(
    name="pyhs3",
    help="Command-line tools for HS3 workspaces: validate, inspect, and evaluate.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
)

app.command()(graph.graph)
app.command()(spec.validate)
app.command()(spec.inspect)
app.command()(infer.nll)
app.command()(plot.plot)


__all__ = ["app"]
