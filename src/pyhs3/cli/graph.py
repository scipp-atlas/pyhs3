"""Graph-visualization pyhs3 CLI command: ``graph``.

Renders a distribution's PyTensor computation graph via
:meth:`pyhs3.Model.visualize_graph`, which requires the optional
``pydot`` dependency (``pip install 'pyhs3[graph]'``) and the system
Graphviz ``dot`` executable, which pydot cannot install on its own.
"""

from __future__ import annotations

from typing import Annotated

import typer

from pyhs3.cli._shared import load_workspace
from pyhs3.cli.infer import _select_target


def graph(
    workspace: Annotated[
        str | None,
        typer.Argument(
            metavar="WORKSPACE",
            help="Path to an HS3 workspace JSON file. Use '-' or omit to read from stdin.",
        ),
    ] = None,
    *,
    name: Annotated[
        str,
        typer.Option("--name", help="Name of the distribution to visualize."),
    ],
    analysis: Annotated[
        str | None,
        typer.Option(
            "--analysis",
            help="Name of the analysis (or likelihood) to build the model from. Defaults to the sole one.",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--fmt", help="Output format: svg, png, or pdf."),
    ] = "svg",
    outfile: Annotated[
        str | None,
        typer.Option(
            "--outfile",
            help="Output file path. Defaults to '{name}_graph.{fmt}'.",
        ),
    ] = None,
    path: Annotated[
        str | None,
        typer.Option(
            "--path",
            help="Directory to write the output file in. Ignored if --outfile is set.",
        ),
    ] = None,
    show_id: Annotated[
        bool,
        typer.Option("--show-id", help="Append each op's toposort index to its label."),
    ] = False,
    show_dtype: Annotated[
        bool,
        typer.Option("--show-dtype", help="Keep dtype annotations in node labels."),
    ] = False,
    show_shape: Annotated[
        bool,
        typer.Option("--show-shape", help="Keep shape annotations in node labels."),
    ] = False,
) -> None:
    # This docstring is dumped verbatim into Markdown by
    # docs/_scripts/generate_cli_docs.py, so its cross-reference below uses
    # MyST's {role}`target` syntax rather than this codebase's usual RST
    # :role:`target` syntax -- MyST doesn't parse the colon form inline.
    r"""Render a distribution's computation graph to an image file.

    Builds the model the same way ``nll`` does (``--analysis`` selects which
    analysis or likelihood to build from) and delegates to
    {meth}`~pyhs3.Model.visualize_graph`. Prints the path of the
    rendered file to stdout. Requires the ``graph`` optional dependency
    (``pip install 'pyhs3\[graph]'``, which installs ``pydot``) **and** the
    system Graphviz ``dot`` executable, which pydot cannot install on its own
    (``apt install graphviz``, ``brew install graphviz``, or
    ``conda install graphviz``). If either is missing, or the named
    distribution doesn't exist in the model, this reports the problem
    through a clean error and exits non-zero, never a raw traceback.
    """
    ws = load_workspace(workspace)
    target = _select_target(ws, analysis)
    model = ws.model(target, progress=False)

    try:
        outpath = model.visualize_graph(
            name,
            fmt=fmt,
            outfile=outfile,
            path=path,
            show_id=show_id,
            show_dtype=show_dtype,
            show_shape=show_shape,
        )
    except ImportError as exc:
        msg = "pyhs3 graph requires pydot: install it with `pip install 'pyhs3[graph]'`"
        raise typer.BadParameter(msg) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except FileNotFoundError as exc:
        msg = (
            "pyhs3 graph requires the Graphviz `dot` executable, which pydot "
            "cannot install on its own: install Graphviz separately "
            "(e.g. `apt install graphviz`, `brew install graphviz`, or "
            "`conda install graphviz`)"
        )
        raise typer.BadParameter(msg) from exc

    typer.echo(outpath)
