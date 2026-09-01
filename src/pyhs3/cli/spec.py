"""Spec-level pyhs3 CLI commands: ``validate`` and ``inspect``.

These operate on the workspace specification itself, without building a
computation graph or evaluating any likelihood.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from pyhs3.cli._shared import (
    display_name,
    finite_or_none,
    load_workspace,
    read_spec,
    stdout_is_interactive,
)
from pyhs3.exceptions import WorkspaceValidationError
from pyhs3.workspace import Workspace

_err_console = Console(stderr=True)


def validate(
    workspace: Annotated[
        str | None,
        typer.Argument(
            metavar="WORKSPACE",
            help="Path to an HS3 workspace JSON file. Use '-' or omit to read from stdin.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Show all validation errors, not just the first 20."
        ),
    ] = False,
) -> None:
    """Load and validate an HS3 workspace, reporting success or the errors found."""
    source = display_name(workspace)
    try:
        spec = read_spec(workspace)
    except json.JSONDecodeError as exc:
        _err_console.print(f"[red]Invalid JSON in {source}:[/red] {exc}")
        raise typer.Exit(code=1) from None
    except typer.BadParameter as exc:
        _err_console.print(f"[red]Invalid workspace spec in {source}:[/red] {exc}")
        raise typer.Exit(code=1) from None

    try:
        Workspace(**spec)
    except ValidationError as exc:
        summary = Workspace.format_validation_error(exc, source, verbose)
        _err_console.print(summary)
        raise typer.Exit(code=1) from None
    except WorkspaceValidationError as exc:
        _err_console.print(
            f"[red]Workspace validation failed for {source}:[/red]\n{exc}"
        )
        raise typer.Exit(code=1) from None

    typer.echo(f"{source} is a valid HS3 workspace.")


def _summarize(ws: Workspace) -> dict[str, Any]:
    """Build a JSON-serializable summary of the workspace contents."""
    distributions = [
        {"name": dist.name, "type": getattr(dist, "type", None)}
        for dist in (ws.distributions or [])
    ]

    domains = []
    for dom in ws.domains or []:
        axes = [
            {
                "name": axis.name,
                "min": finite_or_none(float(getattr(axis, "min", float("-inf")))),
                "max": finite_or_none(float(getattr(axis, "max", float("inf")))),
            }
            for axis in getattr(dom, "axes", []) or []
        ]
        domains.append(
            {"name": dom.name, "type": getattr(dom, "type", None), "axes": axes}
        )

    data = []
    for datum in ws.data or []:
        entries = getattr(datum, "entries", None)
        data.append(
            {
                "name": datum.name,
                "type": getattr(datum, "type", None),
                "entries": None if entries is None else len(entries),
            }
        )

    likelihoods = [
        {
            "name": lk.name,
            "distributions": [_ref_name(d) for d in lk.distributions],
            "data": [_ref_name(d) for d in lk.data],
        }
        for lk in (ws.likelihoods or [])
    ]

    analyses = [
        {
            "name": an.name,
            "likelihood": _ref_name(an.likelihood),
            "parameters_of_interest": an.parameters_of_interest or [],
            "domains": [_ref_name(d) for d in an.domains],
        }
        for an in (ws.analyses or [])
    ]

    parameter_points = [
        {
            "name": ps.name,
            "parameters": [
                {"name": pp.name, "value": pp.value, "const": pp.const}
                for pp in ps.parameters
            ],
        }
        for ps in (ws.parameter_points or [])
    ]

    return {
        "metadata": {"hs3_version": ws.metadata.hs3_version},
        "distributions": distributions,
        "domains": domains,
        "data": data,
        "likelihoods": likelihoods,
        "analyses": analyses,
        "parameter_points": parameter_points,
    }


def _ref_name(ref: Any) -> str:
    """Return the name of a resolved FK object, or the string reference itself."""
    return ref if isinstance(ref, str) else ref.name


def _render_table(summary: dict[str, Any]) -> None:
    """Render a human-readable summary of the workspace to stdout."""
    console = Console()
    version = summary["metadata"]["hs3_version"]
    console.print(f"[bold]HS3 workspace[/bold] (hs3_version {version})")

    dist_table = Table(title="Distributions", title_justify="left")
    dist_table.add_column("name")
    dist_table.add_column("type")
    for dist in summary["distributions"]:
        dist_table.add_row(dist["name"], str(dist["type"]))
    console.print(dist_table)

    dom_table = Table(title="Domains", title_justify="left")
    dom_table.add_column("domain")
    dom_table.add_column("parameter")
    dom_table.add_column("min")
    dom_table.add_column("max")
    for dom in summary["domains"]:
        for axis in dom["axes"]:
            dom_table.add_row(
                dom["name"],
                axis["name"],
                _fmt_bound(axis["min"], "-inf"),
                _fmt_bound(axis["max"], "+inf"),
            )
    console.print(dom_table)

    data_table = Table(title="Data", title_justify="left")
    data_table.add_column("name")
    data_table.add_column("type")
    data_table.add_column("entries")
    for datum in summary["data"]:
        entries = datum["entries"]
        data_table.add_row(
            datum["name"], str(datum["type"]), "-" if entries is None else str(entries)
        )
    console.print(data_table)

    param_table = Table(title="Parameter points", title_justify="left")
    param_table.add_column("set")
    param_table.add_column("parameter")
    param_table.add_column("value")
    param_table.add_column("const")
    for pset in summary["parameter_points"]:
        for param in pset["parameters"]:
            param_table.add_row(
                pset["name"], param["name"], f"{param['value']:g}", str(param["const"])
            )
    console.print(param_table)

    lk_table = Table(title="Likelihoods", title_justify="left")
    lk_table.add_column("name")
    lk_table.add_column("distributions")
    lk_table.add_column("data")
    for lk in summary["likelihoods"]:
        lk_table.add_row(
            lk["name"], ", ".join(lk["distributions"]), ", ".join(lk["data"])
        )
    console.print(lk_table)

    an_table = Table(title="Analyses", title_justify="left")
    an_table.add_column("name")
    an_table.add_column("likelihood")
    an_table.add_column("parameters of interest")
    an_table.add_column("domains")
    for an in summary["analyses"]:
        an_table.add_row(
            an["name"],
            an["likelihood"],
            ", ".join(an["parameters_of_interest"]),
            ", ".join(an["domains"]),
        )
    console.print(an_table)


def _fmt_bound(value: float | None, fallback: str) -> str:
    """Format a possibly-infinite bound (stored as ``None``) for a table cell."""
    return fallback if value is None else f"{value:g}"


def inspect(
    workspace: Annotated[
        str | None,
        typer.Argument(
            metavar="WORKSPACE",
            help="Path to an HS3 workspace JSON file. Use '-' or omit to read from stdin.",
        ),
    ] = None,
    output_json: Annotated[
        bool | None,
        typer.Option(
            "--json/--no-json",
            help="Force JSON (--json) or table (--no-json) output. Default: autodetect from the terminal.",
        ),
    ] = None,
) -> None:
    """Summarize a workspace: distributions, domains, data, likelihoods, analyses.

    Prints a Rich table when attached to an interactive terminal and machine
    readable JSON when piped. Override the autodetection with ``--json`` or
    ``--no-json``.
    """
    ws = load_workspace(workspace)
    summary = _summarize(ws)

    if output_json is None:
        output_json = not stdout_is_interactive()

    if output_json:
        typer.echo(json.dumps(summary, indent=2))
    else:
        _render_table(summary)
