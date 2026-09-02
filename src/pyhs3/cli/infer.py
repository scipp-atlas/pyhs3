"""Inference-level pyhs3 CLI commands: ``nll``.

Builds a computation graph from the workspace and evaluates the negative
log-likelihood at a chosen point in parameter space.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pytensor.compile.maker import function
from pytensor.graph.traversal import explicit_graph_inputs
from rich.console import Console

from pyhs3.analyses import Analysis
from pyhs3.cli._shared import load_workspace, parse_param
from pyhs3.likelihoods import Likelihood
from pyhs3.workspace import Workspace

_err_console = Console(stderr=True)


def _select_target(ws: Workspace, name: str | None) -> Analysis | Likelihood:
    """Choose the analysis or likelihood to evaluate.

    With *name*, look it up in analyses first, then likelihoods. Without a
    name, use the sole analysis if there is exactly one, else the sole
    likelihood; ambiguity (more than one, none) is an error the caller must
    resolve with ``--analysis``.
    """
    if name is not None:
        if ws.analyses and name in ws.analyses:
            return ws.analyses[name]
        if ws.likelihoods and name in ws.likelihoods:
            return ws.likelihoods[name]
        msg = f"no analysis or likelihood named {name!r} in the workspace"
        raise typer.BadParameter(msg)

    if ws.analyses and len(ws.analyses) == 1:
        return ws.analyses[0]
    if ws.analyses and len(ws.analyses) > 1:
        names = ", ".join(a.name for a in ws.analyses)
        msg = f"workspace has multiple analyses ({names}); choose one with --analysis"
        raise typer.BadParameter(msg)
    if ws.likelihoods and len(ws.likelihoods) == 1:
        return ws.likelihoods[0]
    if ws.likelihoods and len(ws.likelihoods) > 1:
        names = ", ".join(lk.name for lk in ws.likelihoods)
        msg = (
            f"workspace has multiple likelihoods ({names}); choose one with --analysis"
        )
        raise typer.BadParameter(msg)

    msg = "workspace has no analyses or likelihoods to evaluate"
    raise typer.BadParameter(msg)


def _compute_nll(
    ws: Workspace, target: Analysis | Likelihood, overrides: dict[str, float]
) -> float:
    """Compile the joint log-prob and evaluate ``NLL = -2 * log_prob``.

    Parameter values are layered, later layers winning: the workspace's own
    ``parameter_points`` (a convenience base so an analysis without an ``init``
    set is still evaluable), then the model's free (non-const) parameter values,
    then the caller's *overrides*. Observable data is taken from the likelihood
    the model was built from, and always wins over an override of the same
    name: ``--param`` overrides a free parameter, never the observed data.
    """
    model = ws.model(target, progress=False)
    log_prob = model.log_prob
    data = model.data

    # Free symbolic inputs: observable data arrays plus the non-const parameters.
    inputs_map = {
        var.name: var
        for var in explicit_graph_inputs([log_prob])
        if var.name is not None
    }

    # Base layer: every value declared in the workspace's parameter_points.
    # A model built from an analysis with no `init` has empty free_params, so
    # this base makes it evaluable from the workspace's nominal values. When
    # multiple sets declare the same name, the last one seen wins; the model's
    # own set (applied next) takes precedence for names it defines.
    params: dict[str, float] = {}
    for pset in ws.parameter_points or []:
        for point in pset:
            params[point.name] = float(point.value)
    params.update(model.free_params)

    for name, value in overrides.items():
        if name in data:
            _err_console.print(
                f"[yellow]warning:[/yellow] --param {name!r} names observable data, not a free parameter; ignoring"
            )
            continue
        if name not in inputs_map:
            _err_console.print(
                f"[yellow]warning:[/yellow] --param {name!r} is not a free parameter of this model; ignoring"
            )
            continue
        params[name] = value

    call_kwargs: dict[str, Any] = {
        **{name: value for name, value in params.items() if name in inputs_map},
        **data,
    }
    missing = [name for name in inputs_map if name not in call_kwargs]
    if missing:
        msg = "no value available for model input(s): " + ", ".join(sorted(missing))
        raise typer.BadParameter(msg)

    fn = function(list(inputs_map.values()), log_prob)
    # log_prob sums over events, returning shape (M,); M == 1 for scalar params.
    log_prob_value = float(fn(**call_kwargs).item())
    return -2.0 * log_prob_value


def nll(
    workspace: Annotated[
        str | None,
        typer.Argument(
            metavar="WORKSPACE",
            help="Path to an HS3 workspace JSON file. Use '-' or omit to read from stdin.",
        ),
    ] = None,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            "-p",
            metavar="NAME=VALUE",
            help="Override a parameter value. Repeatable.",
        ),
    ] = None,
    params_file: Annotated[
        Path | None,
        typer.Option(
            "--params-file",
            help="JSON file mapping parameter names to values.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    analysis: Annotated[
        str | None,
        typer.Option(
            "--analysis",
            help="Name of the analysis (or likelihood) to evaluate. Defaults to the sole one.",
        ),
    ] = None,
) -> None:
    """Compute the negative log-likelihood at a point in parameter space.

    Parameter values come from ``--params-file`` (a JSON name->value mapping)
    and/or repeated ``--param name=value`` options, layered over the
    workspace's own initial values. Repeated ``--param`` overrides win over the
    file. The scalar NLL is printed to stdout.
    """
    overrides: dict[str, float] = {}
    if params_file is not None:
        raw = json.loads(params_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"--params-file {params_file} must contain a JSON object mapping names to values"
            raise typer.BadParameter(msg)
        overrides.update({str(name): float(value) for name, value in raw.items()})
    for item in param or []:
        name, value = parse_param(item)
        overrides[name] = value

    ws = load_workspace(workspace)
    target = _select_target(ws, analysis)
    value = _compute_nll(ws, target, overrides)
    typer.echo(f"{value!r}")
