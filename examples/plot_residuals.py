
"""Plot a graph showing the progression of offset and residual
between qf and pyhs3 nll values.
consumed by eval_simple_muscan --plot-resid
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

# Positional schema for the workspace naming convention, e.g.
#   1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json
# Each token is one field; rename/extend to match your generator.
FIELDS = ["channels", "bkg", "sig", "shape", "np", "constr", "yield"]


def parse_workspace(workspace: str) -> dict[str, str]:
    """Split a workspace filename into its named fields."""
    stem = Path(workspace).stem  # drops the ".json"
    return dict(zip(FIELDS, stem.split("_")))


def value_of(token: str) -> str:
    """Strip the lowercase prefix from a field token, leaving the value.

    sigGauss -> Gauss, bkgRooExp -> RooExp, npOn -> On,
    yield1x -> 1x, 1ch -> 1ch (no leading lowercase, unchanged)
    """
    return re.sub(r"^[a-z]+", "", token) or token


def label_for(workspace: str, label_field: str | list[str]) -> str:
    """Build an x-axis label from one or more fields of the workspace name."""
    fields = parse_workspace(workspace)
    keys = [label_field] if isinstance(label_field, str) else list(label_field)
    return "\n".join(value_of(fields[k]) for k in keys)


def _label_sort_key(label: str):
    """Sort labels so e.g. 2ch comes before 10ch (numeric where possible)."""
    m = re.match(r"\D*(\d+)", label)
    return (0, int(m.group(1))) if m else (1, label)

def numeric_of(token: str) -> int | None:
    """Pull the first integer out of a field value, or None if there isn't one.
 
    1ch -> 1, 30ch -> 30, yield1x -> 1, bkgRooExp -> None
    """
    m = re.search(r"\d+", token)
    return int(m.group()) if m else None
 
 
def _numeric_for(workspace: str, label_field: str | list[str]) -> int | None:
    """Numeric x value for a workspace, only defined for a single numeric field."""
    if isinstance(label_field, list):
        if len(label_field) != 1:
            return None  # multiple fields -> no single numeric axis
        label_field = label_field[0]
    token = parse_workspace(workspace).get(label_field, "")
    return numeric_of(value_of(token))

def describe_fixed(workspaces, label_field) -> list[str]:
    """Summarise the fields held constant across the workspaces,
    i.e. everything except the varied (label) field(s)."""
    varied = {label_field} if isinstance(label_field, str) else set(label_field)
    parsed = [parse_workspace(w) for w in workspaces]
    parts = []
    for key in FIELDS:
        if key in varied:
            continue
        vals = {value_of(p[key]) for p in parsed if key in p}
        parts.append(f"{key}={vals.pop()}" if len(vals) == 1 else f"{key}=mixed")
    return parts

def plot_residual_and_offset(
    results: list[dict],
    output_pdf: Path,
    label_field: str | list[str] = "channels",
    numeric_x: bool | str = "auto",
    log_x: bool = False,
    log_y: bool = True
) -> Path:
    """Plot mean offset and max residual per workspace.

    label_field: which parsed field(s) become the x-axis point labels,
                 i.e. the field you are iterating over (e.g. "channels",
                 "yield", or ["channels", "yield"]).
    sort_by:     "resid" keeps the original ordering by max residual;
                 "label" orders points by the iterated field instead,
                 which reads better as a progression.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    rows = [
        (
            result["mean_offset"],
            result["max_abs_resid"],
            result["workspace"],
            label_for(result["workspace"], label_field),
            _numeric_for(result["workspace"], label_field)
        )
        for result in results
    ]

    # Decide whether the x-axis is a true numeric scale or evenly-spaced bins.
    have_all_numeric = all(r[4] is not None for r in rows)
    use_numeric = have_all_numeric if numeric_x == "auto" else bool(numeric_x)
    if use_numeric and not have_all_numeric:
        use_numeric = False  # asked for it, but some field value has no number
 
    if use_numeric:
        rows.sort(key=lambda r: r[4])  # ascending by the numeric value
    elif sort_by == "label":
        rows.sort(key=lambda r: _label_sort_key(r[3]))
    else:  # default: preserve the original sort by max residual
        rows.sort(key=lambda r: r[1])
 
    diffs, resids, workspaces, names, values = zip(*rows)
    diffs = [abs(diff) for diff in diffs]
 
    axis_label = (
        label_field if isinstance(label_field, str) else ", ".join(label_field)
    )
 
    if use_numeric:
        x = list(values)
        ax1.scatter(x, diffs)
        ax2.scatter(x, resids)
        # Mark each actual data point, labelled with its short value (e.g. 1ch).
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
    else:
        x = range(len(rows))
        ax1.plot(x, diffs, label="diffs")
        ax2.plot(x, resids, label="resids")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(names, rotation=45, ha="right")
 
    ax1.set_title("mean offset by workspace")
    ax1.grid(True, alpha=0.25)
    ax1.set_yscale("log") if log_y else None
 
    ax2.set_title("max residual by workspace")
    ax2.grid(True, alpha=0.25)
    ax2.set_xlabel(axis_label)
    ax2.set_yscale("log") if log_y else None
    ax2.set_xscale("log") if log_x else None
 
    varied_label = (
        label_field if isinstance(label_field, str) else ", ".join(label_field)
    )
    fixed = ", ".join(describe_fixed(workspaces, label_field))
    fig.suptitle(f"{fixed}\nvarying: {varied_label}")

    fig.tight_layout(rect=(0, 0, 1, 0.94))  # leave headroom for suptitle
    plt.savefig(output_pdf)
    return output_pdf 



