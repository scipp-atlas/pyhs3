# # ruff: noqa: T201
# """Plot a graph showing the progression of offset and residual 
# between qf and pyhs3 nll values.  

# consumed by eval_simple_muscan --plot-resid
# """
# from __future__ import annotations

# import math
# from pathlib import Path

# import matplotlib as mpl

# mpl.use("Agg")

# import numpy as np
# from matplotlib import pyplot as plt

# def plot_residual_and_offset(results: list[dict], output_pdf: Path) -> Path:
#     fig, (ax1, ax2) = plt.subplots(2, 1)

#     diffs_and_resids = []
#     [
#         diffs_and_resids.append(
#             (
#             result['mean_offset'], 
#             result['max_abs_resid'], 
#             result['workspace']
#             )
#         ) for result in results
#     ]

#     sorted_results = sorted(diffs_and_resids, key=lambda x: x[1])

#     diffs, resids, names = zip(*sorted_results)

#     names = [] 

#     x = range(len(sorted_results))
#     ax1.plot(x, diffs, label="diffs")
#     ax1.set_title("mean offset by workspace")
#     ax1.grid(True, alpha=0.25)
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(names)

#     ax2.plot(x, resids, label="resids")
#     ax2.set_title("max residual by workspace")
#     ax2.grid(True, alpha=0.25)
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(names)

#     plt.savefig(output_pdf)

# ruff: noqa: T201
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


def plot_residual_and_offset(
    results: list[dict],
    output_pdf: Path,
    label_field: str | list[str] = "channels",
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

    diffs_and_resids = [
        (
            result["mean_offset"],
            result["max_abs_resid"],
            result["workspace"],
            label_for(result["workspace"], label_field),
        )
        for result in results
    ]

    sorted_results = sorted(diffs_and_resids, key=lambda x: x[1])   

    diffs, resids, workspaces, names = zip(*sorted_results)
    diffs = [abs(diff) for diff in diffs]
    x = range(len(sorted_results))

    ax1.scatter(x, diffs, label="diffs")
    ax1.set_title("mean offset by workspace")
    ax1.grid(True, alpha=0.25)

    ax2.scatter(x, resids, label="resids")
    ax2.set_yscale('log')
    ax2.set_title("max residual by workspace")
    ax2.grid(True, alpha=0.25)

    # Shared x-axis: setting ticks on the bottom axes is enough.
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    axis_label = (
        label_field if isinstance(label_field, str) else ", ".join(label_field)
    )
    ax2.set_xlabel(axis_label)

    fig.tight_layout()
    plt.savefig(output_pdf)
    return output_pdf



