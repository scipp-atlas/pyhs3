"""Paper/figure-friendly rendering of a PyTensor computation graph.

:func:`pytensor.printing.pydotprint` is the only way to draw a graph, but its
labels always carry every dtype/shape annotation and toposort-collision id it
can, and large constant arrays (e.g. 64-point Gauss-Legendre quadrature nodes)
print as half-truncated number soup. This module post-processes pydotprint's
own ``.dot`` output with :mod:`pydot` to make those things independently
optional, and re-lays-out the graph afterward so node boxes shrink to fit the
shorter labels instead of keeping pydotprint's original (now oversized)
layout.

Used by :meth:`pyhs3.model.Model.visualize_graph`.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pydot

    from pyhs3.typing.aliases import TensorVar

OpParamMode = Literal["orig", "elide", "none"]
ConstArrayMode = Literal["orig", "truncate", "elide"]

# Ops that only exist to satisfy PyTensor's broadcasting machinery - they don't
# combine or transform values, so a reader gets nothing from seeing them.
# DimShuffle is ExpandDims's older name in some PyTensor versions.
DEFAULT_OP_PARAMS: Mapping[str, OpParamMode] = MappingProxyType(
    {"ExpandDims": "elide", "DimShuffle": "elide"}
)

# Matches a type annotation pydotprint appends to unnamed vars/constants, e.g.
# "Scalar(float32, shape=())" or "Matrix(float64, shape=(1, 1))". Restricted to
# parens whose contents mention a dtype or "shape=" so we never eat legitimate
# op parameters like "ExpandDims{axes=(0, 1)}" (braces, not parens) or a
# hypothetical "SomeOp(axis=-1)".
_TYPE_SUFFIX_RE = re.compile(r"\s*([A-Za-z_]\w*)\(((?:[^()]|\([^()]*\))*)\)")
_DTYPE_HINT_RE = re.compile(r"(float\d+|u?int\d+|bool)\b|shape=")
_TYPE_INNER_RE = re.compile(r"^(?P<dtype>\w+),\s*shape=(?P<shape>\([^()]*\))$")


def _op_name(raw_label: str) -> str:
    """Extract the bare op name pydotprint's apply-node label starts with.

    "ExpandDims{axes=(0, 1)} id=6" -> "ExpandDims"; "Add" -> "Add".
    """
    return raw_label.split("{", maxsplit=1)[0].split(" id=", maxsplit=1)[0].strip()


def _resolve_op_mode(
    op_params: OpParamMode | Mapping[str, OpParamMode], op_name: str
) -> OpParamMode:
    if isinstance(op_params, str):
        return op_params
    return op_params.get(op_name, "orig")


def _render_type_suffix(
    match: re.Match[str], *, show_dtype: bool, show_shape: bool
) -> str:
    """Rewrite one "TypeName(dtype, shape=X)" match per the dtype/shape knobs."""
    whole = match.group(0)
    if not _DTYPE_HINT_RE.search(whole):
        return (
            whole  # not a dtype/shape annotation - leave untouched (e.g. an op param)
        )

    leading_space = whole[: len(whole) - len(whole.lstrip())]
    typename, inner = match.group(1), match.group(2)
    parsed = _TYPE_INNER_RE.match(inner)
    if parsed is None:
        # Unexpected format (e.g. a future PyTensor type repr) - leave as-is
        # rather than guess and risk mangling the label.
        return whole

    if show_dtype and show_shape:
        return whole
    if show_dtype:
        return f"{leading_space}{typename}({parsed['dtype']})"
    if show_shape:
        return f"{leading_space}{typename}(shape={parsed['shape']})"
    return ""  # drop the annotation (and its typename) entirely


def _clean_label(
    label: str, *, show_id: bool, show_dtype: bool, show_shape: bool
) -> str:
    """Apply the show_id/show_dtype/show_shape knobs to one node/edge label."""
    label = _TYPE_SUFFIX_RE.sub(
        lambda m: _render_type_suffix(m, show_dtype=show_dtype, show_shape=show_shape),
        label,
    )
    label = re.sub(r"^val=", "", label)
    label = re.sub(r"^name=", "", label)
    if not show_id:
        label = re.sub(r"\s+id=\d+$", "", label)  # pydotprint's toposort-index suffix
    return label.strip()


def _strip_op_params(label: str) -> str:
    """Drop an op's parameter block, e.g. "ExpandDims{axes=(0, 1)}" -> "ExpandDims"."""
    return re.sub(r"\{[^}]*\}", "", label)


def _elide_nodes(
    graph: pydot.Dot, op_params: OpParamMode | Mapping[str, OpParamMode]
) -> None:
    """Splice single-input plumbing nodes (e.g. ExpandDims) out of the drawing.

    Rewires each qualifying node's outgoing edges to originate from its own
    (sole) input instead, then deletes the node - a purely visual contraction.
    Nodes with more than one incoming edge are left alone rather than guessed
    at, since a splice only makes unambiguous sense for single-input ops.
    """
    while True:
        for node in graph.get_nodes():
            label = node.get_label()
            if label is None or node.get_shape() != "ellipse":
                continue
            op_name = _op_name(label.strip('"'))
            if _resolve_op_mode(op_params, op_name) != "elide":
                continue
            name = node.get_name()
            incoming = [e for e in graph.get_edges() if e.get_destination() == name]
            if len(incoming) != 1:
                continue
            outgoing = [e for e in graph.get_edges() if e.get_source() == name]
            new_source = incoming[0].get_source()
            for edge in outgoing:
                edge.obj_dict["points"] = (new_source, edge.get_destination())
            graph.del_edge(incoming[0].get_source(), incoming[0].get_destination())
            graph.del_node(name)
            break  # node/edge lists changed - restart the scan
        else:
            return


def _reset_layout(graph: pydot.Dot) -> None:
    """Drop pydotprint's precomputed layout.

    Node boxes/positions were sized for the original dtype/shape-annotated
    labels; once those shrink, `dot` needs to re-lay-out the graph from
    scratch rather than leave oversized boxes and whitespace behind.
    """
    graph.obj_dict["attributes"].pop("bb", None)
    for element in (*graph.get_nodes(), *graph.get_edges()):
        attrs = element.obj_dict["attributes"]
        for key in ("pos", "width", "height", "lp"):
            attrs.pop(key, None)


def _clean_labels(
    graph: pydot.Dot,
    *,
    op_params: OpParamMode | Mapping[str, OpParamMode],
    show_id: bool,
    show_dtype: bool,
    show_shape: bool,
) -> None:
    for node in graph.get_nodes():
        label = node.get_label()
        if label is None:
            continue
        label = label.strip('"')
        label = _clean_label(
            label, show_id=show_id, show_dtype=show_dtype, show_shape=show_shape
        )
        if node.get_shape() == "ellipse":
            mode = _resolve_op_mode(op_params, _op_name(label))
            if mode == "none":
                label = _strip_op_params(label)
        node.set_label(f'"{label}"' if label else '""')

    for edge in graph.get_edges():
        label = edge.get_label()
        if label is None:
            continue
        label = label.strip('"')
        label = _clean_label(
            label, show_id=show_id, show_dtype=show_dtype, show_shape=show_shape
        )
        if not label:
            edge.obj_dict["attributes"].pop("label", None)
        else:
            edge.set_label(f'"{label}"')


def _const_array_label(data: np.ndarray, mode: ConstArrayMode) -> str:
    shape_str = ",".join(str(d) for d in data.shape)
    if mode == "elide":
        return f"const[{shape_str}]"
    if mode == "truncate":
        preview = ", ".join(f"{v:.3g}" for v in data.ravel()[:3])
        return f"[{preview}, ...] ({shape_str})"
    msg = f"Unknown const_arrays mode: {mode!r}"
    raise ValueError(msg)


def _prepare_constants(
    var: TensorVar, *, const_arrays: ConstArrayMode, threshold: int
) -> TensorVar:
    """Clone the graph and give oversized constant arrays a compact name.

    Cloning is required so this never mutates the model's real graph -
    pydotprint would otherwise be handed (and could rename) the live nodes
    the compiled model still uses for evaluation. Renaming has to go through
    ``fg.replace`` with a freshly-constructed ``Constant`` rather than setting
    ``.name`` directly: ``FunctionGraph(clone=True)`` does not deep-copy
    ``Constant`` nodes (they're treated as immutable and shared), so mutating
    ``.name`` on one found via the "clone" would mutate the original graph's
    constant in place.
    """
    if const_arrays == "orig":
        return var

    fg = FunctionGraph(outputs=[var], clone=True)
    for node in list(fg.toposort()):
        for v in node.inputs:
            if isinstance(v, Constant) and v.name is None:
                data = np.asarray(v.data)
                if data.size > threshold:
                    label = _const_array_label(data, const_arrays)
                    fg.replace(v, Constant(type=v.type, data=v.data, name=label))
    return cast("TensorVar", fg.outputs[0])


def render_graph(
    var: TensorVar,
    outfile: str,
    fmt: str = "svg",
    *,
    op_params: OpParamMode | Mapping[str, OpParamMode] = DEFAULT_OP_PARAMS,
    show_id: bool = False,
    show_dtype: bool = False,
    show_shape: bool = False,
    const_arrays: ConstArrayMode = "elide",
    const_array_threshold: int = 8,
) -> str:
    """Render a PyTensor graph to an image, with reader-friendly labels.

    Args:
        var: The graph output to visualize (e.g. ``model.distributions[name]``).
        outfile: Destination image path.
        fmt: Output format passed to graphviz ("svg", "png", "pdf").
        op_params: Per-op-name display mode, or one mode applied to every op.
            ``"orig"`` keeps the op's full label (e.g. ``Sum{axis=0}``);
            ``"none"`` drops the parameter block (``Sum``); ``"elide"``
            removes the node from the drawing entirely, rewiring its
            consumers to its own (sole) input - only sound for single-input
            ops. Op names absent from a mapping default to ``"orig"``.
        show_id: Append each op's toposort index to its label
            (e.g. ``Add id=9``).
        show_dtype: Keep the dtype portion of PyTensor's type annotations
            (e.g. ``Matrix(float32)``).
        show_shape: Keep the shape portion of PyTensor's type annotations
            (e.g. ``Matrix(shape=(1, 1))``).
        const_arrays: How to render constant arrays larger than
            ``const_array_threshold`` elements. ``"orig"`` leaves pydotprint's
            own (possibly mid-token-truncated) value dump untouched;
            ``"truncate"`` shows a deliberate ``[v0, v1, v2, ...] (shape)``
            preview; ``"elide"`` collapses it to ``const[shape]``.
        const_array_threshold: Element count above which ``const_arrays``
            applies.

    Returns:
        The path the image was written to.
    """
    import pydot  # noqa: PLC0415  # pylint: disable=import-outside-toplevel,import-error
    from pytensor.printing import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        pydotprint,
    )

    var = _prepare_constants(
        var, const_arrays=const_arrays, threshold=const_array_threshold
    )

    # A name derived from outfile (e.g. outfile.with_suffix(".dot")) can equal
    # outfile itself when outfile already ends in ".dot" - then the cleanup
    # unlink below would delete the very file we just rendered. mkstemp
    # guarantees a name distinct from outfile regardless of its suffix.
    fd, tmp_name = tempfile.mkstemp(suffix=".dot", dir=str(Path(outfile).parent))
    os.close(fd)
    tmp_dot = Path(tmp_name)
    try:
        pydotprint(
            var,
            outfile=str(tmp_dot),
            format="dot",
            with_ids=show_id,
            high_contrast=True,
            var_with_name_simple=True,
            compact=True,
            print_output_file=False,
        )

        (graph,) = pydot.graph_from_dot_file(str(tmp_dot))
        _elide_nodes(graph, op_params)
        _reset_layout(graph)
        _clean_labels(
            graph,
            op_params=op_params,
            show_id=show_id,
            show_dtype=show_dtype,
            show_shape=show_shape,
        )

        graph.write(outfile, prog="dot", format=fmt)
    finally:
        tmp_dot.unlink(missing_ok=True)
    return outfile
