"""Unit tests for pyhs3.graph_viz, the paper-friendly graph renderer."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pydot
import pytensor.tensor as pt
import pytest
from pytensor.graph.basic import Constant
from pytensor.graph.fg import FunctionGraph

from pyhs3.graph_viz import (
    _clean_label,
    _const_array_label,
    _elide_nodes,
    _op_name,
    _prepare_constants,
    _reset_layout,
    _resolve_op_mode,
    _strip_op_params,
    render_graph,
)

pytestmark = pytest.mark.pydot


def _constants_by_size(var):
    """All Constant inputs across var's graph, keyed by their element count."""
    fg = FunctionGraph(outputs=[var], clone=False)
    return {
        v.data.size: v
        for node in fg.toposort()
        for v in node.inputs
        if isinstance(v, Constant)
    }


class TestOpName:
    """Tests for _op_name: extracting the bare op name from a pydotprint label."""

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Add", "Add"),
            ("Add id=111", "Add"),
            ("ExpandDims{axes=(0, 1)}", "ExpandDims"),
            ("ExpandDims{axes=(0, 1)} id=6", "ExpandDims"),
            ("Sum{axis=0}", "Sum"),
        ],
    )
    def test_extracts_bare_name(self, label, expected):
        assert _op_name(label) == expected


class TestResolveOpMode:
    """Tests for _resolve_op_mode: op_params as a single mode vs a per-op mapping."""

    def test_single_mode_applies_to_every_op(self):
        assert _resolve_op_mode("none", "Sum") == "none"
        assert _resolve_op_mode("none", "ExpandDims") == "none"

    def test_mapping_looks_up_by_op_name(self):
        mapping = {"ExpandDims": "elide"}
        assert _resolve_op_mode(mapping, "ExpandDims") == "elide"

    def test_mapping_defaults_to_orig_for_unlisted_ops(self):
        mapping = {"ExpandDims": "elide"}
        assert _resolve_op_mode(mapping, "Sum") == "orig"


class TestStripOpParams:
    def test_drops_parameter_block(self):
        assert _strip_op_params("ExpandDims{axes=(0, 1)}") == "ExpandDims"
        assert _strip_op_params("Sum{axis=0}") == "Sum"

    def test_leaves_bare_names_untouched(self):
        assert _strip_op_params("Add") == "Add"


class TestCleanLabel:
    """Tests for _clean_label: the show_id/show_dtype/show_shape knobs."""

    @pytest.mark.parametrize(
        ("show_dtype", "show_shape", "expected"),
        [
            (False, False, "0"),
            (True, False, "0 Matrix(float32)"),
            (False, True, "0 Matrix(shape=(1, 1))"),
            (True, True, "0 Matrix(float32, shape=(1, 1))"),
        ],
    )
    def test_dtype_shape_combinations_on_edge_label(
        self, show_dtype, show_shape, expected
    ):
        label = "0 Matrix(float32, shape=(1, 1))"
        assert (
            _clean_label(
                label, show_id=False, show_dtype=show_dtype, show_shape=show_shape
            )
            == expected
        )

    def test_drops_val_prefix(self):
        label = "val=1.0 Scalar(float32, shape=())"
        assert (
            _clean_label(label, show_id=False, show_dtype=False, show_shape=False)
            == "1.0"
        )

    def test_drops_name_prefix(self):
        label = "name=mu1 Scalar(float64, shape=())"
        assert (
            _clean_label(label, show_id=False, show_dtype=False, show_shape=False)
            == "mu1"
        )

    def test_show_id_false_strips_id_suffix(self):
        assert (
            _clean_label(
                "Add id=111", show_id=False, show_dtype=False, show_shape=False
            )
            == "Add"
        )

    def test_show_id_true_keeps_id_suffix(self):
        assert (
            _clean_label("Add id=111", show_id=True, show_dtype=False, show_shape=False)
            == "Add id=111"
        )

    def test_op_param_braces_are_not_mistaken_for_type_annotations(self):
        # Curly braces, not parens - _TYPE_SUFFIX_RE must never touch these.
        label = "ExpandDims{axes=(0, 1)}"
        assert (
            _clean_label(label, show_id=False, show_dtype=False, show_shape=False)
            == label
        )

    def test_parenthesized_text_without_a_dtype_hint_is_left_untouched(self):
        # Matches _TYPE_SUFFIX_RE's "word(...)" shape but has no dtype/shape=
        # hint inside - a hypothetical op param, not a type annotation.
        label = "SomeOp(axis=-1)"
        assert (
            _clean_label(label, show_id=False, show_dtype=False, show_shape=False)
            == label
        )

    def test_unparseable_type_inner_is_left_untouched(self):
        # Contains a dtype hint ("float32") but not the "dtype, shape=(...)"
        # format _TYPE_INNER_RE expects - e.g. a future PyTensor type repr.
        # Left as-is rather than guessed at, regardless of the knobs.
        label = "0 Matrix(float32)"
        assert (
            _clean_label(label, show_id=False, show_dtype=False, show_shape=False)
            == label
        )
        assert (
            _clean_label(label, show_id=False, show_dtype=True, show_shape=True)
            == label
        )


class TestConstArrayLabel:
    def test_elide_shows_shape_only(self):
        data = np.zeros((64,))
        assert _const_array_label(data, "elide") == "const[64]"

    def test_truncate_shows_value_preview_and_shape(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        label = _const_array_label(data, "truncate")
        assert label.startswith("[1, 2, 3, ...]")
        assert "(5)" in label

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown const_arrays mode"):
            _const_array_label(np.zeros(10), "bogus")


class TestPrepareConstants:
    def test_orig_mode_returns_same_variable_unchanged(self):
        var = pt.constant(np.arange(20.0)) + pt.scalar("x")
        out = _prepare_constants(var, const_arrays="orig", threshold=8)
        assert out is var

    def test_elide_renames_large_constant_on_a_clone_without_mutating_original(self):
        big_const = pt.constant(np.arange(20.0))
        var = big_const + pt.scalar("x")

        out = _prepare_constants(var, const_arrays="elide", threshold=8)

        assert big_const.name is None  # original graph untouched
        cloned_const = _constants_by_size(out)[20]
        assert cloned_const.name == "const[20]"

    def test_small_constants_are_left_unnamed(self):
        var = pt.constant(np.arange(4.0)) + pt.scalar("x")
        out = _prepare_constants(var, const_arrays="elide", threshold=8)
        assert _constants_by_size(out)[4].name is None


class TestElideNodes:
    def _make_graph(self, *, n_incoming: int) -> pydot.Dot:
        graph = pydot.Dot(graph_type="digraph")
        graph.add_node(pydot.Node("producer", label='"Log"', shape="ellipse"))
        graph.add_node(
            pydot.Node("target", label='"ExpandDims{axes=(0, 1)}"', shape="ellipse")
        )
        graph.add_node(pydot.Node("consumer", label='"Add"', shape="ellipse"))
        graph.add_edge(pydot.Edge("producer", "target"))
        if n_incoming == 2:
            graph.add_node(pydot.Node("other", label='"Log"', shape="ellipse"))
            graph.add_edge(pydot.Edge("other", "target"))
        graph.add_edge(pydot.Edge("target", "consumer", label='"1"'))
        return graph

    def test_splices_single_input_node_out(self):
        graph = self._make_graph(n_incoming=1)
        _elide_nodes(graph, {"ExpandDims": "elide"})

        names = {n.get_name() for n in graph.get_nodes()}
        assert "target" not in names

        edges = graph.get_edges()
        assert len(edges) == 1
        assert edges[0].get_source() == "producer"
        assert edges[0].get_destination() == "consumer"
        assert edges[0].get_label() == '"1"'  # preserved from the outgoing edge

    def test_leaves_multi_input_node_alone(self):
        graph = self._make_graph(n_incoming=2)
        _elide_nodes(graph, {"ExpandDims": "elide"})

        names = {n.get_name() for n in graph.get_nodes()}
        assert "target" in names

    def test_orig_mode_leaves_node_alone(self):
        graph = self._make_graph(n_incoming=1)
        _elide_nodes(graph, {"ExpandDims": "orig"})

        names = {n.get_name() for n in graph.get_nodes()}
        assert "target" in names


class TestResetLayout:
    def test_drops_layout_attributes(self):
        graph = pydot.Dot(graph_type="digraph")
        graph.obj_dict["attributes"]["bb"] = "0,0,100,100"
        graph.add_node(pydot.Node("n", pos='"1,1"', width="0.5", height="0.5"))

        _reset_layout(graph)

        assert "bb" not in graph.obj_dict["attributes"]
        attrs = graph.get_node("n")[0].obj_dict["attributes"]
        assert "pos" not in attrs
        assert "width" not in attrs
        assert "height" not in attrs


class TestRenderGraphEndToEnd:
    """render_graph() against small real PyTensor graphs, checking svg text."""

    def test_default_strips_dtype_shape_id_and_elides_expand_dims(self, tmp_path):
        x = pt.scalar("x")
        wide = pt.expand_dims(x, (0, 1))
        out = wide + pt.constant(1.0)

        outfile = render_graph(out, str(tmp_path / "g.svg"))
        svg = Path(outfile).read_text()

        assert "ExpandDims" not in svg
        assert "Scalar" not in svg
        assert "shape=" not in svg
        # our injected "Add id=111"-style suffix, not graphviz's own id="node1"
        # SVG attributes (always quoted, never followed directly by a digit)
        assert not re.search(r"id=\d", svg)

    def test_show_id_true_keeps_toposort_index(self, tmp_path):
        x = pt.scalar("x")
        out = x + x + x  # two Adds -> distinguishable ids

        outfile = render_graph(out, str(tmp_path / "g.svg"), show_id=True)
        svg = Path(outfile).read_text()

        # our injected "Add id=111"-style suffix, not graphviz's own id="node1"
        # SVG attributes (always quoted, never followed directly by a digit)
        assert re.search(r"id=\d", svg)

    def test_show_dtype_and_show_shape_bring_type_info_back(self, tmp_path):
        x = pt.scalar("x", dtype="float64")
        out = x + pt.constant(1.0)

        outfile = render_graph(
            out, str(tmp_path / "g.svg"), show_dtype=True, show_shape=True
        )
        svg = Path(outfile).read_text()

        assert "float64" in svg
        assert "shape=" in svg

    def test_const_arrays_elide_collapses_large_constant(self, tmp_path):
        out = pt.constant(np.arange(20.0)) + pt.vector("v")

        outfile = render_graph(
            out, str(tmp_path / "g.svg"), const_arrays="elide", const_array_threshold=8
        )
        svg = Path(outfile).read_text()

        assert "const[20]" in svg

    def test_const_arrays_orig_leaves_pydotprint_default(self, tmp_path):
        out = pt.constant(np.arange(20.0)) + pt.vector("v")

        outfile = render_graph(out, str(tmp_path / "g.svg"), const_arrays="orig")
        svg = Path(outfile).read_text()

        assert "const[20]" not in svg

    def test_op_params_none_strips_parameter_block(self, tmp_path):
        v = pt.vector("v")
        out = pt.sum(v, axis=0)

        outfile = render_graph(out, str(tmp_path / "g.svg"), op_params="none")
        svg = Path(outfile).read_text()

        assert "Sum" in svg
        assert "axis" not in svg

    def test_op_params_orig_keeps_parameter_block_by_default(self, tmp_path):
        v = pt.vector("v")
        out = pt.sum(v, axis=0)

        outfile = render_graph(out, str(tmp_path / "g.svg"))  # default op_params
        svg = Path(outfile).read_text()

        assert "axis" in svg

    def test_outfile_ending_in_dot_is_not_deleted(self, tmp_path):
        # The internal scratch file is also a ".dot" file; outfile must not
        # collide with it and get unlinked as cleanup.
        out = pt.scalar("x") + pt.constant(1.0)

        outfile = render_graph(out, str(tmp_path / "graph.dot"), fmt="dot")

        assert Path(outfile).exists()
        assert "digraph" in Path(outfile).read_text()
