"""Unit tests for pyhs3.tensorutils graph helpers."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function
from pytensor.graph.traversal import explicit_graph_inputs, io_toposort

from pyhs3.tensorutils import merge_duplicate_nodes


def _n_nodes(output):
    return len(io_toposort([], [output]))


class TestMergeDuplicateNodes:
    """merge_duplicate_nodes builds each structurally identical subgraph once."""

    def test_duplicate_subexpressions_are_built_once(self):
        """Two separately built copies of the same expression become one."""
        x = pt.dscalar("x")
        y = pt.dscalar("y")
        # exp(x * y) built twice, independently
        first = pt.exp(x * y)
        second = pt.exp(x * y)
        output = first + second
        assert _n_nodes(output) == 5  # 2 Mul, 2 Exp, 1 Add

        merged = merge_duplicate_nodes(output)

        assert _n_nodes(merged) == 3  # 1 Mul, 1 Exp, 1 Add
        fn = function([x, y], merged)
        np.testing.assert_allclose(fn(0.3, 2.0), 2 * np.exp(0.6))

    def test_equal_constants_are_merged(self):
        """Constants with the same type and value are treated as the same input."""
        x = pt.dvector("x")
        output = (x + pt.constant(2.0)) * (x + pt.constant(2.0))

        merged = merge_duplicate_nodes(output)

        adds = [
            node
            for node in io_toposort([], [merged])
            if type(getattr(node.op, "scalar_op", None)).__name__ == "Add"
        ]
        assert len(adds) == 1
        fn = function([x], merged)
        np.testing.assert_allclose(fn(np.array([1.0, -1.0])), [9.0, 1.0])

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            (pt.constant(2.0), pt.constant(3.0)),
            (pt.constant(np.float64(2.0)), pt.constant(np.float32(2.0))),
            (pt.constant(np.array([2.0])), pt.constant(np.array([2.0, 2.0]))),
        ],
    )
    def test_different_constants_are_kept_apart(self, left, right):
        """Constants differing in value, dtype or shape are not merged."""
        x = pt.dscalar("x")
        output = (x + left).sum() + (x + right).sum()

        merged = merge_duplicate_nodes(output)

        # Both constants still feed the merged graph as separate inputs.
        used = {id(var) for node in io_toposort([], [merged]) for var in node.inputs}
        assert id(left) in used
        assert id(right) in used

    def test_graph_inputs_are_preserved(self):
        """The merged graph depends on the very same input variables."""
        x = pt.dvector("x")
        mu = pt.dscalar("mu")
        output = ((x - mu) ** 2).sum() + ((x - mu) ** 2).sum()

        merged = merge_duplicate_nodes(output)

        assert set(explicit_graph_inputs([merged])) == {x, mu}

    def test_graph_without_duplicates_is_returned_unchanged(self):
        """With nothing to merge, the original output variable comes back."""
        x = pt.dscalar("x")
        output = pt.exp(x) + pt.log(x)

        assert merge_duplicate_nodes(output) is output

    def test_input_variable_is_returned_unchanged(self):
        """An output with no apply nodes is returned as is."""
        x = pt.dscalar("x")
        assert merge_duplicate_nodes(x) is x

    def test_kept_constants_are_not_merged(self):
        """A constant in ``keep`` stays its own input, even next to an equal one."""
        x = pt.dvector("x")
        kept = pt.constant(np.array([1.0, 2.0]), name="kept")
        equal = pt.constant(np.array([1.0, 2.0]))
        output = ((x + kept) * (x + equal)).sum()

        merged = merge_duplicate_nodes(output, keep=[kept])

        used = {id(var) for node in io_toposort([], [merged]) for var in node.inputs}
        assert id(kept) in used
        assert id(equal) in used
