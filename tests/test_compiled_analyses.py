"""Tests for opt-in cross-analysis compiled-template sharing."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor.compile.maker import function
from pytensor.graph.traversal import explicit_graph_inputs

from pyhs3 import Workspace
from pyhs3.compiled import (
    _AnalysisGraph,
    _compile_equivalent_group,
)
from pyhs3.model import Model


def _workspace() -> Workspace:
    return Workspace.model_validate(
        {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "g1",
                    "type": "gaussian_dist",
                    "x": "x1",
                    "mean": "mean1",
                    "sigma": 1.0,
                },
                {
                    "name": "g2",
                    "type": "gaussian_dist",
                    "x": "x2",
                    "mean": "mean2",
                    "sigma": 2.0,
                },
            ],
            "domains": [
                {
                    "name": "d1",
                    "type": "product_domain",
                    "axes": [{"name": "mean1", "min": -10.0, "max": 10.0}],
                },
                {
                    "name": "d2",
                    "type": "product_domain",
                    "axes": [{"name": "mean2", "min": -10.0, "max": 10.0}],
                },
            ],
            "data": [
                {
                    "name": "data1",
                    "type": "unbinned",
                    "axes": [{"name": "x1", "min": -10.0, "max": 10.0}],
                    "entries": [[0.0], [1.0], [2.0]],
                },
                {
                    "name": "data2",
                    "type": "unbinned",
                    "axes": [{"name": "x2", "min": -10.0, "max": 10.0}],
                    "entries": [[-1.0], [1.5], [3.0]],
                },
            ],
            "likelihoods": [
                {"name": "L1", "distributions": ["g1"], "data": ["data1"]},
                {"name": "L2", "distributions": ["g2"], "data": ["data2"]},
            ],
            "analyses": [
                {"name": "A1", "likelihood": "L1", "domains": ["d1"], "init": "p1"},
                {"name": "A2", "likelihood": "L2", "domains": ["d2"], "init": "p2"},
            ],
            "parameter_points": [
                {"name": "p1", "parameters": [{"name": "mean1", "value": 0.5}]},
                {"name": "p2", "parameters": [{"name": "mean2", "value": 1.0}]},
            ],
        }
    )


def _independent_value(workspace: Workspace, analysis_name: str) -> np.ndarray:
    model = workspace.model(
        workspace.analyses[analysis_name], progress=False, mode="FAST_COMPILE"
    )
    inputs = [
        variable
        for variable in explicit_graph_inputs([model.log_prob])
        if variable.name is not None
    ]
    compiled = function(
        inputs=inputs,
        outputs=model.log_prob,
        mode="FAST_COMPILE",
        on_unused_input="ignore",
        trust_input=True,
    )
    defaults = {**model.data, **model.free_params}
    return compiled(
        *(
            np.asarray(defaults[variable.name], dtype=variable.type.dtype)
            for variable in inputs
        )
    )


def _synthetic_graph(
    name: str,
    input_name: str,
    constant_values: tuple[int | float, ...],
) -> _AnalysisGraph:
    variable = pt.scalar(input_name)
    constants = tuple(pt.constant(value) for value in constant_values)

    expression = variable
    for constant in constants:
        expression = expression + constant

    model = cast(
        Model,
        SimpleNamespace(
            data={input_name: 1.0},
            free_params={},
        ),
    )

    return _AnalysisGraph(
        name=name,
        model=model,
        expression=expression,
        explicit_inputs=(variable,),
        constants=constants,
        skeleton="synthetic-equivalent-graph",
    )


def test_equivalent_analyses_share_one_compiled_function():
    workspace = _workspace()
    compiled = workspace.compile_analyses(mode="FAST_COMPILE")

    assert list(compiled) == ["A1", "A2"]
    assert compiled.compiled_function_count == 1
    np.testing.assert_allclose(compiled["A1"](), _independent_value(workspace, "A1"))
    np.testing.assert_allclose(compiled["A2"](), _independent_value(workspace, "A2"))


def test_analysis_specific_override_uses_original_input_name():
    workspace = _workspace()
    compiled = workspace.compile_analyses(mode="FAST_COMPILE")

    default = compiled["A2"]()
    overridden = compiled["A2"](mean2=2.5)
    assert not np.allclose(default, overridden)


def test_default_argument_fast_path_matches_explicit_override():
    workspace = _workspace()
    compiled = workspace.compile_analyses(mode="FAST_COMPILE")

    cached_default = compiled["A2"]()
    explicit_default = compiled["A2"](mean2=1.0)
    np.testing.assert_allclose(cached_default, explicit_default)


def test_subset_selection_by_name():
    workspace = _workspace()
    compiled = workspace.compile_analyses(["A2"], mode="FAST_COMPILE")

    assert list(compiled) == ["A2"]
    assert compiled.compiled_function_count == 1


def test_unknown_analysis_is_rejected():
    workspace = _workspace()

    with pytest.raises(KeyError, match="missing"):
        workspace.compile_analyses(["missing"], mode="FAST_COMPILE")


def test_missing_override_input_is_rejected():
    workspace = _workspace()
    compiled = workspace.compile_analyses(mode="FAST_COMPILE")
    analysis = compiled["A1"]

    sources = list(analysis._input_sources)
    input_index = next(
        index for index, (kind, _) in enumerate(sources) if kind == "input"
    )
    sources[input_index] = ("input", "missing")

    broken = replace(analysis, _input_sources=tuple(sources))

    with pytest.raises(KeyError, match="missing"):
        broken(mean1=0.5)


def test_different_constant_counts_fall_back_to_individual_compilation():
    first = _synthetic_graph("A1", "x1", (1.0,))
    second = _synthetic_graph("A2", "x2", (1.0, 2.0))

    compiled, function_count = _compile_equivalent_group(
        [first, second],
        "FAST_COMPILE",
    )

    assert list(compiled) == ["A1", "A2"]
    assert function_count == 2
    np.testing.assert_allclose(compiled["A1"](), 2.0)
    np.testing.assert_allclose(compiled["A2"](), 4.0)


def test_varying_integer_constants_fall_back_to_individual_compilation():
    first = _synthetic_graph("A1", "x1", (1,))
    second = _synthetic_graph("A2", "x2", (2,))

    compiled, function_count = _compile_equivalent_group(
        [first, second],
        "FAST_COMPILE",
    )

    assert list(compiled) == ["A1", "A2"]
    assert function_count == 2
    np.testing.assert_allclose(compiled["A1"](), 2.0)
    np.testing.assert_allclose(compiled["A2"](), 3.0)


def test_equal_constants_share_template_without_placeholders():
    first = _synthetic_graph("A1", "x1", (2.0,))
    second = _synthetic_graph("A2", "x2", (2.0,))

    compiled, function_count = _compile_equivalent_group(
        [first, second],
        "FAST_COMPILE",
    )

    assert function_count == 1
    np.testing.assert_allclose(compiled["A1"](), 3.0)
    np.testing.assert_allclose(compiled["A2"](), 3.0)


def test_workspace_without_analyses_is_rejected():
    workspace = Workspace.model_validate(
        {
            "metadata": {"hs3_version": "0.2"},
        }
    )

    with pytest.raises(ValueError, match="Workspace has no analyses"):
        workspace.compile_analyses(mode="FAST_COMPILE")


def test_empty_analysis_selection_is_rejected():
    workspace = _workspace()

    with pytest.raises(ValueError, match="At least one analysis is required"):
        workspace.compile_analyses([], mode="FAST_COMPILE")
