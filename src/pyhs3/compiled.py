"""Compilation helpers for structurally equivalent analysis models."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from pytensor.compile.maker import function
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import explicit_graph_inputs, graph_inputs

if TYPE_CHECKING:
    from pyhs3.analyses import Analysis
    from pyhs3.model import Model
    from pyhs3.workspace import Workspace


def _type_signature(variable: Any) -> tuple[Any, ...]:
    var_type = variable.type
    return (
        type(var_type).__module__,
        type(var_type).__qualname__,
        getattr(var_type, "dtype", None),
        getattr(var_type, "ndim", None),
        tuple(getattr(var_type, "shape", ()) or ()),
        tuple(getattr(var_type, "broadcastable", ()) or ()),
    )


def _op_signature(op: Any) -> tuple[Any, ...]:
    props = tuple(
        (name, repr(getattr(op, name, "<unreadable>")))
        for name in getattr(op, "__props__", ())
    )
    scalar_op = getattr(op, "scalar_op", None)
    scalar_signature = None
    if scalar_op is not None:
        scalar_signature = (
            type(scalar_op).__module__,
            type(scalar_op).__qualname__,
            str(scalar_op),
        )
    return (
        type(op).__module__,
        type(op).__qualname__,
        str(op),
        props,
        scalar_signature,
    )


@dataclass(frozen=True)
class _AnalysisGraph:
    name: str
    model: Model
    expression: Any
    explicit_inputs: tuple[Any, ...]
    constants: tuple[Constant[Any], ...]
    skeleton: str


def _analysis_graph(name: str, model: Model) -> _AnalysisGraph:
    expression = model.log_prob
    explicit_inputs = tuple(
        variable
        for variable in explicit_graph_inputs([expression])
        if variable.name is not None
    )
    constants = tuple(
        variable
        for variable in graph_inputs([expression])
        if isinstance(variable, Constant)
    )
    fgraph = FunctionGraph(
        inputs=list(explicit_inputs), outputs=[expression], clone=False
    )
    references: dict[Any, str] = {}
    roots: list[tuple[str, str, tuple[Any, ...]]] = []
    for variable in explicit_inputs:
        ref = f"i{len(roots)}"
        references[variable] = ref
        roots.append((ref, "input", _type_signature(variable)))
    for variable in constants:
        ref = f"c{len(roots)}"
        references[variable] = ref
        roots.append((ref, "constant", _type_signature(variable)))

    nodes = []
    for node_index, node in enumerate(fgraph.toposort()):
        input_refs = tuple(references[variable] for variable in node.inputs)
        output_refs = []
        for output_index, variable in enumerate(node.outputs):
            ref = f"n{node_index}o{output_index}"
            references[variable] = ref
            output_refs.append((ref, _type_signature(variable)))
        nodes.append((_op_signature(node.op), input_refs, tuple(output_refs)))
    payload = (
        tuple(roots),
        tuple(nodes),
        tuple(references[variable] for variable in fgraph.outputs),
    )
    skeleton = hashlib.sha256(repr(payload).encode()).hexdigest()
    return _AnalysisGraph(
        name=name,
        model=model,
        expression=expression,
        explicit_inputs=explicit_inputs,
        constants=constants,
        skeleton=skeleton,
    )


@dataclass(frozen=True)
class CompiledAnalysis:
    """Callable view of one analysis backed by a shared compiled function."""

    name: str
    model: Model
    _function: Any
    _template_inputs: tuple[Any, ...]
    _input_sources: tuple[tuple[str, Any], ...]
    _default_arguments: tuple[Any, ...]

    def __call__(
        self, **overrides: float | list[float] | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate with model defaults plus optional analysis-specific overrides."""
        if not overrides:
            result = self._function(*self._default_arguments)
            return cast(npt.NDArray[np.float64], result)

        defaults = {**self.model.data, **self.model.free_params, **overrides}
        values: list[Any] = []
        for kind, source in self._input_sources:
            if kind == "constant":
                values.append(source)
            else:
                if source not in defaults:
                    message = f"No value supplied for input {source!r}"
                    raise KeyError(message)
                variable = self._template_inputs[len(values)]
                values.append(np.asarray(defaults[source], dtype=variable.type.dtype))

        result = self._function(*values)
        return cast(npt.NDArray[np.float64], result)


class CompiledAnalyses(Mapping[str, CompiledAnalysis]):
    """Name-indexed compiled analyses, potentially sharing template functions."""

    def __init__(
        self,
        analyses: dict[str, CompiledAnalysis],
        *,
        compiled_function_count: int,
    ) -> None:
        """Initialize the analysis mapping and compiled-function count."""
        self._analyses = analyses
        self.compiled_function_count = compiled_function_count

    def __getitem__(self, name: str) -> CompiledAnalysis:
        return self._analyses[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._analyses)

    def __len__(self) -> int:
        return len(self._analyses)


def _constant_values_equal(left: Constant[Any], right: Constant[Any]) -> bool:
    return bool(
        np.array_equal(np.asarray(left.data), np.asarray(right.data), equal_nan=True)
    )


def _prepare_default_arguments(
    model: Model,
    template_inputs: tuple[Any, ...],
    input_sources: tuple[tuple[str, Any], ...],
) -> tuple[Any, ...]:
    defaults = {**model.data, **model.free_params}
    values = []
    for variable, (kind, source) in zip(template_inputs, input_sources, strict=True):
        if kind == "constant":
            values.append(source)
        else:
            if source not in defaults:
                message = f"No value supplied for input {source!r}"
                raise KeyError(message)
            values.append(np.asarray(defaults[source], dtype=variable.type.dtype))
    return tuple(values)


def _compile_individual(graph: _AnalysisGraph, mode: str) -> CompiledAnalysis:
    compiled = function(
        inputs=list(graph.explicit_inputs),
        outputs=graph.expression,
        mode=mode,
        on_unused_input="ignore",
        trust_input=True,
        name=f"analysis_{graph.name}",
    )
    sources = tuple(("input", variable.name) for variable in graph.explicit_inputs)
    default_arguments = _prepare_default_arguments(
        graph.model, graph.explicit_inputs, sources
    )
    return CompiledAnalysis(
        name=graph.name,
        model=graph.model,
        _function=compiled,
        _template_inputs=graph.explicit_inputs,
        _input_sources=sources,
        _default_arguments=default_arguments,
    )


def _compile_equivalent_group(
    graphs: Sequence[_AnalysisGraph], mode: str
) -> tuple[dict[str, CompiledAnalysis], int]:
    reference = graphs[0]
    if any(len(graph.constants) != len(reference.constants) for graph in graphs[1:]):
        return (
            {graph.name: _compile_individual(graph, mode) for graph in graphs},
            len(graphs),
        )

    varying_indices = [
        index
        for index, constant in enumerate(reference.constants)
        if not all(
            _constant_values_equal(constant, graph.constants[index])
            for graph in graphs[1:]
        )
    ]
    if any(
        not np.issubdtype(
            np.asarray(reference.constants[index].data).dtype, np.floating
        )
        for index in varying_indices
    ):
        return (
            {graph.name: _compile_individual(graph, mode) for graph in graphs},
            len(graphs),
        )

    replacements: dict[
        Variable[Any, Any],
        Variable[Any, Any],
    ] = {}
    placeholder_indices: dict[str, int] = {}

    for index in varying_indices:
        constant = reference.constants[index]
        placeholder_name = f"__analysis_template_constant_{index}"
        placeholder = cast(
            Variable[Any, Any],
            constant.type(name=placeholder_name),
        )
        replacements[constant] = placeholder
        placeholder_indices[placeholder_name] = index
    expression = (
        clone_replace([reference.expression], replace=replacements)[0]
        if replacements
        else reference.expression
    )
    template_inputs = tuple(
        variable
        for variable in explicit_graph_inputs([expression])
        if variable.name is not None
    )
    compiled = function(
        inputs=list(template_inputs),
        outputs=expression,
        mode=mode,
        on_unused_input="ignore",
        trust_input=True,
        name=f"analysis_template_{reference.skeleton[:12]}",
    )

    result = {}
    for graph in graphs:
        explicit_by_position = iter(graph.explicit_inputs)
        sources = []
        for variable in template_inputs:
            variable_name = variable.name
            constant_index = (
                placeholder_indices.get(variable_name)
                if variable_name is not None
                else None
            )
            if constant_index is not None:
                sources.append(
                    ("constant", np.asarray(graph.constants[constant_index].data))
                )
            else:
                original = next(explicit_by_position)
                sources.append(("input", original.name))
        input_sources = tuple(sources)
        default_arguments = _prepare_default_arguments(
            graph.model, template_inputs, input_sources
        )
        result[graph.name] = CompiledAnalysis(
            name=graph.name,
            model=graph.model,
            _function=compiled,
            _template_inputs=template_inputs,
            _input_sources=input_sources,
            _default_arguments=default_arguments,
        )
    return result, 1


def compile_analyses(
    workspace: Workspace,
    analyses: Sequence[str | Analysis] | None = None,
    *,
    parameter_set: str | None = None,
    progress: bool = False,
    mode: str = "FAST_RUN",
) -> CompiledAnalyses:
    """Compile analysis log-probabilities while sharing equivalent templates.

    Structurally incompatible graphs are compiled independently. Graphs are
    shared only when their pre-rewrite topology and types match; differing
    floating constants become runtime template inputs. Varying non-floating
    constants conservatively disable sharing for the affected group.
    """
    if not workspace.analyses:
        message = "Workspace has no analyses"
        raise ValueError(message)
    selected: list[Analysis] = []
    for item in analyses if analyses is not None else workspace.analyses:
        if isinstance(item, str):
            analysis = workspace.analyses.get(item)
            if analysis is None:
                message = f"Unknown analysis {item!r}"
                raise KeyError(message)
            selected.append(analysis)
        else:
            selected.append(item)
    if not selected:
        message = "At least one analysis is required"
        raise ValueError(message)

    graphs = []
    for analysis in selected:
        model = workspace.model(
            analysis,
            parameter_set=parameter_set,
            progress=progress,
            mode=mode,
        )
        graphs.append(_analysis_graph(analysis.name, model))

    groups: defaultdict[str, list[_AnalysisGraph]] = defaultdict(list)
    for graph in graphs:
        groups[graph.skeleton].append(graph)
    compiled = {}
    function_count = 0
    for group in groups.values():
        group_compiled, count = _compile_equivalent_group(group, mode)
        compiled.update(group_compiled)
        function_count += count
    return CompiledAnalyses(compiled, compiled_function_count=function_count)
