from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
import rustworkx as rx
from pytensor.compile.function import function
from pytensor.graph.basic import applys_between, graph_inputs
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pyhs3 import typing as T
from pyhs3.distributions import DistributionSet
from pyhs3.functions import FunctionSet
from pyhs3.typing_compat import TypeAlias

log = logging.getLogger(__name__)

TDefault = TypeVar("TDefault")

if sys.version_info >= (3, 10):
    Axis: TypeAlias = tuple[float | None, float | None]
else:
    Axis: TypeAlias = tuple["float | None", "float | None"]


class Workspace:
    """
    Workspace for managing HS3 model specifications.

    A workspace contains parameter points, distributions, domains, and functions
    that define a probabilistic model. It provides methods to construct Model
    objects with specific parameter values and domain constraints.

    Attributes:
        parameter_collection (ParameterCollection): Named parameter sets.
        distribution_set (DistributionSet): Available distributions.
        domain_collection (DomainCollection): Domain constraints for parameters.
        function_set (FunctionSet): Available functions for parameter computation.
    """

    def __init__(self, spec: T.HS3Spec):
        """
        Manages the overall structure of the model including parameters, domains, and distributions.

        Args:
            spec (dict): A dictionary containing model definitions including parameter points, distributions,
                and domains.

        Attributes:
            parameter_collection (ParameterCollection): Set of named parameter points.
            distribution_set (DistributionSet): All distributions used in the workspace.
            domain_collection (DomainCollection): Domain definitions for all parameters.
        """

        self.parameter_collection = ParameterCollection(
            spec.get("parameter_points", [])
        )
        self.distribution_set = DistributionSet(spec.get("distributions", []))
        self.domain_collection = DomainCollection(spec.get("domains", []))
        self.function_set = FunctionSet(spec.get("functions", []))

    def model(
        self,
        *,
        domain: int | str | DomainSet = 0,
        parameter_point: int | str | ParameterSet = 0,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ) -> Model:
        """
        Constructs a `Model` object using the provided domain and parameter point.

        Args:
            domain (int | str | DomainSet): Identifier or object specifying the domain to use.
            parameter_point (int | str | ParameterSet): Identifier or object specifying the parameter values to use.
            progress (bool): Whether to show progress bar during dependency graph construction. Defaults to True.
            mode (str): PyTensor compilation mode. Defaults to "FAST_RUN".
                       Options: "FAST_RUN" (apply all rewrites, use C implementations),
                       "FAST_COMPILE" (few rewrites, Python implementations),
                       "NUMBA" (compile using Numba), "JAX" (compile using JAX),
                       "PYTORCH" (compile using PyTorch), "DebugMode" (debugging),
                       "NanGuardMode" (NaN detection).

        Returns:
            Model: The constructed model object.
        """

        domainset = (
            domain if isinstance(domain, DomainSet) else self.domain_collection[domain]
        )
        parameterset = (
            parameter_point
            if isinstance(parameter_point, ParameterSet)
            else self.parameter_collection[parameter_point]
        )

        # Verify that domains are a subset of parameters (not all parameters need bounds)
        param_names = set(parameterset.points.keys())
        domain_names = set(domainset.domains.keys())
        assert domain_names.issubset(param_names), (
            f"Domain names must be a subset of parameter names. "
            f"Extra domains: {domain_names - param_names}"
        )

        return Model(
            parameterset=parameterset,
            distributions=self.distribution_set,
            domains=domainset,
            functions=self.function_set,
            progress=progress,
            mode=mode,
        )


class Model:
    """
    Probabilistic model with compiled tensor operations.

    A model represents a specific instantiation of a workspace with concrete
    parameter values and domain constraints. It builds symbolic computation
    graphs for distributions and functions, with optional compilation for
    performance optimization.

    The model handles dependency resolution between parameters, functions,
    and distributions, ensuring proper evaluation order through topological
    sorting of the computation graph.
    """

    def __init__(
        self,
        *,
        parameterset: ParameterSet,
        distributions: DistributionSet,
        domains: DomainSet,
        functions: FunctionSet,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ):
        """
        Represents a probabilistic model composed of parameters, domains, distributions, and functions.

        Args:
            parameterset (ParameterSet): The parameter set used in the model.
            distributions (DistributionSet): Set of distributions to include.
            domains (DomainSet): Domain constraints for parameters.
            functions (FunctionSet): Set of functions that compute parameter values.
            progress (bool): Whether to show progress bar during dependency graph construction.
            mode (str): PyTensor compilation mode. Defaults to "FAST_RUN".
                       Options: "FAST_RUN" (apply all rewrites, use C implementations),
                       "FAST_COMPILE" (few rewrites, Python implementations),
                       "NUMBA" (compile using Numba), "JAX" (compile using JAX),
                       "PYTORCH" (compile using PyTorch), "DebugMode" (debugging),
                       "NanGuardMode" (NaN detection).

        Attributes:
            parameters (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic parameter variables.
            parameterset (ParameterSet): The original set of parameter values.
            distributions (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic distribution expressions.
            functions (dict[str, pytensor.tensor.variable.TensorVariable]): Computed function values.
            mode (str): PyTensor compilation mode.
            _compiled_functions (dict[str, Callable[..., npt.NDArray[np.float64]]]): Cache of compiled PyTensor functions.
        """
        self.parameters = {}
        self.parameterset = parameterset
        self.functions: dict[str, T.TensorVar] = {}
        self.mode = mode
        self._compiled_functions: dict[str, Callable[..., npt.NDArray[np.float64]]] = {}

        for parameter_point in parameterset:
            # Create scalar parameter with domain bounds applied
            domain = domains.domains.get(parameter_point.name, (None, None))
            self.parameters[parameter_point.name] = boundedscalar(
                parameter_point.name, domain
            )

        self.distributions: dict[str, T.TensorVar] = {}

        # Build dependency graph with proper entity identification
        self._build_dependency_graph(functions, distributions, progress)

    def _build_dependency_graph(
        self,
        functions: FunctionSet,
        distributions: DistributionSet,
        progress: bool = True,
    ) -> None:
        """
        Build and evaluate dependency graph for functions and distributions.

        This method properly handles cross-references between functions, distributions,
        and parameters by building a complete dependency graph first, then evaluating
        in topological order.
        """
        graph = rx.PyDiGraph()
        nodes: dict[str, int] = {}

        # Build entity type mapping for O(1) lookup
        entity_types: dict[str, str] = {}
        # Build constants mapping for O(1) lookup
        constants_map: dict[str, T.TensorVar] = {}

        # Map all parameter names
        for param in self.parameterset:
            entity_types[param.name] = "parameter"

        # Map all function names
        for func in functions:
            entity_types[func.name] = "function"

        # Map all distribution names and collect their constants
        for dist in distributions:
            entity_types[dist.name] = "distribution"
            # Also map any constants generated by this distribution
            for constant_name, constant_tensor in dist.constants.items():
                entity_types[constant_name] = "constant"
                constants_map[constant_name] = constant_tensor

        # First pass: Add all nodes to the graph using entity_types
        for entity_name, entity_type in entity_types.items():
            node_idx = graph.add_node({"type": entity_type, "name": entity_name})
            nodes[entity_name] = node_idx

        # Second pass: Add edges by iterating through all computational entities
        # Both functions and distributions have .parameters, so treat them uniformly
        for entity in [*functions, *distributions]:
            entity_idx = nodes[entity.name]

            # Add dependencies (parameters this entity depends on)
            for param_name in entity.parameters:
                try:
                    param_idx = nodes[param_name]
                except KeyError:
                    msg = (
                        f"Unknown entity referenced: '{param_name}' from '{entity.name}'. "
                        f"Not found in parameters, functions, or distributions."
                    )
                    raise ValueError(msg) from None

                # Add edge: dependency -> entity (param/func/dist feeds into entity)
                graph.add_edge(param_idx, entity_idx, None)

        # Third pass: Evaluate in topological order
        try:
            sorted_nodes = rx.topological_sort(graph)
        except rx.DAGHasCycle as e:
            msg = "Circular dependency detected in model"
            raise ValueError(msg) from e

        # Evaluate nodes in topological order with optional progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", style="cyan"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
            transient=True,  # Progress bar disappears when finished
            disable=not progress,  # Disable progress bar if progress=False
        ) as progress_bar:
            task = progress_bar.add_task(
                "Building expressions...", total=len(sorted_nodes)
            )

            for node_idx in sorted_nodes:
                node_data = graph[node_idx]
                node_type = node_data["type"]
                node_name = node_data["name"]

                # Truncate long names to prevent jumpiness
                max_name_length = 60
                display_name = node_name
                if len(node_name) > max_name_length:
                    display_name = node_name[: max_name_length - 3] + "..."

                # Update progress description with current entity (fixed width)
                progress_bar.update(
                    task,
                    description=f"Building {node_type:<12}: {display_name:<{max_name_length}}",
                )

                # Build context with all currently available entities
                context = {**self.parameters, **self.functions, **self.distributions}

                if node_type == "parameter":
                    # Parameters are already created with bounds applied, nothing to do
                    pass

                elif node_type == "constant":
                    # Constants are pre-created by distributions - add to parameters
                    self.parameters[node_name] = constants_map[node_name]

                elif node_type == "function":
                    # Functions are evaluated by design
                    self.functions[node_name] = functions[node_name].expression(context)

                elif node_type == "distribution":
                    # Distributions are evaluated by design
                    self.distributions[node_name] = distributions[node_name].expression(
                        context
                    )

                # Advance progress
                progress_bar.advance(task)

    def _get_compiled_function(
        self, name: str
    ) -> Callable[..., npt.NDArray[np.float64]]:
        """
        Get or create a compiled PyTensor function for the specified distribution.

        Args:
            name (str): Name of the distribution.

        Returns:
            Callable: Compiled PyTensor function.
        """
        if name not in self._compiled_functions:
            dist = self.distributions[name]
            inputs = [var for var in graph_inputs([dist]) if var.name is not None]

            # Use the specified PyTensor mode
            compilation_mode = self.mode

            self._compiled_functions[name] = cast(
                Callable[..., npt.NDArray[np.float64]],
                function(
                    inputs=inputs,
                    outputs=dist,
                    mode=compilation_mode,
                    on_unused_input="ignore",
                ),  # type: ignore[no-untyped-call]
            )
        return self._compiled_functions[name]

    def pdf(self, name: str, **parametervalues: float) -> npt.NDArray[np.float64]:
        """
        Evaluates the probability density function of the specified distribution.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (float): Values for each distribution parameter.

        Returns:
            npt.NDArray[np.float64]: The evaluated PDF value.
        """
        if self.mode != "FAST_COMPILE":
            # Use compiled function for better performance
            func = self._get_compiled_function(name)
            inputs = [
                var
                for var in graph_inputs([self.distributions[name]])
                if var.name is not None
            ]
            positional_values = []
            for var in inputs:
                assert var.name is not None
                positional_values.append(parametervalues[var.name])
            return func(*positional_values)
        # Use original uncompiled approach
        dist = self.distributions[name]
        inputs = [var for var in graph_inputs([dist]) if var.name is not None]
        keyword_values: dict[str, float] = {}
        for var in inputs:
            assert var.name is not None
            keyword_values[var.name] = parametervalues[var.name]

        func = cast(
            Callable[..., npt.NDArray[np.float64]],
            function(inputs=inputs, outputs=dist),  # type: ignore[no-untyped-call]
        )
        return func(**keyword_values)

    def logpdf(self, name: str, **parametervalues: float) -> npt.NDArray[np.float64]:
        """
        Evaluates the natural logarithm of the PDF.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (float): Values for each distribution parameter.

        Returns:
            npt.NDArray[np.float64]: The log of the PDF.
        """
        return np.log(self.pdf(name, **parametervalues))

    def visualize_graph(
        self,
        name: str,
        fmt: str = "svg",
        outfile: str | None = None,
        path: str | None = None,
    ) -> str:
        """
        Visualize the computation graph for a distribution.

        Args:
            name (str): Distribution name.
            fmt (str): Output format ('svg', 'png', 'pdf'). Defaults to 'svg'.
            outfile (str | None): Output filename. If None, uses '{name}_graph.{fmt}'.
            path (str | None): Directory path for output. If None, uses current working directory.

        Returns:
            str: Path to the generated visualization file.

        Raises:
            ImportError: If pydot is not installed.
        """
        try:
            from pytensor.printing import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
                pydotprint,
            )
        except ImportError as e:
            msg = "Graph visualization requires pydot. Install with: pip install pydot"
            raise ImportError(msg) from e

        if name not in self.distributions:
            msg = f"Distribution '{name}' not found in model"
            raise ValueError(msg)

        dist = self.distributions[name]

        if outfile is not None:
            filename = outfile
        else:
            base_filename = f"{name}_graph.{fmt}"
            if path is not None:
                filename = str(Path(path) / base_filename)
            else:
                filename = base_filename

        pydotprint(  # type: ignore[no-untyped-call]
            dist, outfile=filename, format=fmt, with_ids=True, high_contrast=True
        )
        return filename

    def __repr__(self) -> str:
        """Provide a concise overview of the model structure."""
        param_names = list(self.parameters.keys())
        dist_names = list(self.distributions.keys())
        func_names = list(self.functions.keys())

        param_display = ", ".join(param_names[:5]) + (
            "..." if len(param_names) > 5 else ""
        )
        dist_display = ", ".join(dist_names[:3]) + (
            "..." if len(dist_names) > 3 else ""
        )
        func_display = ", ".join(func_names[:3]) + (
            "..." if len(func_names) > 3 else ""
        )

        mode_status = self.mode

        return f"""Model(
    mode: {mode_status}
    parameters: {len(param_names)} ({param_display})
    distributions: {len(dist_names)} ({dist_display})
    functions: {len(func_names)} ({func_display})
)"""

    def graph_summary(self, name: str) -> str:
        """
        Get a summary of the computation graph structure.

        Args:
            name (str): Distribution name.

        Returns:
            str: Summary of the graph structure.
        """
        if name not in self.distributions:
            msg = f"Distribution '{name}' not found in model"
            raise ValueError(msg)

        dist = self.distributions[name]
        inputs = list(graph_inputs([dist]))

        # Count different types of operations
        applies = list(applys_between(inputs, [dist]))

        op_types: dict[str, int] = {}
        for apply in applies:
            op_name = type(apply.op).__name__
            op_types[op_name] = op_types.get(op_name, 0) + 1

        compile_info = f"\n    Mode: {self.mode}\n    Compiled: {'Yes' if self.mode != 'FAST_COMPILE' and name in self._compiled_functions else 'No'}"

        return f"""Distribution '{name}':
    Input variables: {len(inputs)}
    Graph operations: {len(applies)}
    Operation types: {dict(sorted(op_types.items()))}{compile_info}
"""


class ParameterCollection:
    """
    Collection of named parameter sets for model configuration.

    Manages multiple parameter sets, each containing a collection of
    parameter points with specific names and values. Provides dict-like
    access to parameter sets by name or index.

    Attributes:
        sets (dict[str, ParameterSet]): Mapping from parameter set names to ParameterSet objects.
    """

    def __init__(self, parametersets: list[T.ParameterPoint]):
        """
        A collection of named parameter sets.

        Args:
            parametersets (list): List of parameterset configurations.

        Attributes:
            sets (OrderedDict): Mapping from parameter set names to ParameterSet objects.
        """
        self.sets: dict[str, ParameterSet] = OrderedDict()

        for parameterset_config in parametersets:
            parameterset = ParameterSet(
                parameterset_config["name"], parameterset_config["parameters"]
            )
            self.sets[parameterset.name] = parameterset

    def __getitem__(self, item: str | int) -> ParameterSet:
        key = list(self.sets.keys())[item] if isinstance(item, int) else item
        return self.sets[key]

    def get(
        self, item: str, default: TDefault | None = None
    ) -> ParameterSet | TDefault | None:
        """Get a parameter set by name, returning default if not found."""
        return self.sets.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.sets

    def __iter__(self) -> Iterator[ParameterSet]:
        return iter(self.sets.values())

    def __len__(self) -> int:
        return len(self.sets)


class ParameterSet:
    """
    Named collection of parameter points with specific values.

    Represents a single configuration of parameter values that can be
    used to evaluate a model. Each parameter set contains multiple
    parameter points, each with a name and numeric value.

    Attributes:
        name (str): Name of the parameter set.
        points (dict[str, ParameterPoint]): Mapping of parameter names to ParameterPoint objects.
    """

    def __init__(self, name: str, points: list[T.Parameter]):
        """
        Represents a single named set of parameter values.

        Args:
            name (str): Name of the parameter set.
            points (list): List of parameter point configurations.

        Attributes:
            name (str): Name of the parameter set.
            points (dict[str, ParameterPoint]): Mapping of parameter names to ParameterPoint objects.
        """
        self.name = name

        self.points: dict[str, ParameterPoint] = OrderedDict()

        for points_config in points:
            point = ParameterPoint(points_config["name"], points_config["value"])
            self.points[point.name] = point

    def __getitem__(self, item: str | int) -> ParameterPoint:
        key = list(self.points.keys())[item] if isinstance(item, int) else item
        return self.points[key]

    def get(
        self, item: str, default: TDefault | None = None
    ) -> ParameterPoint | TDefault | None:
        """Get a parameter point by name, returning default if not found."""
        return self.points.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.points

    def __iter__(self) -> Iterator[ParameterPoint]:
        return iter(self.points.values())

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class ParameterPoint:
    """
    Represents a single parameter point.

    Attributes:
        name (str): Name of the parameter.
        value (float): Value of the parameter.
    """

    name: str
    value: float


class DomainCollection:
    """
    Collection of domain constraints for model parameters.

    Manages domain sets that define valid ranges for model parameters.
    Each domain set specifies minimum and maximum bounds for parameters,
    which are used to create bounded tensor variables.

    Attributes:
        domains (dict[str, DomainSet]): Mapping from domain names to DomainSet objects.
    """

    def __init__(self, domainsets: list[T.Domain]):
        """
        Collection of named domain sets.

        Args:
            domainsets (list): List of domain set configurations.

        Attributes:
            domains (OrderedDict): Mapping of domain names to DomainSet objects.
        """

        self.domains: dict[str, DomainSet] = OrderedDict()

        for domain_config in domainsets:
            domain = DomainSet(
                domain_config["axes"], domain_config["name"], domain_config["type"]
            )
            self.domains[domain.name] = domain

    def __getitem__(self, item: str | int) -> DomainSet:
        key = list(self.domains.keys())[item] if isinstance(item, int) else item
        return self.domains[key]

    def get(
        self, item: str, default: TDefault | None = None
    ) -> DomainSet | TDefault | None:
        """Get a domain set by name, returning default if not found."""
        return self.domains.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.domains

    def __iter__(self) -> Iterator[DomainSet]:
        return iter(self.domains.values())

    def __len__(self) -> int:
        return len(self.domains)


@dataclass
class DomainPoint:
    """
    Represents a valid domain (axis) for a single parameter.

    Attributes:
        name (str): Name of the parameter.
        min (float): Minimum value.
        max (float): Maximum value.
        range (tuple): Computed range as (min, max), not included in serialization.
    """

    name: str
    min: float
    max: float
    range: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.range = (self.min, self.max)

    def to_dict(self) -> T.Axis:
        """
        to dictionary
        """
        return {"name": self.name, "min": self.min, "max": self.max}


class DomainSet:
    """
    Set of parameter domain constraints with bounds.

    Defines valid ranges for multiple parameters, specifying minimum
    and maximum bounds for each. Used to create bounded tensor variables
    that are automatically clipped to their valid ranges.

    Attributes:
        name (str): Name of the domain set.
        kind (str): Type of the domain set.
        domains (dict[str, Axis]): Mapping of parameter names to (min, max) tuples.
    """

    def __init__(self, axes: list[T.Axis], name: str, kind: str):
        """
        Represents a set of valid domains for parameters.

        Args:
            axes (list): List of domain configurations.
            name (str): Name of the domain set.
            kind (str): Type of the domain.

        Attributes:
            domains (OrderedDict): Mapping of parameter names to allowed ranges.
        """
        self.name = name
        self.kind = kind
        self.domains: dict[str, Axis] = OrderedDict()

        for axis_config in axes:
            domain = DomainPoint(
                axis_config["name"], axis_config["min"], axis_config["max"]
            )
            self.domains[domain.name] = domain.range

    def __getitem__(self, item: int | str) -> Axis:
        key = list(self.domains.keys())[item] if isinstance(item, int) else item
        return self.domains[key]

    def get(
        self, item: str, default: TDefault | Axis = (None, None)
    ) -> Axis | TDefault:
        """Get domain bounds for a parameter, returning default if not found."""
        return self.domains.get(item, default)

    def __contains__(self, item: str) -> bool:
        return item in self.domains

    def __iter__(self) -> Iterator[Axis]:
        return iter(self.domains.values())

    def __len__(self) -> int:
        return len(self.domains)


def boundedscalar(name: str, domain: Axis) -> T.TensorVar:
    """
    Creates a scalar tensor variable with optional domain constraints.

    Args:
        name: Name of the scalar parameter.
        domain (tuple): Tuple specifying (min, max) range. Use None for unbounded sides.
                       For example: (0.0, None) for lower bound only, (None, 1.0) for upper bound only.
                       If both bounds are None, returns an unbounded scalar.

    Returns:
        pytensor.tensor.variable.TensorVariable: The scalar tensor, clipped to domain if bounds exist.

    Examples:
        >>> boundedscalar("sigma", (0.0, None))  # sigma >= 0
        >>> boundedscalar("fraction", (0.0, 1.0))  # 0 <= fraction <= 1
        >>> boundedscalar("temperature", (None, 100.0))  # temperature <= 100
        >>> boundedscalar("unbounded", (None, None))  # no bounds applied
    """
    min_bound, max_bound = domain

    # Create the base scalar tensor
    tensor = pt.scalar(name)

    # If both bounds are None, return unbounded scalar
    if min_bound is None and max_bound is None:
        return cast(T.TensorVar, tensor)

    # Use infinity constants for unbounded sides
    min_val = pt.constant(-np.inf) if min_bound is None else pt.constant(min_bound)
    max_val = pt.constant(np.inf) if max_bound is None else pt.constant(max_bound)

    clipped = pt.clip(tensor, min_val, max_val)
    clipped.name = tensor.name  # Preserve the original name
    return cast(T.TensorVar, clipped)
