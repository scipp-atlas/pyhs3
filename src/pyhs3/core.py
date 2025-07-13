from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Callable, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
import rustworkx as rx
from pytensor.compile.function import function
from pytensor.graph.basic import graph_inputs

from pyhs3 import typing as T
from pyhs3.distributions import DistributionSet
from pyhs3.functions import FunctionSet

log = logging.getLogger(__name__)


class Workspace:
    """
    Workspace
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
    ) -> Model:
        """
        Constructs a `Model` object using the provided domain and parameter point.

        Args:
            domain (int | str | DomainSet): Identifier or object specifying the domain to use.
            parameter_point (int | str | ParameterSet): Identifier or object specifying the parameter values to use.

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
        )


class Model:
    """
    Model
    """

    def __init__(
        self,
        *,
        parameterset: ParameterSet,
        distributions: DistributionSet,
        domains: DomainSet,
        functions: FunctionSet,
    ):
        """
        Represents a probabilistic model composed of parameters, domains, distributions, and functions.

        Args:
            parameterset (ParameterSet): The parameter set used in the model.
            distributions (DistributionSet): Set of distributions to include.
            domains (DomainSet): Domain constraints for parameters.
            functions (FunctionSet): Set of functions that compute parameter values.

        Attributes:
            parameters (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic parameter variables.
            parameterset (ParameterSet): The original set of parameter values.
            distributions (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic distribution expressions.
            functions (dict[str, pytensor.tensor.variable.TensorVariable]): Computed function values.
        """
        self.parameters = {}
        self.parameterset = parameterset
        self.functions: dict[str, T.TensorVar] = {}

        for parameter_point in parameterset:
            # Create unbounded scalar first, bounds will be applied later in dependency graph
            self.parameters[parameter_point.name] = pt.scalar(parameter_point.name)

        self.distributions: dict[str, T.TensorVar] = {}

        # Build dependency graph with proper entity identification
        self._build_dependency_graph(functions, distributions, domains)

    def _build_dependency_graph(
        self, functions: FunctionSet, distributions: DistributionSet, domains: DomainSet
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

        for node_idx in sorted_nodes:
            node_data = graph[node_idx]
            node_type = node_data["type"]
            node_name = node_data["name"]

            # Build context with all currently available entities
            context = {**self.parameters, **self.functions, **self.distributions}

            if node_type == "parameter":
                # Parameters are already created as pt.scalar, just apply bounds if needed
                tensor = self.parameters[node_name]
                domain = domains.domains.get(node_name, (None, None))
                if domain != (None, None):
                    self.parameters[node_name] = boundedscalar(tensor, domain)

            elif node_type == "constant":
                # Constants are pre-created by distributions - add to parameters
                self.parameters[node_name] = constants_map[node_name]

            elif node_type == "function":
                # Functions are evaluated by design
                result = functions[node_name].expression(context)
                # Apply domain bounds if they exist
                domain = domains.domains.get(node_name, (None, None))
                if domain != (None, None):
                    result = boundedscalar(result, domain)
                self.functions[node_name] = result

            elif node_type == "distribution":
                # Distributions are evaluated by design
                result = distributions[node_name].expression(context)
                # Apply domain bounds if they exist
                domain = domains.domains.get(node_name, (None, None))
                if domain != (None, None):
                    result = boundedscalar(result, domain)
                self.distributions[node_name] = result

    def pdf(self, name: str, **parametervalues: float) -> npt.NDArray[np.float64]:
        """
        Evaluates the probability density function of the specified distribution.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (dict[str: float]): Values for each distribution parameter.

        Returns:
            float: The evaluated PDF value.
        """
        dist = self.distributions[name]

        inputs = [var for var in graph_inputs([dist]) if var.name is not None]
        values: dict[str, float] = {}
        for var in inputs:
            assert var.name is not None
            values[var.name] = parametervalues[var.name]

        func = cast(
            Callable[..., npt.NDArray[np.float64]],
            function(inputs=inputs, outputs=dist),  # type: ignore[no-untyped-call]
        )
        return func(**values)

    def logpdf(self, name: str, **parametervalues: float) -> npt.NDArray[np.float64]:
        """
        Evaluates the natural logarithm of the PDF.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (dict[str: float]): Values for each distribution parameter.

        Returns:
            float: The log of the PDF.
        """
        return np.log(self.pdf(name, **parametervalues))


class ParameterCollection:
    """
    ParameterCollection
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

    def __contains__(self, item: str) -> bool:
        return item in self.sets

    def __iter__(self) -> Iterator[ParameterSet]:
        return iter(self.sets.values())

    def __len__(self) -> int:
        return len(self.sets)


class ParameterSet:
    """
    ParameterSet
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
    DomainCollection
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
    DomainSet
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
        self.domains: dict[str, tuple[float, float]] = OrderedDict()

        for domain_config in axes:
            domain = DomainPoint(
                domain_config["name"], domain_config["min"], domain_config["max"]
            )
            self.domains[domain.name] = domain.range

    def __getitem__(self, item: int | str) -> tuple[float, float]:
        key = list(self.domains.keys())[item] if isinstance(item, int) else item
        return self.domains[key]

    def __contains__(self, item: str) -> bool:
        return item in self.domains


def boundedscalar(
    tensor: T.TensorVar, domain: tuple[float | None, float | None]
) -> T.TensorVar:
    """
    Applies domain constraints to a pytensor tensor variable.

    Args:
        tensor: The tensor variable to constrain.
        domain (tuple): Tuple specifying (min, max) range. Use None for unbounded sides.
                       For example: (0.0, None) for lower bound only, (None, 1.0) for upper bound only.

    Returns:
        pytensor.tensor.variable.TensorVariable: The tensor clipped to the domain range.

    Examples:
        >>> boundedscalar(pt.scalar("sigma"), (0.0, None))  # sigma >= 0
        >>> boundedscalar(pt.scalar("fraction"), (0.0, 1.0))  # 0 <= fraction <= 1
        >>> boundedscalar(pt.scalar("temperature"), (None, 100.0))  # temperature <= 100
    """
    min_bound, max_bound = domain

    # Use infinity constants for unbounded sides
    min_val = pt.constant(-np.inf) if min_bound is None else pt.constant(min_bound)
    max_val = pt.constant(np.inf) if max_bound is None else pt.constant(max_bound)

    clipped = pt.clip(tensor, min_val, max_val)
    clipped.name = tensor.name  # Preserve the original name
    return cast(T.TensorVar, clipped)
