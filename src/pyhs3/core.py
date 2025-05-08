from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.graph.basic import graph_inputs

from pyhs3 import typing as T
from pyhs3.typing import distribution as TD

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

        assert set(parameterset.points.keys()) == set(domainset.domains.keys()), (
            "parameter and domain names do not match"
        )

        return Model(
            parameterset=parameterset,
            distributions=self.distribution_set,
            domains=domainset,
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
    ):
        """
        Represents a probabilistic model composed of parameters, domains, and distributions.

        Args:
            parameterset (ParameterSet): The parameter set used in the model.
            distributions (DistributionSet): Set of distributions to include.
            domains (DomainSet): Domain constraints for parameters.

        Attributes:
            parameters (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic parameter variables.
            parameterset (ParameterSet): The original set of parameter values.
            distributions (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic distribution expressions.
        """
        self.parameters = {}
        self.parameterset = parameterset

        for parameter_point in parameterset:
            self.parameters[parameter_point.name] = boundedscalar(
                parameter_point.name, domains[parameter_point.name]
            )

        self.distributions: dict[str, T.TensorVar] = {}
        graph: nx.DiGraph[str] = nx.DiGraph()
        for dist in distributions:
            graph.add_node(dist.name, type="distribution")
            for parameter in dist.parameters:
                if not (
                    parameter in graph
                    and graph.nodes[parameter]["type"] == "distribution"
                ):
                    graph.add_node(parameter, type="parameter")

                graph.add_edge(parameter, dist.name)

        for node in nx.topological_sort(graph):
            if graph.nodes[node]["type"] != "distribution":
                continue
            self.distributions[node] = distributions[node].expression(
                {**self.parameters, **self.distributions}
            )

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

    def __iter__(self) -> Iterator[ParameterPoint]:
        return iter(self.points.values())


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

    def __init__(self, axes: list[T.Axis], name: str, dtype: str):
        """
        Represents a set of valid domains for parameters.

        Args:
            axes (list): List of domain configurations.
            name (str): Name of the domain set.
            dtype (str): Type of the domain.

        Attributes:
            domains (OrderedDict): Mapping of parameter names to allowed ranges.
        """
        self.name = name
        self.type = dtype
        self.domains: dict[str, tuple[float, float]] = OrderedDict()

        for domain_config in axes:
            domain = DomainPoint(
                domain_config["name"], domain_config["min"], domain_config["max"]
            )
            self.domains[domain.name] = domain.range

    def __getitem__(self, item: int | str) -> tuple[float, float]:
        key = list(self.domains.keys())[item] if isinstance(item, int) else item
        return self.domains[key]


DistT = TypeVar("DistT", bound="Distribution[T.Distribution]")
DistConfigT = TypeVar("DistConfigT", bound=T.Distribution)


class Distribution(Generic[DistConfigT]):
    """
    Distribution
    """

    def __init__(
        self,
        *,
        name: str,
        dtype: str = "Distribution",
        parameters: list[str] | None = None,
        **kwargs: Any,
    ):
        """
        Base class for distributions.

        Args:
            name (str): Name of the distribution.
            dtype (str): Type identifier.

        Attributes:
            name (str): Name of the distribution.
            dtype (str): Type identifier.
            parameters (list[str]): initially empty list to be filled with parameter names.
        """
        self.name = name
        self.type = dtype
        self.parameters = parameters or []
        self.kwargs = kwargs

    def expression(self, _: dict[str, T.TensorVar]) -> T.TensorVar:
        """
        Unimplemented
        """
        msg = f"Distribution type={self.type} is not implemented."
        raise NotImplementedError(msg)

    @classmethod
    def from_dict(
        cls: type[Distribution[DistConfigT]], config: DistConfigT
    ) -> Distribution[DistConfigT]:
        """
        Factory method to create a distribution instance from a dictionary.

        Args:
            config (dict): Dictionary containing configuration for the distribution.

        Returns:
            Distribution: A new instance of the appropriate distribution subclass.
        """
        raise NotImplementedError


class GaussianDist(Distribution[TD.GaussianDistribution]):
    """
    GaussianDist
    """

    # need a way for the distribution to get the scalar function .parameter from parameterset
    def __init__(self, *, name: str, mean: str, sigma: str, x: str):
        """
        Subclass of Distribution representing a Gaussian distribution.

        Args:
            name (str): Name of the distribution.
            mean (str): Parameter name for the mean.
            sigma (str): Parameter name for the standard deviation.
            x (str): Input variable name.

        Attributes:
            name (str): Name of the distribution.
            mean (str): Parameter name for the mean.
            sigma (str): Parameter name for the standard deviation.
            x (str): Input variable name.
            parameters (list[str]): list containing mean, sigma, and x.
        """
        super().__init__(name=name, dtype="gaussian_dist", parameters=[mean, sigma, x])
        self.mean = mean
        self.sigma = sigma
        self.x = x

    @classmethod
    def from_dict(cls, config: TD.GaussianDistribution) -> GaussianDist:
        """
        Creates an instance of GaussianDist from a dictionary configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            GaussianDist: The created GaussianDist instance.
        """
        return cls(
            name=config["name"],
            mean=config["mean"],
            sigma=config["sigma"],
            x=config["x"],
        )

    def expression(
        self, distributionsandparameters: dict[str, T.TensorVar]
    ) -> T.TensorVar:
        """
        Builds a symbolic expression for the Gaussian PDF.

        Args:
            distributionsandparameters (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Gaussian PDF.
        """
        # log.info("parameters: ", parameters)
        norm_const = 1.0 / (
            pt.sqrt(2 * math.pi) * distributionsandparameters[self.sigma]
        )
        exponent = pt.exp(
            -0.5
            * (
                (
                    distributionsandparameters[self.x]
                    - distributionsandparameters[self.mean]
                )
                / distributionsandparameters[self.sigma]
            )
            ** 2
        )
        return cast(T.TensorVar, norm_const * exponent)


class MixtureDist(Distribution[TD.MixtureDistribution]):
    """
    MixtureDist
    """

    def __init__(
        self, *, name: str, coefficients: list[str], extended: bool, summands: list[str]
    ):
        """
        Subclass of Distribution representing a mixture of distributions

        Args:
            name (str): Name of the distribution.
            coefficients (list): Coefficient parameter names.
            extended (bool): Whether the distribution is extended.
            summands (list): List of component distribution names.

        Attributes:
            name (str): Name of the distribution.
            coefficients (list[str]): Coefficient parameter names.
            extended (bool): Whether the distribution is extended.
            summands (list[str]): List of component distribution names.
            parameters (list[str]): List of coefficients and summands
        """
        super().__init__(
            name=name, dtype="mixture_dist", parameters=[*coefficients, *summands]
        )
        self.name = name
        self.coefficients = coefficients
        self.extended = extended
        self.summands = summands

    @classmethod
    def from_dict(cls, config: TD.MixtureDistribution) -> MixtureDist:
        """
        Creates an instance of MixtureDist from a dictionary configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            MixtureDist: The created MixtureDist instance.
        """
        return cls(
            name=config["name"],
            coefficients=config["coefficients"],
            extended=config["extended"],
            summands=config["summands"],
        )

    def expression(
        self, distributionsandparameters: dict[str, T.TensorVar]
    ) -> T.TensorVar:
        """
        Builds a symbolic expression for the mixture distribution.

        Args:
            distributionsandparameters (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the mixture PDF.
        """

        mixturesum = pt.constant(0.0)
        coeffsum = pt.constant(0.0)
        i = 0
        for coeff in self.coefficients:
            coeffsum += distributionsandparameters[coeff]
            mixturesum += (
                distributionsandparameters[coeff]
                * distributionsandparameters[self.summands[i]]
            )
        mixturesum += (1 - coeffsum) * distributionsandparameters[self.summands[i]]
        return mixturesum


registered_distributions: dict[str, type[Distribution[Any]]] = {
    "gaussian_dist": GaussianDist,
    "mixture_dist": MixtureDist,
}


class DistributionSet:
    """
    DistributionSet
    """

    def __init__(self, distributions: list[T.Distribution]) -> None:
        """
        Collection of distributions.

        Args:
            distributions (list[dict[str, str]]): List of distribution configurations.

        Attributes:
            dists (dict): Mapping of distribution names to Distribution objects.
        """
        self.dists: dict[str, Distribution[Any]] = {}
        for dist_config in distributions:
            dist_type = dist_config["type"]
            the_dist = registered_distributions.get(dist_type, Distribution)
            dist = the_dist.from_dict(
                {k: v for k, v in dist_config.items() if k != "type"}
            )
            self.dists[dist.name] = dist

    def __getitem__(self, item: str) -> Distribution[Any]:
        return self.dists[item]

    def __iter__(self) -> Iterator[Distribution[Any]]:
        return iter(self.dists.values())


def boundedscalar(name: str, domain: tuple[float, float]) -> T.TensorVar:
    """
    Creates a pytensor scalar constrained within a given domain.

    Args:
        name (str): Name of the scalar.
        domain (tuple): Tuple specifying (min, max) range.

    Returns:
        pytensor.tensor.variable.TensorVariable: A pytensor scalar clipped to the domain range.
    """
    x = pt.scalar(name)

    i = domain[0]
    f = domain[1]

    return cast(T.TensorVar, pt.clip(x, i, f))
