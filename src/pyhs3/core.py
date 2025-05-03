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
    Manages the overall structure of the model including parameters, domains, and distributions.

    Args:
        spec (dict): A dictionary containing model definitions including parameter points, distributions,
            and domains.

    Attributes:
        parameter_collection (ParameterCollection): Set of named parameter points.
        distribution_set (DistributionSet): All distributions used in the workspace.
        domain_collection (DomainCollection): Domain definitions for all parameters.
    """

    def __init__(self, spec: T.HS3Spec):
        self.parameter_collection = ParameterCollection(spec["parameter_points"])
        self.distribution_set = DistributionSet(spec["distributions"])
        self.domain_collection = DomainCollection(spec["domains"])

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

    def __init__(
        self,
        *,
        parameterset: ParameterSet,
        distributions: DistributionSet,
        domains: DomainSet,
    ):
        self.parameters = {}
        self.parameterset = parameterset

        for parameter_point in parameterset:
            self.parameters[parameter_point.name] = boundedscalar(
                parameter_point.name, domains[parameter_point.name]
            )

        self.distributions: dict[str, T.TensorVar] = {}
        G: nx.DiGraph[str] = nx.DiGraph()
        for dist in distributions:
            G.add_node(dist.name, type="distribution")
            for parameter in dist.parameters:
                if not (
                    parameter in G and G.nodes[parameter]["type"] == "distribution"
                ):
                    G.add_node(parameter, type="parameter")

                G.add_edge(parameter, dist.name)

        for node in nx.topological_sort(G):
            if G.nodes[node]["type"] != "distribution":
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
    A collection of named parameter sets.

    Args:
        parametersets (list): List of parameterset configurations.

    Attributes:
        sets (OrderedDict): Mapping from parameter set names to ParameterSet objects.
    """

    def __init__(self, parametersets: list[T.ParameterPoint]):
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
    Represents a single named set of parameter values.

    Args:
        name (str): Name of the parameter set.
        points (list): List of parameter point configurations.

    Attributes:
        name (str): Name of the parameter set.
        points (dict[str, ParameterPoint]): Mapping of parameter names to ParameterPoint objects.
    """

    def __init__(self, name: str, points: list[T.Parameter]):
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
    Collection of named domain sets.

    Args:
        domainsets (list): List of domain set configurations.

    Attributes:
        domains (OrderedDict): Mapping of domain names to DomainSet objects.
    """

    def __init__(self, domainsets: list[T.Domain]):
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
        return {"name": self.name, "min": self.min, "max": self.max}


class DomainSet:
    """
    Represents a set of valid domains for parameters.

    Args:
        axes (list): List of domain configurations.
        name (str): Name of the domain set.
        type (str): Type of the domain.

    Attributes:
        domains (OrderedDict): Mapping of parameter names to allowed ranges.
    """

    def __init__(self, axes: list[T.Axis], name: str, type: str):
        self.name = name
        self.type = type
        self.domains: dict[str, tuple[float, float]] = OrderedDict()

        for domain_config in axes:
            domain = DomainPoint(
                domain_config["name"], domain_config["min"], domain_config["max"]
            )
            self.domains[domain.name] = domain.range

    def __getitem__(self, item: int | str) -> tuple[float, float]:
        key = list(self.domains.keys())[item] if isinstance(item, int) else item
        return self.domains[key]


DistType = TypeVar("DistType", bound="Distribution[T.Distribution]")
DistConfig = TypeVar("DistConfig", bound=T.Distribution)


class Distribution(Generic[DistConfig]):
    """
    Base class for distributions.

    Args:
        name (str): Name of the distribution.
        type (str): Type identifier.

    Attributes:
        name (str): Name of the distribution.
        type (str): Type identifier.
        parameters (list[str]): initially empty list to be filled with parameter names.
    """

    def __init__(
        self,
        *,
        name: str,
        type: str = "Distribution",
        parameters: list[str] | None = None,
        **kwargs: Any,
    ):
        self.name = name
        self.type = type
        self.parameters = parameters or []
        self.kwargs = kwargs

    def expression(
        self, distributionsandparameters: dict[str, T.TensorVar]
    ) -> T.TensorVar:
        msg = f"Distribution type={self.type} is not implemented."
        raise NotImplementedError(msg)

    @classmethod
    def from_dict(
        cls: type[Distribution[DistConfig]], config: DistConfig
    ) -> Distribution[DistConfig]:
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

    # need a way for the distribution to get the scalar function .parameter from parameterset
    def __init__(self, *, name: str, mean: str, sigma: str, x: str):
        super().__init__(name=name, type="gaussian_dist", parameters=[mean, sigma, x])
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

    def __init__(
        self, *, name: str, coefficients: list[str], extended: bool, summands: list[str]
    ):
        super().__init__(
            name=name, type="mixture_dist", parameters=[*coefficients, *summands]
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
    Collection of distributions.

    Args:
        distributions (list[dict[str, str]]): List of distribution configurations.

    Attributes:
        dists (dict): Mapping of distribution names to Distribution objects.
    """

    def __init__(self, distributions: list[T.Distribution]) -> None:
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
