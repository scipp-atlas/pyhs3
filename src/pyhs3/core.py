from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, cast

import networkx as nx
import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import graph_inputs

from pyhs3 import typing as T

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

        assert set(parameter_point.points.keys()) == set(domain.domains.keys()), (
            "parameter and domain names do not match"
        )

        return Model(
            parameterset=parameter_point,
            distributions=self.distribution_set,
            domains=domain,
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

        for parameter in parameterset:
            self.parameters[parameter.name] = boundedscalar(
                parameter.name, domains[parameter.name]
            )

        self.distributions = {}
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

    def pdf(self, name: str, **parametervalues: float) -> float:
        """
        Evaluates the probability density function of the specified distribution.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (dict[str: float]): Values for each distribution parameter.

        Returns:
            float: The evaluated PDF value.
        """
        log.info(parametervalues)

        dist = self.distributions[name]
        return dist.eval(
            {
                k: v
                for k, v in parametervalues.items()
                if k
                in [var.name for var in graph_inputs([dist]) if var.name is not None]
            }
        )

    def logpdf(self, name: str, **parametervalues: float) -> np.ndarray:
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

    def __getitem__(self, name: str) -> ParameterSet:
        if isinstance(name, str):
            return self.sets[name]
        return self.sets[self.sets.keys()[name]]


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

        self.points: dict[str, T.ParameterPoint] = OrderedDict()

        for points_config in points:
            point = T.ParameterPoint(points_config["name"], points_config["value"])
            self.points[point.name] = point

    def __getitem__(self, name: str) -> T.ParameterPoint:
        if isinstance(name, str):
            return self.points[name]
        return self.points[list(self.points.keys())[name]]

    def __iter__(self):
        return iter(self.points.values())


class DomainCollection:
    """
    Collection of named domain sets.

    Args:
        domainsets (list): List of domain set configurations.

    Attributes:
        domains (OrderedDict): Mapping of domain names to DomainSet objects.
    """

    def __init__(self, domainsets: list[DomainSet]):
        self.domains: dict[str, DomainSet] = OrderedDict()

        for domain_config in domainsets:
            domain = DomainSet(
                domain_config["axes"], domain_config["name"], domain_config["type"]
            )
            self.domains[domain.name] = domain

    def __getitem__(self, item: str | int) -> DomainSet:
        if isinstance(item, int):
            return self.domains.keys()[item]
        return self.domains[item]


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

    def __init__(self, axes: list[DomainPoint], name: str, type: str):
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


class DomainPoint:
    """
    Represents a valid domain for a single parameter.

    Args:
        name (str): Name of the parameter.
        min (float): Minimum value.
        max (float): Maximum value.

    Attributes:
        range (tuple): Tuple containing (min, max).
        name (str): Name of the parameter.
        min (float): Minimum value.
        max (float): Maximum value.
    """

    def __init__(self, name: str, min: float, max: float):
        self.name = name
        self.min = min
        self.max = max
        self.range = (self.min, self.max)


class Distribution:
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

    def __init__(self, name: str, type: str = "Distribution", **kwargs: Any):
        self.name = name
        self.type = type
        self.parameters: list[str] = []
        self.kwargs = kwargs

    def expression(
        self, distributionsandparameters: dict[str, T.TensorVar]
    ) -> T.TensorVar:
        msg = f"Distribution type={self.type} is not implemented."
        raise NotImplementedError(msg)


class GaussianDist(Distribution):
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
        super().__init__(name, "gaussian_dist")
        self.mean = mean
        self.sigma = sigma
        self.x = x
        self.parameters = [mean, sigma, x]

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
        return norm_const * exponent


class MixtureDist(Distribution):
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
        super().__init__(name, "mixture_dist")
        self.name = name
        self.coefficients = coefficients
        self.extended = extended
        self.summands = summands
        self.parameters = [*coefficients, *summands]

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
        mixturesum = 0.0
        coeffsum = 0.0
        i = 0
        for coeff in self.coefficients:
            coeffsum += distributionsandparameters[coeff]
            mixturesum += (
                distributionsandparameters[coeff]
                * distributionsandparameters[self.summands[i]]
            )
        mixturesum += (1 - coeffsum) * distributionsandparameters[self.summands[i]]
        return mixturesum


registered_distributions: dict[str, type[Distribution]] = {
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

    def __init__(self, distributions: list[dict[str, str]]) -> None:
        self.dists: dict[str, Distribution] = {}
        for dist_config in distributions:
            dist_type = dist_config.pop("type")
            the_dist = registered_distributions.get(dist_type, Distribution)
            dist = the_dist(**dist_config)
            self.dists[dist.name] = dist

    def __getitem__(self, item: str) -> Distribution:
        return self.dists[item]

    def __iter__(self) -> Iterator[Distribution]:
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
