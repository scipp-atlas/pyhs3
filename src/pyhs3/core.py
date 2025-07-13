from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
import rustworkx as rx
from pytensor.compile.function import function
from pytensor.graph.basic import graph_inputs

from pyhs3 import typing as T
from pyhs3.functions import FunctionSet
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
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
            # Use domain bounds if available, otherwise use unbounded (None, None)
            domain = domains.domains.get(parameter_point.name, (None, None))
            self.parameters[parameter_point.name] = boundedscalar(
                parameter_point.name, domain
            )

        self.distributions: dict[str, T.TensorVar] = {}

        # Build dependency graph including functions and distributions
        graph = rx.PyDiGraph()
        nodes: dict[str, int] = {}

        # Add functions to the graph (same pattern as distributions)
        for func in functions:
            if func.name not in nodes:
                idx = graph.add_node({"type": "function", "name": func.name})
                nodes[func.name] = idx

            for param in func.parameters:
                p_idx = nodes.get(param)
                if p_idx is None:
                    # Could be a parameter, function, or distribution
                    node_type = "parameter"  # Default assumption
                    if param in functions:
                        node_type = "function"
                    elif param in distributions:
                        node_type = "distribution"

                    p_idx = graph.add_node({"type": node_type, "name": param})
                    nodes[param] = p_idx
                graph.add_edge(p_idx, idx, None)

        # Add distributions to the graph (existing logic)
        for dist in distributions:
            if dist.name not in nodes:
                idx = graph.add_node({"type": "distribution", "name": dist.name})
                nodes[dist.name] = idx
            else:
                idx = nodes[dist.name]
                graph[idx] = {"type": "distribution", "name": dist.name}
            for param in dist.parameters:
                p_idx = nodes.get(param)
                if p_idx is None or graph[p_idx]["type"] == "distribution":
                    if p_idx is None:
                        # Could be a parameter or function
                        node_type = "parameter"  # Default assumption
                        if param in functions:
                            node_type = "function"

                        p_idx = graph.add_node({"type": node_type, "name": param})
                        nodes[param] = p_idx
                    else:
                        graph[p_idx] = {"type": "parameter", "name": param}
                graph.add_edge(p_idx, idx, None)

        # Evaluate functions and distributions in topological order
        for node_idx in rx.topological_sort(graph):
            node_data = graph[node_idx]
            if node_data["type"] == "function":
                func_name = node_data["name"]
                context = {**self.parameters, **self.functions, **self.distributions}
                self.functions[func_name] = functions[func_name].expression(context)
            elif node_data["type"] == "distribution":
                dist_name = node_data["name"]
                context = {**self.parameters, **self.functions, **self.distributions}
                self.distributions[dist_name] = distributions[dist_name].expression(
                    context
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
        kind: str = "Distribution",
        parameters: list[str] | None = None,
        **kwargs: Any,
    ):
        """
        Base class for distributions.

        Args:
            name (str): Name of the distribution.
            kind (str): Type identifier.

        Attributes:
            name (str): Name of the distribution.
            kind (str): Type identifier.
            parameters (list[str]): initially empty list to be filled with parameter names.
        """
        self.name = name
        self.kind = kind
        self.parameters = parameters or []
        self.kwargs = kwargs

    def expression(self, _: dict[str, T.TensorVar]) -> T.TensorVar:
        """
        Unimplemented
        """
        msg = f"Distribution type={self.kind} is not implemented."
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
        super().__init__(name=name, kind="gaussian_dist", parameters=[mean, sigma, x])
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
            name=name, kind="mixture_dist", parameters=[*coefficients, *summands]
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

        for i, coeff in enumerate(self.coefficients):
            coeffsum += distributionsandparameters[coeff]
            mixturesum += (
                distributionsandparameters[coeff]
                * distributionsandparameters[self.summands[i]]
            )

        last_index = len(self.summands) - 1
        f_last = distributionsandparameters[self.summands[last_index]]
        mixturesum = mixturesum + (1 - coeffsum) * f_last
        return cast(T.TensorVar, mixturesum)


class ProductDist(Distribution[TD.ProductDistribution]):
    """
    Product distribution implementation.

    Implements a product of PDFs as defined in ROOT's RooProdPdf.

    The probability density function is defined as:

    $$f(x, \\ldots) = \\prod_{i=1}^{N} \\text{PDF}_i(x, \\ldots)$$

    where each PDF_i is a component distribution that may share observables.

    Parameters:
        factors: List of component distribution names to multiply together

    Note:
        In the context of pytensor variables/tensors, this is implemented as
        an elementwise product of all factor distributions.
    """

    def __init__(self, *, name: str, factors: list[str]):
        """
        Initialize a ProductDist.

        Args:
            name: Name of the distribution
            factors: List of component distribution names to multiply together
        """
        super().__init__(name=name, kind="product_dist", parameters=factors)
        self.factors = factors

    @classmethod
    def from_dict(cls, config: TD.ProductDistribution) -> ProductDist:
        """
        Creates an instance of ProductDist from a dictionary configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ProductDist: The created ProductDist instance.
        """
        return cls(name=config["name"], factors=config["factors"])

    def expression(
        self, distributionsandparameters: dict[str, T.TensorVar]
    ) -> T.TensorVar:
        """
        Evaluate the product distribution.

        Args:
            distributionsandparameters: Mapping of names to pytensor variables

        Returns:
            Symbolic representation of the product PDF
        """
        pt_factors = pt.stack(
            [distributionsandparameters[factor] for factor in self.factors]
        )
        return cast(T.TensorVar, pt.prod(pt_factors, axis=0))  # type: ignore[no-untyped-call]


class CrystalDist(Distribution[TD.CrystalDistribution]):
    """
    Crystal Ball distribution implementation.

    Implements the generalized asymmetrical double-sided Crystal Ball line shape
    as defined in ROOT's RooCrystalBall.

    The probability density function is defined as:

    $$f(m; m_0, \\sigma_L, \\sigma_R, \\alpha_L, \\alpha_R, n_L, n_R) = \\begin{cases}
    A_L \\cdot \\left(B_L - \\frac{m - m_0}{\\sigma_L}\\right)^{-n_L}, & \\text{for } \\frac{m - m_0}{\\sigma_L} < -\\alpha_L \\\\
    \\exp\\left(-\\frac{1}{2} \\cdot \\left[\\frac{m - m_0}{\\sigma_L}\\right]^2\\right), & \\text{for } \\frac{m - m_0}{\\sigma_L} \\leq 0 \\\\
    \\exp\\left(-\\frac{1}{2} \\cdot \\left[\\frac{m - m_0}{\\sigma_R}\\right]^2\\right), & \\text{for } \\frac{m - m_0}{\\sigma_R} \\leq \\alpha_R \\\\
    A_R \\cdot \\left(B_R + \\frac{m - m_0}{\\sigma_R}\\right)^{-n_R}, & \\text{otherwise}
    \\end{cases}$$

    where:

    $$\\begin{align}
    A_i &= \\left(\\frac{n_i}{\\alpha_i}\\right)^{n_i} \\cdot \\exp\\left(-\\frac{\\alpha_i^2}{2}\\right) \\\\
    B_i &= \\frac{n_i}{\\alpha_i} - \\alpha_i
    \\end{align}$$

    Parameters:
        m: Observable variable
        m0: Peak position (mean)
        sigma_L: Left-side width parameter (must be > 0)
        sigma_R: Right-side width parameter (must be > 0)
        alpha_L: Left-side transition point (must be > 0)
        alpha_R: Right-side transition point (must be > 0)
        n_L: Left-side power law exponent (must be > 0)
        n_R: Right-side power law exponent (must be > 0)

    Note:
        All parameters except m and m0 must be positive. The distribution
        reduces to a single-sided Crystal Ball when one of the alpha parameters
        is set to zero.
    """

    def __init__(
        self,
        *,
        name: str,
        alpha_L: str,
        alpha_R: str,
        m: str,
        m0: str,
        n_R: str,
        n_L: str,
        sigma_L: str,
        sigma_R: str,
    ):
        """
        Initialize a CrystalDist.

        Args:
            name: Name of the distribution
            alpha_L: Left-side transition point parameter name
            alpha_R: Right-side transition point parameter name
            m: Observable variable name
            m0: Peak position parameter name
            n_L: Left-side power law exponent parameter name
            n_R: Right-side power law exponent parameter name
            sigma_L: Left-side width parameter name
            sigma_R: Right-side width parameter name
        """
        super().__init__(
            name=name,
            kind="crystal_dist",
            parameters=[alpha_L, alpha_R, m, m0, n_R, n_L, sigma_L, sigma_R],
        )
        self.alpha_L = alpha_L
        self.alpha_R = alpha_R
        self.m = m
        self.m0 = m0
        self.n_R = n_R
        self.n_L = n_L
        self.sigma_L = sigma_L
        self.sigma_R = sigma_R

    @classmethod
    def from_dict(cls, config: TD.CrystalDistribution) -> CrystalDist:
        """
        Create a CrystalDist from a dictionary configuration.

        Args:
            config: Configuration dictionary

        Returns:
            The created CrystalDist instance
        """
        return cls(
            name=config["name"],
            alpha_L=config["alpha_L"],
            alpha_R=config["alpha_R"],
            m=config["m"],
            m0=config["m0"],
            n_R=config["n_R"],
            n_L=config["n_L"],
            sigma_L=config["sigma_L"],
            sigma_R=config["sigma_R"],
        )

    def expression(
        self, distributionsandparameters: dict[str, T.TensorVar]
    ) -> T.TensorVar:
        """
        Evaluate the Crystal Ball distribution.

        Implements the ROOT RooCrystalBall formula with proper parameter validation.
        All shape parameters (alpha, n, sigma) are assumed to be positive.
        """
        alpha_L = distributionsandparameters[self.alpha_L]
        alpha_R = distributionsandparameters[self.alpha_R]
        m = distributionsandparameters[self.m]
        m0 = distributionsandparameters[self.m0]
        n_L = distributionsandparameters[self.n_L]
        n_R = distributionsandparameters[self.n_R]
        sigma_L = distributionsandparameters[self.sigma_L]
        sigma_R = distributionsandparameters[self.sigma_R]

        # Calculate A_i and B_i per ROOT formula
        # Note: alpha, n, sigma are assumed to be positive
        A_L = (n_L / alpha_L) ** n_L * pt.exp(-(alpha_L**2) / 2)
        A_R = (n_R / alpha_R) ** n_R * pt.exp(-(alpha_R**2) / 2)
        B_L = (n_L / alpha_L) - alpha_L
        B_R = (n_R / alpha_R) - alpha_R

        # Calculate normalized distance from peak for each side
        t_L = (m - m0) / sigma_L
        t_R = (m - m0) / sigma_R

        # Calculate each region per ROOT formula
        left_tail = A_L * ((B_L - t_L) ** (-n_L))
        core_left = pt.exp(-(t_L**2) / 2)
        core_right = pt.exp(-(t_R**2) / 2)
        right_tail = A_R * ((B_R + t_R) ** (-n_R))

        # Apply ROOT conditions
        return cast(
            T.TensorVar,
            pt.switch(
                t_L < -alpha_L,
                left_tail,
                pt.switch(
                    t_L <= 0,
                    core_left,
                    pt.switch(t_R <= alpha_R, core_right, right_tail),
                ),
            ),
        )


class GenericDist(Distribution[TD.GenericDistribution]):
    """
    Generic distribution implementation.

    Evaluates custom mathematical expressions using SymPy parsing and
    PyTensor computation graphs.

    Parameters:
        name: Name of the distribution
        expression: Mathematical expression string to be evaluated

    Supported Functions:
        - Basic arithmetic: +, -, *, /, **
        - Trigonometric: sin, cos, tan
        - Exponential/Logarithmic: exp, log
        - Other: sqrt, abs

    Examples:
        - "x**2 + 2*x + 1"
        - "exp(-x**2/2) * cos(y)"
        - "sin(x) + log(abs(y))"
    """

    def __init__(self, *, name: str, expression: str, **_kwargs: Any):
        """
        Initialize a GenericDist.

        Args:
            name: Name of the distribution
            expression: Mathematical expression string
            **_kwargs: Additional keyword arguments (ignored)
        """
        # Parse and analyze the expression during initialization
        self.expression_str = expression
        self.sympy_expr = parse_expression(expression)

        # Analyze the expression to determine dependencies
        analysis = analyze_sympy_expr(self.sympy_expr)
        independent_vars = [str(symbol) for symbol in analysis["independent_vars"]]
        self.dependent_vars = [str(symbol) for symbol in analysis["dependent_vars"]]

        # Initialize the parent with the independent variables as parameters
        super().__init__(name=name, kind="generic_dist", parameters=independent_vars)

    @classmethod
    def from_dict(cls, config: TD.GenericDistribution) -> GenericDist:
        """
        Create a GenericDist from a dictionary configuration.

        Args:
            config: Configuration dictionary

        Returns:
            The created GenericDist instance
        """
        return cls(name=config["name"], expression=config["expression"])

    def expression(
        self, distributionsandparameters: dict[str, T.TensorVar]
    ) -> T.TensorVar:
        """
        Evaluate the generic distribution using expression parsing.

        Args:
            distributionsandparameters: Mapping of names to pytensor variables

        Returns:
            PyTensor expression representing the parsed mathematical expression

        Raises:
            ValueError: If the expression cannot be parsed or contains undefined variables
        """
        # Get the required variables using the parameters determined during initialization
        variables = [distributionsandparameters[name] for name in self.parameters]

        # Convert using the pre-parsed sympy expression
        result = sympy_to_pytensor(self.sympy_expr, variables)

        return cast(T.TensorVar, result)


registered_distributions: dict[str, type[Distribution[Any]]] = {
    "gaussian_dist": GaussianDist,
    "mixture_dist": MixtureDist,
    "product_dist": ProductDist,
    "crystalball_doublesided_dist": CrystalDist,
    "generic_dist": GenericDist,
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
            if the_dist is Distribution:
                msg = f"Unknown distribution type: {dist_type}"
                raise ValueError(msg)
            dist = the_dist.from_dict(
                {k: v for k, v in dist_config.items() if k != "type"}
            )
            self.dists[dist.name] = dist

    def __getitem__(self, item: str) -> Distribution[Any]:
        return self.dists[item]

    def __contains__(self, item: str) -> bool:
        return item in self.dists

    def __iter__(self) -> Iterator[Distribution[Any]]:
        return iter(self.dists.values())

    def __len__(self) -> int:
        return len(self.dists)


def boundedscalar(name: str, domain: tuple[float | None, float | None]) -> T.TensorVar:
    """
    Creates a pytensor scalar constrained within a given domain.

    Args:
        name (str): Name of the scalar.
        domain (tuple): Tuple specifying (min, max) range. Use None for unbounded sides.
                       For example: (0.0, None) for lower bound only, (None, 1.0) for upper bound only.

    Returns:
        pytensor.tensor.variable.TensorVariable: A pytensor scalar clipped to the domain range.

    Examples:
        >>> boundedscalar("sigma", (0.0, None))  # sigma >= 0
        >>> boundedscalar("fraction", (0.0, 1.0))  # 0 <= fraction <= 1
        >>> boundedscalar("temperature", (None, 100.0))  # temperature <= 100
    """
    x = pt.scalar(name)

    i = domain[0]
    f = domain[1]

    # Use infinity constants for unbounded sides
    ninf = pt.constant(-np.inf)
    inf = pt.constant(np.inf)

    clipped = pt.clip(x, i or ninf, f or inf)
    clipped.name = name  # Preserve the original name
    return cast(T.TensorVar, clipped)
