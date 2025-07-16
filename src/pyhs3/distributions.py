"""
HS3 Distribution implementations.

Provides classes for handling various probability distributions including
Gaussian, Mixture, Product, Crystal Ball, and Generic distributions.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, cast

import pytensor.tensor as pt

from pyhs3 import typing as T
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
from pyhs3.typing import distribution as TD

log = logging.getLogger(__name__)


DistT = TypeVar("DistT", bound="Distribution[T.Distribution]")
DistConfigT = TypeVar("DistConfigT", bound=T.Distribution)


def process_parameter(
    config: T.Distribution, param_key: str
) -> tuple[str, T.TensorVar | None]:
    """
    Process a parameter that can be either a string reference or a numeric value.

    For numeric values, creates a pt.constant and generates a unique name.
    For string values, returns the value as-is with None for the constant.

    Args:
        config: The distribution configuration
        param_key: The parameter key to process (e.g., "mean", "sigma")

    Returns:
        Tuple of (processed_name, constant_tensor_or_none)
    """
    param_value = config[param_key]  # type: ignore[literal-required]
    if isinstance(param_value, (float, int)):
        # Generate unique constant name
        constant_name = f"constant_{config['name']}_{param_key}"
        # Create the constant tensor
        constant_tensor = cast(T.TensorVar, pt.constant(param_value))
        return constant_name, constant_tensor
    # It's a string reference - return as-is with no constant
    return param_value, None


class Distribution(Generic[DistConfigT]):
    """
    Base class for probability distributions in HS3.

    Provides the foundation for all distribution implementations,
    handling parameter management, constant generation, and symbolic
    expression evaluation using PyTensor.

    Attributes:
        name (str): Name of the distribution.
        kind (str): Type identifier for the distribution.
        parameters (list[str]): List of parameter names this distribution depends on.
        constants (dict[str, T.TensorVar]): Generated constants for numeric parameter values.
    """

    def __init__(
        self,
        *,
        name: str,
        kind: str = "Distribution",
        parameters: list[str] | None = None,
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
            constants (dict[str, pt.TensorVar]): Generated constants for numeric parameter values.
        """
        self.name = name
        self.kind = kind
        self.parameters = parameters or []
        self.constants: dict[str, T.TensorVar] = {}

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
    r"""
    Gaussian (normal) probability distribution.

    Implements the standard Gaussian probability density function:

    .. math::

        f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

    Parameters:
        mean (str): Parameter name for the mean (μ).
        sigma (str): Parameter name for the standard deviation (sigma).
        x (str): Input variable name.
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
        # Process parameters first to get correct string names and constants
        mean_name, mean_constant = process_parameter(config, "mean")
        sigma_name, sigma_constant = process_parameter(config, "sigma")
        x_name, x_constant = process_parameter(config, "x")

        # Create instance with processed string names
        instance = cls(
            name=config["name"],
            mean=mean_name,
            sigma=sigma_name,
            x=x_name,
        )

        # Add any generated constants to the instance
        if mean_constant is not None:
            instance.constants[mean_name] = mean_constant
        if sigma_constant is not None:
            instance.constants[sigma_name] = sigma_constant
        if x_constant is not None:
            instance.constants[x_name] = x_constant

        return instance

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
    r"""
    Mixture of probability distributions.

    Implements a weighted combination of multiple distributions:

    .. math::

        f(x) = \sum_{i=1}^{n-1} c_i \cdot f_i(x) + (1 - \sum_{i=1}^{n-1} c_i) \cdot f_n(x)

    The last component is automatically normalized to ensure the
    coefficients sum to 1.

    Parameters:
        coefficients (list[str]): Names of coefficient parameters.
        summands (list[str]): Names of component distributions.
        extended (bool): Whether the mixture is extended (affects normalization).
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
    r"""
    Product distribution implementation.

    Implements a product of PDFs as defined in ROOT's RooProdPdf.

    The probability density function is defined as:

    .. math::

        f(x, \ldots) = \prod_{i=1}^{N} \text{PDF}_i(x, \ldots)

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


class CrystalBallDist(Distribution[TD.CrystalBallDistribution]):
    r"""
    Crystal Ball distribution implementation.

    Implements the generalized asymmetrical double-sided Crystal Ball line shape
    as defined in ROOT's RooCrystalBall.

    The probability density function is defined as:

    .. math::

        f(m; m_0, \sigma_L, \sigma_R, \alpha_L, \alpha_R, n_L, n_R) = \begin{cases}
        A_L \cdot \left(B_L - \frac{m - m_0}{\sigma_L}\right)^{-n_L}, & \text{for } \frac{m - m_0}{\sigma_L} < -\alpha_L \\
        \exp\left(-\frac{1}{2} \cdot \left[\frac{m - m_0}{\sigma_L}\right]^2\right), & \text{for } \frac{m - m_0}{\sigma_L} \leq 0 \\
        \exp\left(-\frac{1}{2} \cdot \left[\frac{m - m_0}{\sigma_R}\right]^2\right), & \text{for } \frac{m - m_0}{\sigma_R} \leq \alpha_R \\
        A_R \cdot \left(B_R + \frac{m - m_0}{\sigma_R}\right)^{-n_R}, & \text{otherwise}
        \end{cases}

    where:

    .. math::

        \begin{align}
        A_i &= \left(\frac{n_i}{\alpha_i}\right)^{n_i} \cdot \exp\left(-\frac{\alpha_i^2}{2}\right) \\
        B_i &= \frac{n_i}{\alpha_i} - \alpha_i
        \end{align}

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
        Initialize a CrystalBallDist.

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
    def from_dict(cls, config: TD.CrystalBallDistribution) -> CrystalBallDist:
        """
        Create a CrystalBallDist from a dictionary configuration.

        Args:
            config: Configuration dictionary

        Returns:
            The created CrystalBallDist instance
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
        Create a quadratic distribution:

        >>> dist = GenericDist(name="quadratic", expression="x**2 + 2*x + 1")

        Create a custom exponential with oscillation:

        >>> dist = GenericDist(name="exp_cos", expression="exp(-x**2/2) * cos(y)")

        Create a complex mathematical function:

        >>> dist = GenericDist(name="complex", expression="sin(x) + log(abs(y) + 1)")
    """

    def __init__(self, *, name: str, expression: str):
        """
        Initialize a GenericDist.

        Args:
            name: Name of the distribution
            expression: Mathematical expression string
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
    "crystalball_doublesided_dist": CrystalBallDist,
    "generic_dist": GenericDist,
}


class DistributionSet:
    """
    Collection of distributions for a probabilistic model.

    Manages a set of distribution instances, providing dict-like access
    by distribution name. Handles distribution creation from configuration
    dictionaries and maintains a registry of available distribution types.

    Attributes:
        dists (dict[str, Distribution[Any]]): Mapping from distribution names to Distribution instances.
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
