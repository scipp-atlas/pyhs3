"""
HS3 Distribution implementations.

Provides classes for handling various probability distributions including
Gaussian, Mixture, Product, Crystal Ball, and Generic distributions.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeVar, cast

import pytensor.tensor as pt
import sympy as sp
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    model_validator,
)

from pyhs3.exceptions import custom_error_msg
from pyhs3.generic_parse import analyze_sympy_expr, parse_expression, sympy_to_pytensor
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


DistT = TypeVar("DistT", bound="Distribution")


def process_parameter(
    config: Distribution, param_key: str
) -> tuple[str, float | int | None]:
    """
    Process a parameter that can be either a string reference or a numeric value.

    For numeric values, generates a unique name and returns the numeric value.
    For string values, returns the value as-is with None for the numeric value.

    Args:
        config: The distribution configuration
        param_key: The parameter key to process (e.g., "mean", "sigma")

    Returns:
        Tuple of (processed_name, numeric_value_or_none)
    """
    param_value = getattr(config, param_key)
    if isinstance(param_value, float | int):
        # Generate unique constant name
        constant_name = f"constant_{config.name}_{param_key}"
        return constant_name, param_value
    # It's a string reference - return as-is with no numeric value
    return param_value, None


class Distribution(BaseModel):
    """
    Base class for probability distributions in HS3.

    Provides the foundation for all distribution implementations,
    handling parameter management, constant generation, and symbolic
    expression evaluation using PyTensor.

    Attributes:
        name (str): Name of the distribution.
        type (str): Type identifier for the distribution.
        parameters (list[str]): List of parameter names this distribution depends on.
        constants (dict[str, float | int]): Generated constants for numeric parameter values.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    name: str
    type: str
    _parameters: dict[str, str] = PrivateAttr(default_factory=dict)
    _constants_values: dict[str, float | int] = PrivateAttr(default_factory=dict)

    @property
    def parameters(self) -> dict[str, str]:
        """Access to parameter mapping."""
        return self._parameters

    @property
    def constants(self) -> dict[str, TensorVar]:
        """Convert stored numeric constants to PyTensor constants."""
        return {
            name: cast(TensorVar, pt.constant(value))
            for name, value in self._constants_values.items()
        }

    def expression(self, _: dict[str, TensorVar]) -> TensorVar:
        """
        Unimplemented
        """
        msg = f"Distribution type={self.type} is not implemented."
        raise NotImplementedError(msg)


class GaussianDist(Distribution):
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

    type: Literal["gaussian_dist"] = "gaussian_dist"
    mean: str | float | int
    sigma: str | float | int
    x: str | float | int

    @model_validator(mode="after")
    def process_parameters(self) -> GaussianDist:
        """Process parameters and build the parameters dict with constants."""
        # Process parameters and build the parameters dict
        mean_name, mean_value = process_parameter(self, "mean")
        sigma_name, sigma_value = process_parameter(self, "sigma")
        x_name, x_value = process_parameter(self, "x")

        self._parameters = {"mean": mean_name, "sigma": sigma_name, "x": x_name}

        # Add any generated constants
        if mean_value is not None:
            self._constants_values[mean_name] = mean_value
        if sigma_value is not None:
            self._constants_values[sigma_name] = sigma_value
        if x_value is not None:
            self._constants_values[x_name] = x_value

        return self

    def expression(self, distributionsandparameters: dict[str, TensorVar]) -> TensorVar:
        """
        Builds a symbolic expression for the Gaussian PDF.

        Args:
            distributionsandparameters (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Gaussian PDF.
        """
        # log.info("parameters: ", parameters)
        norm_const = 1.0 / (
            pt.sqrt(2 * math.pi) * distributionsandparameters[self._parameters["sigma"]]
        )
        exponent = pt.exp(
            -0.5
            * (
                (
                    distributionsandparameters[self._parameters["x"]]
                    - distributionsandparameters[self._parameters["mean"]]
                )
                / distributionsandparameters[self._parameters["sigma"]]
            )
            ** 2
        )
        return cast(TensorVar, norm_const * exponent)


class MixtureDist(Distribution):
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

    type: Literal["mixture_dist"] = "mixture_dist"
    summands: list[str]
    coefficients: list[str]
    extended: bool = False

    @model_validator(mode="after")
    def process_parameters(self) -> MixtureDist:
        """Build the parameters dict from coefficients and summands."""
        self._parameters = {name: name for name in [*self.coefficients, *self.summands]}
        return self

    def expression(self, distributionsandparameters: dict[str, TensorVar]) -> TensorVar:
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
        return cast(TensorVar, mixturesum)


class ProductDist(Distribution):
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

    type: Literal["product_dist"] = "product_dist"
    factors: list[str]

    @model_validator(mode="after")
    def process_parameters(self) -> ProductDist:
        """Build the parameters dict from factors."""
        self._parameters = {name: name for name in self.factors}
        return self

    def expression(self, distributionsandparameters: dict[str, TensorVar]) -> TensorVar:
        """
        Evaluate the product distribution.

        Args:
            distributionsandparameters: Mapping of names to pytensor variables

        Returns:
            Symbolic representation of the product PDF
        """
        if not self.factors:
            return cast(TensorVar, pt.constant(1.0))

        pt_factors = pt.stack(
            [distributionsandparameters[factor] for factor in self.factors]
        )
        return cast(TensorVar, pt.prod(pt_factors, axis=0))  # type: ignore[no-untyped-call]


class CrystalBallDist(Distribution):
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

    type: Literal["crystalball_doublesided_dist"] = "crystalball_doublesided_dist"
    alpha_L: str
    alpha_R: str
    m: str
    m0: str
    n_L: str
    n_R: str
    sigma_R: str
    sigma_L: str

    @model_validator(mode="after")
    def process_parameters(self) -> CrystalBallDist:
        """Build the parameters dict from crystal ball parameters."""
        params = [
            self.alpha_L,
            self.alpha_R,
            self.m,
            self.m0,
            self.n_R,
            self.n_L,
            self.sigma_L,
            self.sigma_R,
        ]
        self._parameters = {name: name for name in params}
        return self

    def expression(self, distributionsandparameters: dict[str, TensorVar]) -> TensorVar:
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
            TensorVar,
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


class GenericDist(Distribution):
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

    model_config = ConfigDict(arbitrary_types_allowed=True, serialize_by_alias=True)

    type: Literal["generic_dist"] = "generic_dist"
    expression_str: str = Field(alias="expression")
    _sympy_expr: sp.Expr = PrivateAttr(default=None)
    _dependent_vars: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def setup_expression(self) -> GenericDist:
        """Parse and analyze the expression during initialization."""
        # Parse and analyze the expression during initialization
        self._sympy_expr = parse_expression(self.expression_str)

        # Analyze the expression to determine dependencies
        analysis = analyze_sympy_expr(self._sympy_expr)
        independent_vars = [str(symbol) for symbol in analysis["independent_vars"]]
        self._dependent_vars = [str(symbol) for symbol in analysis["dependent_vars"]]

        # Set parameters based on the analyzed expression
        self._parameters = {var: var for var in independent_vars}
        return self

    def expression(self, distributionsandparameters: dict[str, TensorVar]) -> TensorVar:
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
        variables = [
            distributionsandparameters[name] for name in self._parameters.values()
        ]

        # Convert using the pre-parsed sympy expression
        result = sympy_to_pytensor(self._sympy_expr, variables)

        return cast(TensorVar, result)


class PoissonDist(Distribution):
    r"""
    Poisson probability distribution.

    Implements the Poisson probability mass function:

    .. math::

        P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

    Parameters:
        mean (str): Parameter name for the rate parameter (λ).
        x (str): Input variable name (discrete count).
    """

    type: Literal["poisson_dist"] = "poisson_dist"
    mean: str | float | int
    x: str | float | int

    @model_validator(mode="after")
    def process_parameters(self) -> PoissonDist:
        """Process parameters and build the parameters dict with constants."""
        # Process parameters and build the parameters dict
        mean_name, mean_value = process_parameter(self, "mean")
        x_name, x_value = process_parameter(self, "x")

        self._parameters = {"mean": mean_name, "x": x_name}

        # Add any generated constants
        if mean_value is not None:
            self._constants_values[mean_name] = mean_value
        if x_value is not None:
            self._constants_values[x_name] = x_value

        return self

    def expression(self, distributionsandparameters: dict[str, TensorVar]) -> TensorVar:
        """
        Builds a symbolic expression for the Poisson PMF.

        Args:
            distributionsandparameters (dict): Mapping of names to pytensor variables.

        Returns:
            pytensor.tensor.variable.TensorVariable: Symbolic representation of the Poisson PMF.
        """
        mean = distributionsandparameters[self._parameters["mean"]]
        x = distributionsandparameters[self._parameters["x"]]

        # Poisson PMF: λ^k * e^(-λ) / k!
        # Using pt.gammaln for log(k!) = log(Γ(k+1))
        log_pmf = x * pt.log(mean) - mean - pt.gammaln(x + 1)
        return cast(TensorVar, pt.exp(log_pmf))


registered_distributions: dict[str, type[Distribution]] = {
    "gaussian_dist": GaussianDist,
    "mixture_dist": MixtureDist,
    "product_dist": ProductDist,
    "crystalball_doublesided_dist": CrystalBallDist,
    "generic_dist": GenericDist,
    "poisson_dist": PoissonDist,
}

# Type alias for all distribution types using discriminated union
DistributionType = Annotated[
    GaussianDist
    | MixtureDist
    | ProductDist
    | CrystalBallDist
    | GenericDist
    | PoissonDist,
    Field(discriminator="type"),
]


class Distributions(RootModel[list[DistributionType]]):
    """
    Collection of distributions for a probabilistic model.

    Manages a set of distribution instances, providing dict-like access
    by distribution name. Handles distribution creation from configuration
    dictionaries and maintains a registry of available distribution types.

    Attributes:
        dists: Mapping from distribution names to Distribution instances.
    """

    root: Annotated[
        list[DistributionType],
        custom_error_msg(
            {
                "union_tag_invalid": "Unknown distribution type '{tag}' does not match any of the expected distributions: {expected_tags}"
            }
        ),
    ] = Field(default_factory=list)
    _map: dict[str, Distribution] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {dist.name: dist for dist in self.root}

    def __getitem__(self, item: str) -> Distribution:
        return self._map[item]

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Distribution]:  # type: ignore[override]  # https://github.com/pydantic/pydantic/issues/8872
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)
