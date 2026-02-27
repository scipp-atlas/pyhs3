.. _normalization:

==================
PDF Normalization
==================

.. currentmodule:: pyhs3

Overview
========

In pyhs3, **all continuous distributions are automatically normalized** over the domain of their observables. This ensures that probability density functions (PDFs) integrate to 1.0 over the specified observable range, making them proper probability distributions.

Mathematical Formulation
========================

For a single observable :math:`x` with domain :math:`[x_{\min}, x_{\max}]`, the normalized PDF is:

.. math::

    f_{\text{norm}}(x) = \frac{f(x)}{\int_{x_{\min}}^{x_{\max}} f(x)\,dx}

where :math:`f(x)` is the raw (unnormalized) likelihood and the denominator is the normalization constant :math:`Z`.

For multi-dimensional distributions with observables :math:`x` and :math:`y`, normalization uses the N-dimensional integral:

.. math::

    Z = \int_{x_{\min}}^{x_{\max}} \int_{y_{\min}}^{y_{\max}} f(x,y)\,dx\,dy

pyhs3 computes this via nested Gauss-Legendre quadrature using Fubini's theorem, treating the inner integral as a function to be integrated over.

pyhs3 Behavior
==============

When you create a model with observables, pyhs3 automatically normalizes distributions:

- **With observables**: Distributions are normalized over the observable domain
- **Without observables**: Distributions remain unnormalized (raw likelihood)
- **HistFactory distributions**: Explicitly opt out of normalization (already normalized by design)

Example: Effect of Normalization
=================================

Let's see how normalization affects a Gaussian distribution over a finite domain:

.. doctest::

   >>> from pyhs3.core import Model
   >>> from pyhs3.distributions.basic import GaussianDist
   >>> from pyhs3.domains import ProductDomain
   >>> from pyhs3.parameter_points import ParameterSet, ParameterPoint
   >>> from pyhs3.distributions import Distributions
   >>> from pyhs3.functions import Functions
   >>> from pyhs3.context import Context
   >>> import pytensor.tensor as pt
   >>> from pytensor.compile.function import function
   >>> from scipy.integrate import quad
   >>>
   >>> # Create a GaussianDist
   >>> gaussian = GaussianDist(name="signal", mean="mu", sigma="sigma", x="x")
   >>>
   >>> # Model WITH observables - normalized
   >>> model_norm = Model(
   ...     parameterset=ParameterSet(
   ...         name="default",
   ...         parameters=[
   ...             ParameterPoint(name="mu", value=130.0),
   ...             ParameterPoint(name="sigma", value=10.0),
   ...         ],
   ...     ),
   ...     distributions=Distributions([gaussian]),
   ...     domain=ProductDomain(name="default"),
   ...     functions=Functions([]),
   ...     observables={"x": (100.0, 160.0)},  # Finite domain
   ...     progress=False,
   ... )
   >>>
   >>> # Model WITHOUT observables - raw PDF
   >>> model_unnorm = Model(
   ...     parameterset=ParameterSet(
   ...         name="default",
   ...         parameters=[
   ...             ParameterPoint(name="mu", value=130.0),
   ...             ParameterPoint(name="sigma", value=10.0),
   ...         ],
   ...     ),
   ...     distributions=Distributions([gaussian]),
   ...     domain=ProductDomain(name="default"),
   ...     functions=Functions([]),
   ...     observables=None,  # No observables
   ...     progress=False,
   ... )
   >>>
   >>> # Compare the expressions using pytensor printing
   >>> from pytensor.printing import pp
   >>> expr_norm = model_norm.distributions["signal"]
   >>> expr_unnorm = model_unnorm.distributions["signal"]
   >>>
   >>> # The normalized version includes a division by the integral (using Scan for quadrature)
   >>> "Scan" in pp(expr_norm)  # Scan operation for numerical integration
   True
   >>> "Scan" in pp(expr_unnorm)  # No Scan in raw PDF
   False
   >>>
   >>> # Verify normalization: integral over [100, 160] = 1.0
   >>> x_var = model_norm.parameters["x"]
   >>> mu_var = model_norm.parameters["mu"]
   >>> sigma_var = model_norm.parameters["sigma"]
   >>> f_norm = function([x_var, mu_var, sigma_var], expr_norm)
   >>> integral_norm, _ = quad(lambda x: f_norm(x, 130.0, 10.0), 100, 160)
   >>> abs(integral_norm - 1.0) < 1e-5  # Normalized PDF integrates to 1
   True
   >>>
   >>> # Raw PDF integrates to something less than 1 over finite domain
   >>> x_var_u = model_unnorm.parameters["x"]
   >>> mu_var_u = model_unnorm.parameters["mu"]
   >>> sigma_var_u = model_unnorm.parameters["sigma"]
   >>> f_unnorm = function([x_var_u, mu_var_u, sigma_var_u], expr_unnorm)
   >>> integral_unnorm, _ = quad(lambda x: f_unnorm(x, 130.0, 10.0), 100, 160)
   >>> 0.99 < integral_unnorm < 1.0  # Gaussian over [100, 160] is almost all of the probability
   True
   >>> abs(integral_unnorm - 1.0) > 1e-5  # But not exactly 1.0
   True

Providing Analytical Normalization
===================================

Distributions can provide analytical normalization expressions to avoid numerical integration. Override the ``normalization_expression()`` method to return the antiderivative :math:`F(x)`:

.. doctest::

   >>> from pyhs3.distributions.core import Distribution
   >>> from pyhs3.context import Context
   >>> from pyhs3.typing.aliases import TensorVar
   >>> import pytensor.tensor as pt
   >>> from typing import Literal, ClassVar
   >>> from pydantic import Field, ConfigDict
   >>>
   >>> class LinearDist(Distribution):
   ...     """Distribution with f(x) = x."""
   ...     _parameters: ClassVar = {"x": "x"}
   ...     model_config = ConfigDict(arbitrary_types_allowed=True, serialize_by_alias=True)
   ...     type: Literal["linear_dist"] = Field(default="linear_dist", repr=False)
   ...     def likelihood(self, context: Context) -> TensorVar:
   ...         return context["x"]
   ...     def normalization_expression(
   ...         self, context: Context, observable_name: str
   ...     ) -> TensorVar:
   ...         # Return the antiderivative F(x) = x^2/2
   ...         observable = context[observable_name]
   ...         return observable**2 / 2.0
   ...
   >>>
   >>> # Verify the distribution normalizes correctly
   >>> dist = LinearDist(name="linear", expression="x")
   >>> x_var = pt.dscalar("x")
   >>> from pyhs3.context import Context
   >>> context = Context(parameters={"x": x_var}, observables={"x": (0, 10)})
   >>> expr = dist.expression(context)
   >>> from pytensor.compile.function import function
   >>> f = function([x_var], expr)
   >>> from scipy.integrate import quad
   >>> integral, _ = quad(lambda x: f(x), 0, 10)
   >>> abs(integral - 1.0) < 1e-10  # Perfectly normalized
   True

The framework automatically evaluates :math:`F(x_{\max}) - F(x_{\min})` using the antiderivative you provide. You only need to return the expression—the framework handles bound substitution via ``clone_replace``.

Default Numerical Integration
==============================

When ``normalization_expression()`` returns ``None`` (the default), pyhs3 uses 64-point Gauss-Legendre quadrature to compute the normalization integral numerically. The implementation uses ``pytensor.scan`` for compact symbolic loops that create a single optimized computation graph rather than 64 separate graph copies.

See :func:`~pyhs3.normalization.gauss_legendre_integral` for the implementation details.

Opting Out of Normalization
============================

HistFactory distributions are already normalized by design (binned data with expected counts), so they explicitly opt out by setting ``_normalizable = False``. When this flag is False, the normalization machinery is skipped entirely, and the distribution returns its raw likelihood.

See :class:`~pyhs3.distributions.histfactory.HistFactoryDistChannel` for the implementation.

Integration with Workspace
===========================

The ``Workspace`` class automatically extracts observable domains from data objects when creating a model. When you define data with axes (e.g., ``BinnedData`` with ``axes=[{"name": "x", "min": 100.0, "max": 160.0, "nbins": 60}]``), the workspace identifies ``x`` as an observable with bounds ``(100, 160)`` and passes this to the model.

The model will then normalize all distributions over :math:`x \in [100, 160]` without requiring explicit ``observables`` specification.

See :class:`~pyhs3.core.Workspace` for more details on observable extraction.

See Also
========

- :ref:`defining_components` - Distribution architecture and implementation
- :ref:`architecture` - Overall pyhs3 design patterns
- :class:`~pyhs3.distributions.core.Distribution` - Base distribution class
- :func:`~pyhs3.normalization.gauss_legendre_integral` - Numerical integration routine
