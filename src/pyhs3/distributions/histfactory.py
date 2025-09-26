"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Annotated, Any, Literal, cast

import pytensor.tensor as pt
from pydantic import BaseModel, Field, PrivateAttr, RootModel, model_validator

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
from pyhs3.domains import Axis
from pyhs3.exceptions import custom_error_msg
from pyhs3.typing.aliases import TensorVar


def interpolate_lin(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    Linear interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} + \alpha \cdot (\text{hi} - \text{nom}) & \text{if } \alpha \geq 0 \\
        \text{nom} + \alpha \cdot (\text{nom} - \text{lo}) & \text{if } \alpha < 0
        \end{cases}
    """
    return cast(
        TensorVar,
        pt.where(alpha >= 0, nom + alpha * (hi - nom), nom + alpha * (nom - lo)),  # type: ignore[no-untyped-call]
    )


def interpolate_log(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    Logarithmic interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} \cdot \left(\frac{\text{hi}}{\text{nom}}\right)^{\alpha} & \text{if } \alpha \geq 0 \\
        \text{nom} \cdot \left(\frac{\text{lo}}{\text{nom}}\right)^{-\alpha} & \text{if } \alpha < 0
        \end{cases}
    """
    return cast(
        TensorVar,
        pt.where(  # type: ignore[no-untyped-call]
            alpha >= 0,
            nom * pt.power(hi / nom, alpha),  # type: ignore[no-untyped-call]
            nom * pt.power(lo / nom, -alpha),  # type: ignore[no-untyped-call]
        ),
    )


def interpolate_parabolic(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    Parabolic interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} + (2s+d)(\alpha-1) + (\text{hi} - \text{nom}) & \text{if } \alpha > 1 \\
        \text{nom} - (2s-d)(\alpha+1) + (\text{lo} - \text{nom}) & \text{if } \alpha < -1 \\
        \text{nom} + s \alpha^2 + d \alpha & \text{otherwise}
        \end{cases}

    where :math:`s = \frac{1}{2}(\text{hi} + \text{lo}) - \text{nom}` and :math:`d = \frac{1}{2}(\text{hi} - \text{lo})`.
    """
    s = 0.5 * (hi + lo) - nom
    d = 0.5 * (hi - lo)

    # Central parabolic region
    central = nom + s * alpha**2 + d * alpha

    # Linear extensions for |alpha| > 1
    high_ext = nom + (2 * s + d) * (alpha - 1) + (hi - nom)
    low_ext = nom - (2 * s - d) * (alpha + 1) + (lo - nom)

    return cast(
        TensorVar,
        pt.where(alpha > 1, high_ext, pt.where(alpha < -1, low_ext, central)),  # type: ignore[no-untyped-call]
    )


def interpolate_poly6(
    alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    r"""
    6th-order polynomial interpolation between hi/lo values.

    .. math::

        f(\alpha) = \begin{cases}
        \text{nom} + \alpha (\text{hi} - \text{nom}) & \text{if } \alpha > 1 \\
        \text{nom} + \alpha (\text{nom} - \text{lo}) & \text{if } \alpha < -1 \\
        \text{nom} + \alpha (S + \alpha A (15 + \alpha^2 (3\alpha^2 - 10))) & \text{otherwise}
        \end{cases}

    where :math:`S = \frac{1}{2}(\text{hi} - \text{lo})` and :math:`A = \frac{1}{16}(\text{hi} + \text{lo} - 2\text{nom})`.
    """
    S = 0.5 * (hi - lo)
    A = (1.0 / 16.0) * (hi + lo - 2 * nom)

    # 6th-order polynomial for |alpha| <= 1
    poly_term = S + alpha * A * (15 + alpha**2 * (3 * alpha**2 - 10))
    central = nom + alpha * poly_term

    # Linear extensions for |alpha| > 1
    high_ext = nom + alpha * (hi - nom)
    low_ext = nom + alpha * (nom - lo)

    return cast(
        TensorVar,
        pt.where(alpha > 1, high_ext, pt.where(alpha < -1, low_ext, central)),  # type: ignore[no-untyped-call]
    )


def apply_interpolation(
    method: str, alpha: TensorVar, nom: TensorVar, hi: TensorVar, lo: TensorVar
) -> TensorVar:
    """Apply the specified interpolation method."""
    if method == "lin":
        return interpolate_lin(alpha, nom, hi, lo)
    if method == "log":
        return interpolate_log(alpha, nom, hi, lo)
    if method == "parabolic":
        return interpolate_parabolic(alpha, nom, hi, lo)
    if method == "poly6":
        return interpolate_poly6(alpha, nom, hi, lo)
    # Default to linear interpolation
    return interpolate_lin(alpha, nom, hi, lo)


class BinnedAxis(Axis):
    """
    Binned axis specification for HistFactory distributions.

    Extends the base Axis class to support binned data structures used in
    HistFactory models. Supports both explicit bin edges and uniform binning.
    """

    nbins: int | None = Field(default=None, description="Number of bins")
    edges: list[float] | None = Field(default=None, description="Explicit bin edges")

    @model_validator(mode="after")
    def validate_binning(self) -> BinnedAxis:
        """Ensure either nbins or edges is provided, but not both."""
        if self.nbins is None and self.edges is None:
            msg = f"BinnedAxis '{self.name}' must specify either 'nbins' or 'edges'"
            raise ValueError(msg)
        if self.nbins is not None and self.edges is not None:
            msg = f"BinnedAxis '{self.name}' cannot specify both 'nbins' and 'edges'"
            raise ValueError(msg)
        if self.edges is not None and len(self.edges) < 2:
            msg = f"BinnedAxis '{self.name}' must have at least 2 edges"
            raise ValueError(msg)
        return self

    def get_nbins(self) -> int:
        """Get the number of bins."""
        if self.nbins is not None:
            return self.nbins
        if self.edges is not None:
            return len(self.edges) - 1
        msg = f"BinnedAxis '{self.name}' has no binning information"
        raise ValueError(msg)


class ModifierData(BaseModel):
    """Base class for modifier data."""


class SampleData(BaseModel):
    """Sample data containing bin contents and errors."""

    contents: list[float]
    errors: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_lengths(self) -> SampleData:
        """Ensure contents and errors have same length if both provided."""
        if self.errors is not None and len(self.contents) != len(self.errors):
            msg = f"Sample data contents ({len(self.contents)}) and errors ({len(self.errors)}) must have same length"
            raise ValueError(msg)
        return self


class NormSysData(ModifierData):
    """Data for normsys modifier."""

    hi: float
    lo: float
    interpolation: str = Field(default="lin")


class HistoSysData(ModifierData):
    """Data for histosys modifier."""

    hi: SampleData
    lo: SampleData
    interpolation: str = Field(default="lin")


class ShapeSysData(ModifierData):
    """Data for shapesys modifier."""

    vals: list[float]


class Modifier(BaseModel):
    """Base class for all HistFactory modifiers."""

    type: str
    parameter: str | None = Field(default=None)
    parameters: list[str] | None = Field(default=None)
    constraint: Literal["Gauss", "Poisson", "LogNormal"] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_parameters(self) -> Modifier:
        """Ensure either parameter or parameters is provided."""
        if self.parameter is None and self.parameters is None:
            msg = f"Modifier '{self.type}' must specify either 'parameter' or 'parameters'"
            raise ValueError(msg)
        if self.parameter is not None and self.parameters is not None:
            msg = f"Modifier '{self.type}' cannot specify both 'parameter' and 'parameters'"
            raise ValueError(msg)
        return self


class NormFactorModifier(Modifier):
    """Normalization factor modifier (simple scaling by parameter value)."""

    type: Literal["normfactor"] = "normfactor"
    parameter: str


class NormSysModifier(Modifier):
    """Normalization systematic modifier (with hi/lo interpolation)."""

    type: Literal["normsys"] = "normsys"
    parameter: str
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: NormSysData


class HistoSysModifier(Modifier):
    """Additive correlated shape systematic modifier."""

    type: Literal["histosys"] = "histosys"
    parameter: str
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: HistoSysData


class ShapeFactorModifier(Modifier):
    """Uncorrelated multiplicative bin-by-bin scaling modifier."""

    type: Literal["shapefactor"] = "shapefactor"
    parameters: list[str]


class ShapeSysModifier(Modifier):
    """Uncorrelated shape systematic with Poisson constraints."""

    type: Literal["shapesys"] = "shapesys"
    parameter: str
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Poisson"
    data: ShapeSysData


class StatErrorModifier(Modifier):
    """Statistical uncertainty modifier (Barlow-Beeston method)."""

    type: Literal["staterror"] = "staterror"
    parameters: list[str]
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Poisson"


# Union type for all modifier types using discriminated union
ModifierType = Annotated[
    NormFactorModifier
    | NormSysModifier
    | HistoSysModifier
    | ShapeFactorModifier
    | ShapeSysModifier
    | StatErrorModifier,
    Field(discriminator="type"),
    custom_error_msg(
        {
            "union_tag_invalid": "Unknown modifier type '{tag}' does not match any supported modifier types"
        }
    ),
]


class Modifiers(RootModel[list[ModifierType]]):
    """
    Collection of modifiers for a HistFactory sample.

    Manages a set of modifier instances, providing list-like access and
    validation. Handles modifier creation from configuration dictionaries
    and maintains type safety through discriminated unions.
    """

    root: Annotated[
        list[ModifierType],
        custom_error_msg(
            {
                "union_tag_invalid": "Unknown modifier type '{tag}' does not match any supported modifier types"
            }
        ),
    ] = Field(default_factory=list)

    def __iter__(self) -> Iterator[ModifierType]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, index: int) -> ModifierType:
        return self.root[index]


class Sample(BaseModel):
    """HistFactory sample specification."""

    name: str
    data: SampleData
    modifiers: Modifiers = Field(default_factory=Modifiers)


class Samples(RootModel[list[Sample]]):
    """
    Collection of samples for a HistFactory distribution.

    Manages a set of sample instances, providing dict-like access by sample name
    and list-like iteration. Handles sample validation and maintains name uniqueness.
    """

    root: list[Sample] = Field(default_factory=list)
    _map: dict[str, Sample] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._map = {sample.name: sample for sample in self.root}

    def __getitem__(self, item: str | int) -> Sample:
        if isinstance(item, int):
            return self.root[item]
        return self._map[item]

    def __contains__(self, item: str) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Sample]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


class HistFactoryDist(Distribution):
    r"""
    HistFactory probability distribution.

    Implements binned statistical models consisting of histograms (step functions)
    with various modifiers as defined in the HS3 specification. Each HistFactory
    distribution describes one "channel" or "region" of a binned measurement.

    The prediction for a binned region is given as:

    .. math::

        \lambda(x) = \sum_{s \in \text{samples}} \left[
            \left( d_s(x) + \sum_{\delta \in M_\delta} \delta(x,\theta_\delta) \right)
            \prod_{\kappa \in M_\kappa} \kappa(x,\theta_\kappa)
        \right]

    where:
        - :math:`d_s(x)` is the nominal prediction for sample :math:`s`
        - :math:`M_\delta` are additive modifiers (histosys)
        - :math:`M_\kappa` are multiplicative modifiers (normfactor, normsys, shapefactor, etc.)

    Parameters:
        axes (list): Array of axis definitions with binning information
        samples (list): Array of sample definitions with data and modifiers

    Supported Modifiers:
        - normfactor: Multiplicative scaling by parameter value
        - normsys: Multiplicative systematic with hi/lo interpolation
        - histosys: Additive correlated shape systematic
        - shapefactor: Uncorrelated multiplicative bin-by-bin scaling
        - shapesys: Uncorrelated shape systematic with Poisson constraints
        - staterror: Statistical uncertainty via Barlow-Beeston method

    HS3 Reference:
        :hs3:label:`histfactory_dist <hs3.histfactory-distribution>`
    """

    type: Literal["histfactory_dist"] = "histfactory_dist"
    axes: list[BinnedAxis] = Field(..., json_schema_extra={"preprocess": False})
    samples: Samples = Field(..., json_schema_extra={"preprocess": False})

    def expression(self, context: Context) -> TensorVar:
        """
        Build the HistFactory likelihood expression.

        This creates a simultaneous distribution combining:
        1. Main model: Poisson distribution for observed bin counts
        2. Constraint model: Gaussian/Poisson constraints for nuisance parameters

        Args:
            context: Mapping of parameter names to PyTensor variables

        Returns:
            PyTensor expression for the HistFactory likelihood
        """
        # Extract binning information
        total_bins = self._get_total_bins()

        # Process all samples and compute expected rates
        expected_rates = self._compute_expected_rates(context, total_bins)

        # Build main Poisson model for observed data (returns log probability)
        main_log_prob = self._build_main_model(context, expected_rates)

        # Build constraint model for nuisance parameters (returns log probability)
        constraint_log_prob = self._build_constraint_model(context)

        # Combine main and constraint models (sum log probabilities)
        if constraint_log_prob is not None:
            total_log_prob = main_log_prob + constraint_log_prob
            return cast(TensorVar, total_log_prob)

        return main_log_prob

    def _get_total_bins(self) -> int:
        """Calculate total number of bins across all axes."""
        total_bins = 1
        for axis in self.axes:
            total_bins *= axis.get_nbins()
        return total_bins

    def _compute_expected_rates(self, context: Context, total_bins: int) -> TensorVar:
        """
        Compute expected event rates for all bins.

        Applies all modifiers to sample predictions to get final rates.
        """
        # Start with zeros for total prediction
        total_rates = pt.zeros(total_bins)  # type: ignore[no-untyped-call]

        # Process each sample
        for sample in self.samples:
            sample_rates = self._process_sample(context, sample, total_bins)
            total_rates = total_rates + sample_rates

        return cast(TensorVar, total_rates)

    def _process_sample(
        self, context: Context, sample: Sample, total_bins: int
    ) -> TensorVar:
        """Process a single sample with its modifiers."""
        # Get nominal bin contents
        contents = sample.data.contents
        if len(contents) != total_bins:
            msg = (
                f"Sample {sample.name} has {len(contents)} bins, expected {total_bins}"
            )
            raise ValueError(msg)

        nominal_rates = pt.as_tensor_variable(contents)

        # Apply modifiers
        modified_rates = nominal_rates

        # Apply additive modifiers first (histosys)
        for modifier in sample.modifiers:
            if isinstance(modifier, HistoSysModifier):
                modified_rates = self._apply_histosys(context, modified_rates, modifier)

        # Apply multiplicative modifiers (normfactor, normsys, shapefactor, etc.)
        for modifier in sample.modifiers:
            if isinstance(
                modifier,
                (
                    NormFactorModifier,
                    NormSysModifier,
                    ShapeFactorModifier,
                    ShapeSysModifier,
                    StatErrorModifier,
                ),
            ):
                modified_rates = self._apply_multiplicative_modifier(
                    context, modified_rates, modifier
                )

        return cast(TensorVar, modified_rates)

    def _apply_histosys(
        self, context: Context, rates: TensorVar, modifier: HistoSysModifier
    ) -> TensorVar:
        """Apply histosys (additive systematic) modifier."""
        alpha = context[modifier.parameter]

        # Get hi/lo variations
        hi_contents = modifier.data.hi.contents
        lo_contents = modifier.data.lo.contents

        hi_variation = pt.as_tensor_variable(hi_contents)
        lo_variation = pt.as_tensor_variable(lo_contents)
        zero_variation = pt.zeros_like(hi_variation)  # type: ignore[no-untyped-call]

        # Apply interpolation method
        interpolation = modifier.data.interpolation
        variation = apply_interpolation(
            interpolation, alpha, zero_variation, hi_variation, lo_variation
        )

        return cast(TensorVar, rates + variation)

    def _apply_multiplicative_modifier(
        self, context: Context, rates: TensorVar, modifier: ModifierType
    ) -> TensorVar:
        """Apply multiplicative modifiers (normfactor, normsys, etc.)."""
        if isinstance(modifier, NormFactorModifier):
            return self._apply_normfactor(context, rates, modifier)
        if isinstance(modifier, NormSysModifier):
            return self._apply_normsys(context, rates, modifier)
        if isinstance(modifier, ShapeFactorModifier):
            return self._apply_shapefactor(context, rates, modifier)
        if isinstance(modifier, ShapeSysModifier):
            return self._apply_shapesys(context, rates, modifier)
        if isinstance(modifier, StatErrorModifier):
            return self._apply_staterror(context, rates, modifier)
        msg = f"Unknown multiplicative modifier type: {type(modifier)}"
        raise ValueError(msg)

    def _apply_normfactor(
        self, context: Context, rates: TensorVar, modifier: NormFactorModifier
    ) -> TensorVar:
        """Apply normfactor modifier (simple scaling by parameter)."""
        mu = context[modifier.parameter]
        return cast(TensorVar, rates * mu)

    def _apply_normsys(
        self, context: Context, rates: TensorVar, modifier: NormSysModifier
    ) -> TensorVar:
        """Apply normsys modifier (systematic with hi/lo interpolation)."""
        alpha = context[modifier.parameter]

        hi_factor = modifier.data.hi
        lo_factor = modifier.data.lo

        # Apply interpolation method
        interpolation = modifier.data.interpolation
        nominal_factor = pt.constant(1.0)
        hi_factor_tensor = pt.constant(hi_factor)
        lo_factor_tensor = pt.constant(lo_factor)

        factor = apply_interpolation(
            interpolation, alpha, nominal_factor, hi_factor_tensor, lo_factor_tensor
        )

        return cast(TensorVar, rates * factor)

    def _apply_shapefactor(
        self, context: Context, rates: TensorVar, modifier: ShapeFactorModifier
    ) -> TensorVar:
        """Apply shapefactor modifier (uncorrelated bin-by-bin scaling)."""
        param_names = modifier.parameters
        factors = pt.stack([context[name] for name in param_names])
        return cast(TensorVar, rates * factors)

    def _apply_shapesys(
        self, context: Context, rates: TensorVar, modifier: ShapeSysModifier
    ) -> TensorVar:
        """Apply shapesys modifier (shape systematic with constraints)."""
        if modifier.parameters:
            param_names = modifier.parameters
            factors = pt.stack([context[name] for name in param_names])
        elif modifier.parameter:
            # Single parameter case
            param_name = modifier.parameter
            factors = context[param_name]
        else:
            msg = "shapesys modifier missing parameter specification"
            raise ValueError(msg)

        return cast(TensorVar, rates * factors)

    def _apply_staterror(
        self, context: Context, rates: TensorVar, modifier: StatErrorModifier
    ) -> TensorVar:
        """Apply staterror modifier (Barlow-Beeston statistical uncertainties)."""
        param_names = modifier.parameters
        # Staterror is bin-by-bin statistical uncertainty
        # Each bin gets its own gamma parameter
        factors = pt.stack([context[name] for name in param_names])
        return cast(TensorVar, rates * factors)

    def _build_main_model(
        self, context: Context, expected_rates: TensorVar
    ) -> TensorVar:
        """Build the main Poisson model for observed data."""
        # Create a Poisson likelihood for the observed bin counts
        # In HistFactory, we typically have observed data as separate parameters
        observed_data_param = f"{self.name}_observed"

        if observed_data_param in context:
            observed_data = context[observed_data_param]
            # Build sum of individual Poisson likelihoods for each bin
            # log P(observed_i | expected_i) = observed_i * log(expected_i) - expected_i - log(observed_i!)
            log_probs = (
                observed_data * pt.log(expected_rates)
                - expected_rates
                - pt.gammaln(observed_data + 1)
            )
            total_log_prob = pt.sum(log_probs)  # type: ignore[no-untyped-call]
            return cast(
                TensorVar, total_log_prob
            )  # Return log probability for numerical stability
        # If no observed data, return log of expected rates (for testing)
        return cast(TensorVar, pt.sum(pt.log(expected_rates)))  # type: ignore[no-untyped-call]

    def _build_constraint_model(self, context: Context) -> TensorVar | None:
        """Build constraint model for nuisance parameters."""
        constraint_log_probs = []

        # Collect all constraint terms from modifiers
        for sample in self.samples:
            for modifier in sample.modifiers:
                if modifier.constraint:
                    log_prob = self._build_constraint_term(
                        context, modifier, modifier.constraint
                    )
                    if log_prob is not None:
                        constraint_log_probs.append(log_prob)

        if not constraint_log_probs:
            return None

        # Sum all constraint log probabilities
        total_log_prob = pt.sum(pt.stack(constraint_log_probs))  # type: ignore[no-untyped-call]
        return cast(TensorVar, total_log_prob)

    def _build_constraint_term(
        self, context: Context, modifier: ModifierType, constraint: str
    ) -> TensorVar | None:
        """Build a single constraint term (returns log probability)."""
        # Handle single parameter modifiers
        if isinstance(modifier, (NormSysModifier, HistoSysModifier, ShapeSysModifier)):
            param_name = modifier.parameter
            param_value = context[param_name]

            if constraint == "Gauss":
                # Gaussian constraint: N(0, 1)
                # Standard normal log probability: -0.5 * alpha^2 - 0.5 * log(2*pi)
                log_prob = -0.5 * param_value**2 - 0.5 * math.log(2 * math.pi)
                return cast(TensorVar, log_prob)
            if constraint == "Poisson":
                # Poisson constraint
                aux_data = pt.constant(1.0)  # Placeholder - should be from data
                log_prob = (
                    aux_data * pt.log(param_value)
                    - param_value
                    - pt.gammaln(aux_data + 1)
                )
                return cast(TensorVar, log_prob)
            if constraint == "LogNormal":
                # LogNormal constraint: typically used for rate parameters
                # Log-normal with location=0, scale=1: log(mu) ~ N(0, 1)
                log_mu = pt.log(param_value)
                log_prob = -0.5 * log_mu**2 - 0.5 * math.log(2 * math.pi) - log_mu
                return cast(TensorVar, log_prob)

        # Handle multiple parameter modifiers
        elif isinstance(modifier, StatErrorModifier):
            param_names = modifier.parameters
            log_probs = []

            for param_name in param_names:
                param_value = context[param_name]

                if constraint == "Gauss":
                    # Gaussian constraint: N(0, 1)
                    log_prob = -0.5 * param_value**2 - 0.5 * math.log(2 * math.pi)
                elif constraint == "Poisson":
                    # Poisson constraint
                    aux_data = pt.constant(1.0)  # Placeholder - should be from data
                    log_prob = (
                        aux_data * pt.log(param_value)
                        - param_value
                        - pt.gammaln(aux_data + 1)
                    )
                elif constraint == "LogNormal":
                    # LogNormal constraint
                    log_mu = pt.log(param_value)
                    log_prob = -0.5 * log_mu**2 - 0.5 * math.log(2 * math.pi) - log_mu
                else:
                    continue

                log_probs.append(log_prob)

            if log_probs:
                return cast(TensorVar, pt.sum(pt.stack(log_probs)))  # type: ignore[no-untyped-call]

        return None


# Registry of histfactory distributions
distributions: dict[str, type[Distribution]] = {
    "histfactory_dist": HistFactoryDist,
}

# Define what should be exported from this module
__all__ = [
    "HistFactoryDist",
    "distributions",
]
