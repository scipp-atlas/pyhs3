"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

import math
from typing import Literal, cast

import pytensor.tensor as pt
from pydantic import Field, model_validator

from pyhs3.context import Context

# Import existing distributions for constraint terms
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.histfactory import interpolations, modifiers
from pyhs3.distributions.histfactory.samples import Sample, Samples
from pyhs3.domains import Axis
from pyhs3.typing.aliases import TensorVar


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


class HistFactoryDist(Distribution):
    r"""
    HistFactory probability distribution.

    Implements binned statistical models consisting of histograms (step functions)
    with various modifiers as defined in the HS3 specification. Each HistFactory
    distribution describes one "channel" or "region" of a binned measurement.

    The total likelihood consists of:
    1. **Main likelihood**: Poisson likelihood for observed bin counts vs expected rates
    2. **Constraint likelihoods**: Constraint terms for nuisance parameters

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

    **Observed Data Convention:**
        Observed data must be provided in the context as ``{name}_observed`` where
        ``name`` is the HistFactory distribution name. This is required for likelihood
        evaluation.

    **Constraint Types:**
        - **Gaussian constraints** (default): histosys, normsys, staterror
        - **Poisson constraints** (default): shapesys
        - All constraint types can be overridden via the ``constraint`` field

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
            if isinstance(modifier, modifiers.HistoSysModifier):
                modified_rates = self._apply_histosys(context, modified_rates, modifier)

        # Apply multiplicative modifiers (normfactor, normsys, shapefactor, etc.)
        for modifier in sample.modifiers:
            if isinstance(
                modifier,
                (
                    modifiers.NormFactorModifier,
                    modifiers.NormSysModifier,
                    modifiers.ShapeFactorModifier,
                    modifiers.ShapeSysModifier,
                    modifiers.StatErrorModifier,
                ),
            ):
                modified_rates = self._apply_multiplicative_modifier(
                    context, modified_rates, modifier
                )

        return cast(TensorVar, modified_rates)

    def _apply_histosys(
        self, context: Context, rates: TensorVar, modifier: modifiers.HistoSysModifier
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
        variation = interpolations.apply_interpolation(
            interpolation, alpha, zero_variation, hi_variation, lo_variation
        )

        return cast(TensorVar, rates + variation)

    def _apply_multiplicative_modifier(
        self, context: Context, rates: TensorVar, modifier: modifiers.ModifierType
    ) -> TensorVar:
        """Apply multiplicative modifiers (normfactor, normsys, etc.)."""
        if isinstance(modifier, modifiers.NormFactorModifier):
            return self._apply_normfactor(context, rates, modifier)
        if isinstance(modifier, modifiers.NormSysModifier):
            return self._apply_normsys(context, rates, modifier)
        if isinstance(modifier, modifiers.ShapeFactorModifier):
            return self._apply_shapefactor(context, rates, modifier)
        if isinstance(modifier, modifiers.ShapeSysModifier):
            return self._apply_shapesys(context, rates, modifier)
        if isinstance(modifier, modifiers.StatErrorModifier):
            return self._apply_staterror(context, rates, modifier)
        msg = f"Unknown multiplicative modifier type: {type(modifier)}"
        raise ValueError(msg)

    def _apply_normfactor(
        self, context: Context, rates: TensorVar, modifier: modifiers.NormFactorModifier
    ) -> TensorVar:
        """Apply normfactor modifier (simple scaling by parameter)."""
        mu = context[modifier.parameter]
        return cast(TensorVar, rates * mu)

    def _apply_normsys(
        self, context: Context, rates: TensorVar, modifier: modifiers.NormSysModifier
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

        factor = interpolations.apply_interpolation(
            interpolation, alpha, nominal_factor, hi_factor_tensor, lo_factor_tensor
        )

        return cast(TensorVar, rates * factor)

    def _apply_shapefactor(
        self,
        context: Context,
        rates: TensorVar,
        modifier: modifiers.ShapeFactorModifier,
    ) -> TensorVar:
        """Apply shapefactor modifier (uncorrelated bin-by-bin scaling)."""
        param_names = modifier.parameters
        factors = pt.stack([context[name] for name in param_names])
        return cast(TensorVar, rates * factors)

    def _apply_shapesys(
        self, context: Context, rates: TensorVar, modifier: modifiers.ShapeSysModifier
    ) -> TensorVar:
        """Apply shapesys modifier (shape systematic with constraints)."""
        # Single parameter case
        param_name = modifier.parameter
        factors = context[param_name]
        return cast(TensorVar, rates * factors)

    def _apply_staterror(
        self, context: Context, rates: TensorVar, modifier: modifiers.StatErrorModifier
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
        """
        Build the main Poisson model for observed data.

        Observed data must be provided in the context as '{name}_observed' where
        name is the HistFactory distribution name. This is a required parameter
        for likelihood evaluation.
        """
        # Create a Poisson likelihood for the observed bin counts
        # Observed data is required - no defensive programming needed
        observed_data_param = f"{self.name}_observed"
        observed_data = context[observed_data_param]

        # Build sum of individual Poisson likelihoods for each bin
        # log P(observed_i | expected_i) = observed_i * log(expected_i) - expected_i - log(observed_i!)
        log_probs = (
            observed_data * pt.log(expected_rates)
            - expected_rates
            - pt.gammaln(observed_data + 1)
        )
        main_log_prob = pt.sum(log_probs)  # type: ignore[no-untyped-call]

        # Add auxiliary data terms from constraint modifiers
        aux_log_probs = []
        for sample in self.samples:
            for modifier in sample.modifiers:
                parameters = getattr(modifier, "parameters", None)
                if modifier.constraint and parameters:
                    # For modifiers with constraints, add auxiliary Poisson terms
                    aux_data_list = modifier.auxdata
                    if isinstance(aux_data_list, list):
                        # For staterror, add Poisson(aux_data | parameter) terms
                        for param_name, aux_data in zip(
                            parameters, aux_data_list, strict=False
                        ):
                            aux_observed = pt.constant(aux_data)
                            aux_expected = context[param_name]
                            aux_log_prob = (
                                aux_observed * pt.log(aux_expected)
                                - aux_expected
                                - pt.gammaln(aux_observed + 1)
                            )
                            aux_log_probs.append(aux_log_prob)

        # Combine main and auxiliary terms
        if aux_log_probs:
            total_aux_log_prob = pt.sum(pt.stack(aux_log_probs))  # type: ignore[no-untyped-call]
            total_log_prob = main_log_prob + total_aux_log_prob
        else:
            total_log_prob = main_log_prob

        return cast(TensorVar, total_log_prob)

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
        self, context: Context, modifier: modifiers.ModifierType, constraint: str
    ) -> TensorVar | None:
        """Build a single constraint term using existing PyHS3 distributions."""
        # Handle single parameter modifiers
        if isinstance(
            modifier,
            (
                modifiers.NormSysModifier,
                modifiers.HistoSysModifier,
                modifiers.ShapeSysModifier,
            ),
        ):
            param_name = modifier.parameter
            aux_data = modifier.auxdata

            return self._create_constraint_distribution(
                constraint, param_name, aux_data, context, modifier
            )

        # Handle multiple parameter modifiers (staterror)
        if isinstance(modifier, modifiers.StatErrorModifier):
            param_names = modifier.parameters
            aux_data_list = modifier.auxdata
            log_probs = []

            for param_name, aux_data in zip(param_names, aux_data_list, strict=False):
                log_prob = self._create_constraint_distribution(
                    constraint, param_name, aux_data, context, modifier
                )
                if log_prob is not None:
                    log_probs.append(log_prob)

            if log_probs:
                return cast(TensorVar, pt.sum(pt.stack(log_probs)))  # type: ignore[no-untyped-call]

        return None

    def _create_constraint_distribution(
        self,
        constraint: str,
        param_name: str,
        aux_data: float,
        context: Context,
        modifier: modifiers.ModifierType,
    ) -> TensorVar | None:
        """Create constraint term using existing PyHS3 distributions."""
        if constraint == "Gauss":
            # Gaussian constraint: Normal(auxdata | mean=parameter, std=sigma)
            param_value = context[param_name]
            aux_tensor = pt.constant(aux_data)

            # Get sigma based on modifier type
            if isinstance(modifier, modifiers.StatErrorModifier):
                # For staterror, sigma comes from the uncertainty data
                param_index = modifier.parameters.index(param_name)
                sigma = modifier.data.uncertainties[param_index]
                sigma_tensor = pt.constant(sigma)
            else:
                # For normsys/histosys, use standard deviation of 1.0
                sigma_tensor = pt.constant(1.0)

            # Gaussian log probability: -0.5 * ((aux - param)/sigma)^2 - 0.5 * log(2*pi*sigma^2)
            diff = (aux_tensor - param_value) / sigma_tensor
            log_prob = -0.5 * diff**2 - 0.5 * pt.log(2 * math.pi * sigma_tensor**2)
            return cast(TensorVar, log_prob)

        if constraint == "Poisson":
            # Poisson constraint with auxiliary data
            param_value = context[param_name]
            aux_tensor = pt.constant(aux_data)

            # For shapesys modifiers, pyhf uses scaled constraint:
            # Poisson(aux_data | param_value * aux_data)
            # For other modifiers, use direct constraint:
            # Poisson(aux_data | param_value)
            if isinstance(modifier, modifiers.ShapeSysModifier):
                # Scaled constraint for shapesys
                rate = param_value * aux_tensor
            else:
                # Direct constraint for other modifiers
                rate = param_value

            # Poisson log probability: k * log(lambda) - lambda - log(k!)
            log_prob = aux_tensor * pt.log(rate) - rate - pt.gammaln(aux_tensor + 1)
            return cast(TensorVar, log_prob)

        if constraint == "LogNormal":
            # LogNormal constraint: log(param) ~ N(0, 1)
            param_value = context[param_name]
            log_mu = pt.log(param_value)
            log_prob = -0.5 * log_mu**2 - 0.5 * math.log(2 * math.pi) - log_mu
            return cast(TensorVar, log_prob)

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
