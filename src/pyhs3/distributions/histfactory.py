"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import pytensor.tensor as pt

from pyhs3.context import Context
from pyhs3.distributions.core import Distribution
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
    axes: list[dict[str, Any]]
    samples: list[dict[str, Any]]

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
            if "nbins" in axis:
                total_bins *= axis["nbins"]
            elif "edges" in axis:
                total_bins *= len(axis["edges"]) - 1
            else:
                msg = f"Axis {axis.get('name', 'unnamed')} missing nbins or edges"
                raise ValueError(msg)
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
        self, context: Context, sample: dict[str, Any], total_bins: int
    ) -> TensorVar:
        """Process a single sample with its modifiers."""
        # Get nominal bin contents
        contents = sample["data"]["contents"]
        if len(contents) != total_bins:
            msg = f"Sample {sample.get('name', 'unnamed')} has {len(contents)} bins, expected {total_bins}"
            raise ValueError(msg)

        nominal_rates = pt.as_tensor_variable(contents)

        # Apply modifiers
        modified_rates = nominal_rates

        # Apply additive modifiers first (histosys)
        for modifier in sample.get("modifiers", []):
            if modifier["type"] == "histosys":
                modified_rates = self._apply_histosys(context, modified_rates, modifier)

        # Apply multiplicative modifiers (normfactor, normsys, shapefactor, etc.)
        for modifier in sample.get("modifiers", []):
            if modifier["type"] in [
                "normfactor",
                "normsys",
                "shapefactor",
                "shapesys",
                "staterror",
            ]:
                modified_rates = self._apply_multiplicative_modifier(
                    context, modified_rates, modifier
                )

        return cast(TensorVar, modified_rates)

    def _apply_histosys(
        self, context: Context, rates: TensorVar, modifier: dict[str, Any]
    ) -> TensorVar:
        """Apply histosys (additive systematic) modifier."""
        param_name = modifier["parameter"]
        alpha = context[param_name]

        # Get hi/lo variations
        hi_contents = modifier["data"]["hi"]["contents"]
        lo_contents = modifier["data"]["lo"]["contents"]

        hi_variation = pt.as_tensor_variable(hi_contents)
        lo_variation = pt.as_tensor_variable(lo_contents)
        zero_variation = pt.zeros_like(hi_variation)  # type: ignore[no-untyped-call]

        # Apply interpolation method if specified
        interpolation = modifier.get("interpolation", "lin")
        variation = apply_interpolation(
            interpolation, alpha, zero_variation, hi_variation, lo_variation
        )

        return cast(TensorVar, rates + variation)

    def _apply_multiplicative_modifier(
        self, context: Context, rates: TensorVar, modifier: dict[str, Any]
    ) -> TensorVar:
        """Apply multiplicative modifiers (normfactor, normsys, etc.)."""
        modifier_type = modifier["type"]

        if modifier_type == "normfactor":
            return self._apply_normfactor(context, rates, modifier)
        if modifier_type == "normsys":
            return self._apply_normsys(context, rates, modifier)
        if modifier_type == "shapefactor":
            return self._apply_shapefactor(context, rates, modifier)
        if modifier_type == "shapesys":
            return self._apply_shapesys(context, rates, modifier)
        if modifier_type == "staterror":
            return self._apply_staterror(context, rates, modifier)
        msg = f"Unknown multiplicative modifier type: {modifier_type}"
        raise ValueError(msg)

    def _apply_normfactor(
        self, context: Context, rates: TensorVar, modifier: dict[str, Any]
    ) -> TensorVar:
        """Apply normfactor modifier (simple scaling by parameter)."""
        param_name = modifier["parameter"]
        mu = context[param_name]
        return cast(TensorVar, rates * mu)

    def _apply_normsys(
        self, context: Context, rates: TensorVar, modifier: dict[str, Any]
    ) -> TensorVar:
        """Apply normsys modifier (systematic with hi/lo interpolation)."""
        param_name = modifier["parameter"]
        alpha = context[param_name]

        hi_factor = modifier["data"]["hi"]
        lo_factor = modifier["data"]["lo"]

        # Apply interpolation method if specified
        interpolation = modifier.get("interpolation", "lin")
        nominal_factor = pt.constant(1.0)
        hi_factor_tensor = pt.constant(hi_factor)
        lo_factor_tensor = pt.constant(lo_factor)

        factor = apply_interpolation(
            interpolation, alpha, nominal_factor, hi_factor_tensor, lo_factor_tensor
        )

        return cast(TensorVar, rates * factor)

    def _apply_shapefactor(
        self, context: Context, rates: TensorVar, modifier: dict[str, Any]
    ) -> TensorVar:
        """Apply shapefactor modifier (uncorrelated bin-by-bin scaling)."""
        param_names = modifier["parameters"]
        factors = pt.stack([context[name] for name in param_names])
        return cast(TensorVar, rates * factors)

    def _apply_shapesys(
        self, context: Context, rates: TensorVar, modifier: dict[str, Any]
    ) -> TensorVar:
        """Apply shapesys modifier (shape systematic with constraints)."""
        if "parameters" in modifier:
            param_names = modifier["parameters"]
            factors = pt.stack([context[name] for name in param_names])
        elif "parameter" in modifier:
            # Single parameter case
            param_name = modifier["parameter"]
            factors = context[param_name]
        else:
            msg = "shapesys modifier missing parameter specification"
            raise ValueError(msg)

        return cast(TensorVar, rates * factors)

    def _apply_staterror(
        self, context: Context, rates: TensorVar, modifier: dict[str, Any]
    ) -> TensorVar:
        """Apply staterror modifier (Barlow-Beeston statistical uncertainties)."""
        param_names = modifier["parameters"]
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
            for modifier in sample.get("modifiers", []):
                constraint = modifier.get("constraint")
                if constraint:
                    log_prob = self._build_constraint_term(
                        context, modifier, constraint
                    )
                    if log_prob is not None:
                        constraint_log_probs.append(log_prob)

        if not constraint_log_probs:
            return None

        # Sum all constraint log probabilities
        total_log_prob = pt.sum(pt.stack(constraint_log_probs))  # type: ignore[no-untyped-call]
        return cast(TensorVar, total_log_prob)

    def _build_constraint_term(
        self, context: Context, modifier: dict[str, Any], constraint: str
    ) -> TensorVar | None:
        """Build a single constraint term (returns log probability)."""
        if constraint == "Gauss":
            # Gaussian constraint: N(0, 1)
            param_name = modifier["parameter"]
            alpha = context[param_name]
            # Standard normal log probability: -0.5 * alpha^2 - 0.5 * log(2*pi)
            log_prob = -0.5 * alpha**2 - 0.5 * math.log(2 * math.pi)
            return cast(TensorVar, log_prob)
        if constraint == "Poisson":
            # Poisson constraint (for shapesys, staterror)
            if "parameters" in modifier:
                # Multiple parameters (e.g., staterror)
                param_names = modifier["parameters"]
                log_probs = []
                for param_name in param_names:
                    gamma = context[param_name]
                    # For staterror, the auxiliary data is typically the nominal value
                    # We'll use a simplified form: Pois(aux_data | gamma)
                    aux_data = pt.constant(1.0)  # Placeholder - should be from data
                    log_prob = (
                        aux_data * pt.log(gamma) - gamma - pt.gammaln(aux_data + 1)
                    )
                    log_probs.append(log_prob)
                return cast(TensorVar, pt.sum(pt.stack(log_probs)))  # type: ignore[no-untyped-call]
            # Single parameter case
            param_name = modifier["parameter"]
            gamma = context[param_name]
            aux_data = pt.constant(1.0)  # Placeholder
            log_prob = aux_data * pt.log(gamma) - gamma - pt.gammaln(aux_data + 1)
            return cast(TensorVar, log_prob)
        if constraint == "LogNormal":
            # LogNormal constraint: typically used for rate parameters
            param_name = modifier["parameter"]
            mu = context[param_name]
            # Log-normal with location=0, scale=1: log(mu) ~ N(0, 1)
            log_mu = pt.log(mu)
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
