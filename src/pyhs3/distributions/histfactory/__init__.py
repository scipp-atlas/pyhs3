"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

from typing import Literal, cast

import pytensor.tensor as pt
from pydantic import Field, model_validator

from pyhs3.context import Context

# Import existing distributions for constraint terms
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.histfactory.modifiers import HasConstraint
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
        main_prob = self._build_main_model(context, expected_rates)

        # Build constraint model for nuisance parameters (returns log probability)
        constraint_prob = self._build_constraint_model(context)

        # Combine main and constraint models (sum log probabilities)
        if constraint_prob is not None:
            total_prob = main_prob * constraint_prob
            return cast(TensorVar, total_prob)

        return main_prob

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

        # Apply additive modifiers first
        for modifier in sample.modifiers:
            if modifier.is_additive:
                modified_rates = modifier.apply(context, modified_rates)

        # Apply multiplicative modifiers
        for modifier in sample.modifiers:
            if modifier.is_multiplicative:
                modified_rates = modifier.apply(context, modified_rates)

        return cast(TensorVar, modified_rates)

    def _build_main_model(
        self, context: Context, expected_rates: TensorVar
    ) -> TensorVar:
        """
        Build the main Poisson model for observed data.

        Observed data must be provided in the context as '{name}_observed' where
        name is the HistFactory distribution name. This is a required parameter
        for likelihood evaluation.

        Returns:
            PyTensor expression for the Poisson probability (not log probability)
        """
        # Create a Poisson likelihood for the observed bin counts
        # Observed data is required - no defensive programming needed
        observed_data_param = f"{self.name}_observed"
        observed_data = context[observed_data_param]

        # Build product of individual Poisson probabilities for each bin
        # P(observed_i | expected_i) = exp(observed_i * log(expected_i) - expected_i - log(observed_i!))
        log_probs = (
            observed_data * pt.log(expected_rates)
            - expected_rates
            - pt.gammaln(observed_data + 1)
        )
        # Convert from log probabilities to probabilities
        probs = pt.exp(log_probs)
        main_prob = pt.prod(probs)  # type: ignore[no-untyped-call]

        return cast(TensorVar, main_prob)

    def _build_constraint_model(self, context: Context) -> TensorVar | None:
        """Build constraint model for nuisance parameters."""
        constraint_probs = []

        # Collect all constraint terms from modifiers
        for sample in self.samples:
            # Prepare sample data for constraint calculations
            for modifier in sample.modifiers:
                if isinstance(modifier, HasConstraint):
                    prob = modifier.make_constraint(context, sample.data)
                    if prob is not None:
                        constraint_probs.append(prob)

        if not constraint_probs:
            return None

        # Multiply all constraint probabilities and take log
        return cast(TensorVar, pt.prod(pt.stack(constraint_probs)))  # type: ignore[no-untyped-call]


# Registry of histfactory distributions
distributions: dict[str, type[Distribution]] = {
    "histfactory_dist": HistFactoryDist,
}

# Define what should be exported from this module
__all__ = [
    "HistFactoryDist",
    "distributions",
]
