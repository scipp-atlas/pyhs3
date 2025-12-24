"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import pytensor.tensor as pt
from pydantic import Field

from pyhs3.context import Context

# Import existing distributions for constraint terms
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.histfactory.axes import Axes, BinnedAxis
from pyhs3.distributions.histfactory.modifiers import HasConstraint, Modifier
from pyhs3.distributions.histfactory.samples import Sample, Samples
from pyhs3.networks import HasDependencies, HasInternalNodes
from pyhs3.typing.aliases import TensorVar


class HistFactoryDistChannel(Distribution, HasInternalNodes):
    r"""
    HistFactory probability distribution for a single channel/region.

    Implements binned statistical models consisting of histograms (step functions)
    with various modifiers as defined in the HS3 specification. Each HistFactoryDistChannel
    represents one independent measurement channel/region with its own observed data.
    Multiple channels can be combined in a workspace to form a complete HistFactory model.

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

    Modifier Naming in Dependency Graph:
        Modifiers have simple names (e.g., "lumi") in the HS3 specification, but are
        given unique identifiers in the dependency graph by prepending the full context:
        ``{dist_name}/{sample_name}/{modifier_type}/{modifier_name}``

        This design distinguishes individual modifier instances while allowing parameters
        to indicate correlation - modifiers sharing the same parameter name are correlated.

        Example: Two modifiers both named "lumi" in different samples will have unique
        graph nodes like "SR/signal/normsys/lumi" and "CR/background/normsys/lumi", but
        if they share the same parameter name, they are correlated.

    HS3 Reference:
        :hs3:label:`histfactory_dist <hs3.histfactory-distribution>`
    """

    type: Literal["histfactory_dist"] = "histfactory_dist"
    axes: Axes = Field(..., json_schema_extra={"preprocess": False})
    samples: Samples = Field(..., json_schema_extra={"preprocess": False})

    def get_internal_nodes(self) -> list[Any]:
        """
        Return all internal nodes that need to be in the dependency graph.

        Modifiers can have the same name across different samples/types (e.g., "Lumi"
        appearing in multiple places), but the dependency graph requires unique node
        identifiers. We create wrapper objects that provide unique names while
        delegating to the original modifier's functionality.
        """
        nodes = []

        for sample in self.samples:
            for modifier in sample.modifiers:
                # Create unique node name: {dist_name}/{sample_name}/{modifier_type}/{modifier_name}
                # This ensures uniqueness even when modifier names are reused across samples
                node_name = f"{self.name}/{sample.name}/{modifier.type}/{modifier.name}"

                # Create a lightweight wrapper that provides the unique name for the dependency graph
                # while delegating all functionality to the original modifier
                class ModifierNode(HasDependencies):
                    """
                    Wrapper Modifier to provide a globally unique internal name for the dependency graph.
                    """

                    def __init__(self, name: str, modifier: Modifier):
                        self.name = name
                        self._modifier = modifier

                    @property
                    def dependencies(self) -> set[str]:
                        return self._modifier.dependencies

                    def expression(self, context: Context) -> TensorVar:
                        return self._modifier.expression(context)

                nodes.append(ModifierNode(node_name, modifier))

        return nodes

    @property
    def parameters(self) -> set[str]:
        """Return all parameters used by this HistFactory distribution."""
        params = set()
        # HistFactory distribution requires observed data parameter
        params.add(f"{self.name}_observed")
        # Collect parameters from all modifiers
        for sample in self.samples:
            for modifier in sample.modifiers:
                params.update(modifier.dependencies)
        return params

    def likelihood(self, context: Context) -> TensorVar:
        """
        Build the HistFactory main Poisson likelihood.

        Returns the Poisson probability for observed bin counts vs expected rates.
        Does NOT include constraint terms - those are added via extended_likelihood().

        Args:
            context: Mapping of parameter names to PyTensor variables

        Returns:
            PyTensor expression for the main Poisson model probability
        """
        # Extract binning information
        total_bins = self._get_total_bins()

        # Process all samples and compute expected rates
        expected_rates = self._compute_expected_rates(context, total_bins)

        # Build main Poisson model for observed data
        return self._build_main_model(context, expected_rates)

    def extended_likelihood(
        self, context: Context, _data: TensorVar | None = None
    ) -> TensorVar:
        """
        Build constraint model for nuisance parameters.

        Args:
            context: Mapping of parameter names to PyTensor variables
            data: Not used for HistFactory constraints (observed data is in main model)

        Returns:
            PyTensor expression for the constraint likelihood terms
        """
        constraint_probs = []

        # Collect all constraint terms from modifiers
        for sample in self.samples:
            for modifier in sample.modifiers:
                if isinstance(modifier, HasConstraint):
                    prob = modifier.make_constraint(context, sample.data)
                    constraint_probs.append(prob)

        if not constraint_probs:
            return pt.constant(1.0)

        # Multiply all constraint probabilities
        return cast(TensorVar, pt.prod(pt.stack(constraint_probs)))  # type: ignore[no-untyped-call]

    def _get_total_bins(self) -> int:
        """Calculate total number of bins across all axes."""
        return self.axes.get_total_bins()

    def _compute_expected_rates(self, context: Context, total_bins: int) -> TensorVar:
        """
        Compute expected event rates for all bins.

        Applies all modifiers to sample predictions to get final rates.
        """
        # Start with zeros for total prediction
        total_rates = pt.zeros(total_bins)

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

        # Apply modifiers using pre-computed results where possible
        modified_rates = nominal_rates

        for modifier in sample.modifiers:
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


# Registry of histfactory distributions
distributions: dict[str, type[Distribution]] = {
    "histfactory_dist": HistFactoryDistChannel,
}

# Define what should be exported from this module
__all__ = [
    "Axes",
    "BinnedAxis",
    "HistFactoryDistChannel",
    "distributions",
]
