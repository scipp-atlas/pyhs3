"""
HistFactory Distribution implementation.

Provides the HistFactoryDist class for handling binned statistical models
with samples and modifiers as defined in the HS3 specification.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, cast

import hist
import numpy as np
import pytensor.tensor as pt
from pydantic import Field, PrivateAttr, model_validator

from pyhs3.axes import BinnedAxes
from pyhs3.context import Context

# Import existing distributions for constraint terms
from pyhs3.distributions.basic import GaussianDist, PoissonDist
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.histfactory.data import SampleData
from pyhs3.distributions.histfactory.modifiers import (
    HasConstraint,
    Modifier,
    ParameterModifier,
    StatErrorModifier,
)
from pyhs3.distributions.histfactory.samples import Sample, Samples
from pyhs3.networks import HasDependencies, HasInternalNodes
from pyhs3.typing.aliases import TensorVar


class ModifierNode(HasDependencies):
    """Wrapper giving a modifier a globally unique name for the dependency graph.

    Modifiers can share the same name across samples/types (e.g., a ``Lumi``
    modifier appearing in several places), but the dependency graph requires
    unique node identifiers.  This lightweight wrapper provides the unique name
    while delegating all functionality to the wrapped modifier.
    """

    def __init__(self, name: str, modifier: Modifier):
        """Store the unique node name and the wrapped modifier."""
        self.name = name
        self._modifier = modifier

    @property
    def dependencies(self) -> set[str]:
        """Parameter names the wrapped modifier depends on."""
        return self._modifier.dependencies

    def expression(self, context: Context) -> TensorVar:
        """Delegate expression building to the wrapped modifier."""
        return self._modifier.expression(context)


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
    axes: BinnedAxes = Field(..., json_schema_extra={"preprocess": False})
    samples: Samples = Field(..., json_schema_extra={"preprocess": False})
    barlow_beeston_method: Literal["full", "lite"] = Field(
        default="lite",
        json_schema_extra={"preprocess": False},
    )
    _normalizable: bool = PrivateAttr(default=False)

    # Per-bin Poisson log-probabilities built by likelihood() (via
    # _build_main_model), reused by log_likelihood() so that a model build
    # calling both methods for the same context does not rebuild the
    # interpolation-spline/per-bin-modifier subgraph (_compute_expected_rates)
    # and the Poisson log-pmf subgraph (_bin_log_probs) a second time.  This is
    # a per-instance cache, and the same HistFactoryDistChannel instance can be
    # reused across multiple Model builds (e.g. Workspace.model() called more
    # than once on the same workspace), each supplying its own Context with
    # fresh parameter tensors.  _cached_context holds the exact Context object
    # that populated the other two fields; callers must check
    # `self._cached_context is context` (object identity, not equality) before
    # trusting them, and refresh all three fields together whenever the cache
    # is (re)populated.
    _cached_context: Context | None = PrivateAttr(default=None)
    _cached_expected_rates: TensorVar | None = PrivateAttr(default=None)
    _cached_bin_log_probs: TensorVar | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_staterror(self) -> HistFactoryDistChannel:
        total_bins = self.axes.get_total_bins()
        for sample in self.samples:
            for mod in sample.modifiers:
                if not isinstance(mod, StatErrorModifier):
                    continue
                if self.barlow_beeston_method == "full" and mod.data is None:
                    msg = (
                        f"StatErrorModifier '{mod.name}' in channel '{self.name}' "
                        f"requires 'data' (uncertainties) in BB-full mode"
                    )
                    raise ValueError(msg)
                if self.barlow_beeston_method == "lite" and mod.data is not None:
                    msg = (
                        f"StatErrorModifier '{mod.name}' in channel '{self.name}' "
                        f"must not specify 'data' in BB-lite mode; per-bin errors "
                        f"come from the sample data"
                    )
                    raise ValueError(msg)
                if not mod.parameters:
                    mod.parameters = [
                        f"staterror_{self.name}_bin{i}" for i in range(total_bins)
                    ]
        return self

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

        # Process all samples and compute expected rates.  Cached so
        # log_likelihood() can reuse this subgraph for the same context
        # instead of rebuilding it.
        expected_rates = self._compute_expected_rates(context, total_bins)
        self._cached_expected_rates = expected_rates
        self._cached_context = context

        # Build main Poisson model for observed data
        return self._build_main_model(context, expected_rates)

    def constraint_specs(
        self,
    ) -> Iterator[tuple[str | None, HasConstraint, SampleData]]:
        """Yield ``(dedup_key, modifier, sample_data)`` for each constraint modifier.

        ``dedup_key`` is the modifier's parameter name for single-parameter
        modifiers (``normsys``, ``histosys``); callers may use it to dedup
        constraints when multiple modifier instances reference the same nuisance
        parameter — within a channel or across channels in a joint fit.  For
        multi-parameter modifiers (``shapesys``, ``staterror``) ``dedup_key``
        is ``None`` — these constraints are channel-local by workspace validation
        and are always emitted as-is.
        """
        for sample in self.samples:
            for modifier in sample.modifiers:
                if not isinstance(modifier, HasConstraint):
                    continue
                if isinstance(modifier, ParameterModifier):
                    yield modifier.parameter, modifier, sample.data
                else:
                    yield None, modifier, sample.data

    def extended_likelihood(
        self, context: Context, _data: TensorVar | None = None
    ) -> TensorVar:
        """Build constraint product for this channel.

        Constraints are deduped by parameter — multiple ``ParameterModifier``
        instances sharing one nuisance parameter (e.g., two ``normsys`` on
        different samples both pointing at ``alpha_lumi``) emit a single
        constraint factor, not one per modifier.  ``ParametersModifier``
        constraints (``shapesys``, ``staterror``) carry per-bin nominal yields
        and are always emitted per-modifier.
        """
        seen: set[str] = set()
        constraint_probs: list[TensorVar] = []
        for dedup_key, modifier, sample_data in self.constraint_specs():
            # Skip StatErrorModifier in lite mode - constraint built at channel level
            if self.barlow_beeston_method == "lite" and isinstance(
                modifier, StatErrorModifier
            ):
                continue
            if dedup_key is not None:
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
            constraint_probs.append(modifier.make_constraint(context, sample_data))

        # Add channel-level BB-lite constraint if in lite mode
        if self.barlow_beeston_method == "lite":
            lite_constraint = self._make_barlow_beeston_lite_constraint(context)
            if lite_constraint is not None:
                constraint_probs.append(lite_constraint)

        if not constraint_probs:
            return pt.constant(1.0)
        return cast(TensorVar, pt.prod(pt.stack(constraint_probs)))  # type: ignore[no-untyped-call]

    def _get_total_bins(self) -> int:
        """Calculate total number of bins across all axes."""
        return self.axes.get_total_bins()

    def _find_staterror_modifier(self) -> StatErrorModifier | None:
        """Find the first staterror modifier from any sample.

        In BB-lite mode, staterror modifiers share gamma parameters across samples,
        so we only need one modifier to get the parameter names and constraint type.
        """
        for sample in self.samples:
            for modifier in sample.modifiers:
                if isinstance(modifier, StatErrorModifier):
                    return modifier
        return None

    def _make_barlow_beeston_lite_constraint(
        self, context: Context
    ) -> TensorVar | None:
        """Build BB-lite constraint from combined sample uncertainties.

        BB-lite uses shared gamma parameters across samples with a channel-level
        constraint built from combined MC statistical uncertainties.

        The constraint can be either Poisson or Gaussian:
        - Poisson: Poisson(tau | gamma * tau) where tau = (nu/sigma)^2
        - Gaussian: N(1.0 | gamma, relerr) where relerr = sigma/nu

        Combined uncertainties from samples:
        - total_nominal = sum(sample.data.contents)
        - total_sigma = sqrt(sum(sample.data.errors^2))
        """
        total_bins = self._get_total_bins()

        # Find staterror modifier to get gamma params and constraint type
        staterror_mod = self._find_staterror_modifier()
        if staterror_mod is None:
            return None

        gamma_params = staterror_mod.parameters
        constraint_type = staterror_mod.constraint

        # Compute combined uncertainties from sample.data.errors
        total_nominal = np.zeros(total_bins)
        total_variance = np.zeros(total_bins)

        for sample in self.samples:
            # Only include samples that have the staterror modifier
            has_staterror = any(
                isinstance(mod, StatErrorModifier) for mod in sample.modifiers
            )
            if has_staterror:
                total_nominal += np.array(sample.data.contents)
                total_variance += np.square(sample.data.errors)

        total_sigma = np.sqrt(total_variance)

        augmented_context = dict(context)
        dists: list[GaussianDist | PoissonDist] = []

        for i, param_name in enumerate(gamma_params):
            nu, sigma = total_nominal[i], total_sigma[i]

            # Skip bins with zero nominal yield; sigma=0 is caught by the parsing layer
            if nu <= 0:
                continue

            if constraint_type == "Poisson":
                # Poisson: Poisson(tau | gamma * tau) where tau = (nu/sigma)^2
                tau = (nu / sigma) ** 2
                scaled_name = f"{param_name}_scaled"
                augmented_context[scaled_name] = context[param_name] * tau
                dists.append(
                    PoissonDist(
                        name=f"constraint_bblite_{self.name}_{i}",
                        x=float(tau),
                        mean=scaled_name,
                    )
                )
            else:  # "Gauss"
                # Gaussian: N(1.0 | gamma, relerr) where relerr = sigma / nu
                relerr = sigma / nu
                sigma_name = f"{param_name}_sigma"
                augmented_context[sigma_name] = pt.constant(relerr)
                dists.append(
                    GaussianDist(
                        name=f"constraint_bblite_{self.name}_{i}",
                        x=1.0,
                        mean=param_name,
                        sigma=sigma_name,
                    )
                )

        if not dists:
            return None

        # Evaluate all distributions with augmented context and multiply
        factors = []
        for dist in dists:
            dist_ctx = Context({**augmented_context, **dist.constants})
            factors.append(dist.expression(dist_ctx))

        return cast(TensorVar, pt.prod(pt.stack(factors), axis=0))  # type: ignore[no-untyped-call]

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
        """Process a single sample with its modifiers.

        The HistFactory formula is lambda = (N + sum(delta_histosys(N))) * prod(kappa_multiplicative).
        Additive variations (histosys) must each be computed against the sample
        nominal N, then summed; multiplicative modifiers apply to the combined
        result. Applying modifiers sequentially against accumulating rates would
        cause histosys to compute its variation against already-scaled rates,
        violating the formula when a multiplicative modifier precedes it.
        """
        contents = sample.data.contents
        if len(contents) != total_bins:
            msg = (
                f"Sample {sample.name} has {len(contents)} bins, expected {total_bins}"
            )
            raise ValueError(msg)

        nominal_rates = pt.as_tensor_variable(contents)

        # Pass 1: accumulate additive variations against the nominal.
        # modifier.apply(ctx, nominal) returns nominal + variation; subtract to
        # isolate the variation so multiple histosys modifiers each reference
        # the same nominal rather than each other's output.
        additive_sum = pt.zeros(total_bins)
        for modifier in sample.modifiers:
            if modifier.is_additive:
                additive_sum = additive_sum + (
                    modifier.apply(context, nominal_rates) - nominal_rates
                )

        rates = nominal_rates + additive_sum

        # Pass 2: apply multiplicative modifiers to (nominal + Σ additive).
        for modifier in sample.modifiers:
            if modifier.is_multiplicative:
                rates = modifier.apply(context, rates)

        return cast(TensorVar, rates)

    def _bin_log_probs(self, context: Context, expected_rates: TensorVar) -> TensorVar:
        r"""Per-bin Poisson log-probabilities for the observed bin counts.

        Observed data must be provided in the context as '{name}_observed' where
        name is the HistFactory distribution name. This is a required parameter
        for likelihood evaluation.

        Returns:
            PyTensor expression for the per-bin Poisson log-probabilities,
            :math:`\log P(observed_i \mid expected_i)`.
        """
        # Observed data is required - no defensive programming needed
        observed_data_param = f"{self.name}_observed"
        observed_data = context[observed_data_param]

        # Observables are reshaped to (N, 1) by the model builder for broadcasting.
        # Flatten to 1-D here so element-wise Poisson matches the (N,) expected_rates.
        if observed_data.ndim == 2:
            observed_data = observed_data[:, 0]

        # Per-bin Poisson log-probability:
        # log P(observed_i | expected_i) = observed_i * log(expected_i) - expected_i - log(observed_i!)
        #
        # Guard against 0 * log(0) = NaN when observed_i == 0 and expected_i == 0.
        # The Poisson log-pmf for k=0 is just -lambda (since log(0!) = 0), so when
        # observed == 0 we only need -expected_rates; the full expression is only
        # needed (and safe) when observed > 0.  When observed > 0 and expected == 0
        # the result is correctly -inf (P(k>0 | lambda=0) = 0).
        #
        # The log's argument is guarded separately from the switch's output: PyTensor
        # differentiates both branches of a switch, so the untaken `full` branch still
        # contributes a gradient of observed/expected_rates = 0/0 = NaN at this point,
        # and multiplying that NaN by the switch's zero mask does not clear it. Substituting
        # a nonzero placeholder for expected_rates inside the log (only where observed == 0,
        # where `full`'s value and gradient are discarded anyway) keeps both finite.
        safe_rates = pt.switch(pt.eq(observed_data, 0), 1.0, expected_rates)
        full = (
            observed_data * pt.log(safe_rates)
            - expected_rates
            - pt.gammaln(observed_data + 1)
        )
        return cast(
            TensorVar,
            pt.switch(pt.eq(observed_data, 0), -expected_rates, full),
        )

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
        # Cached so log_likelihood() can reuse this subgraph for the same
        # context instead of rebuilding the Poisson log-pmf expression.
        log_probs = self._bin_log_probs(context, expected_rates)
        self._cached_bin_log_probs = log_probs
        self._cached_context = context
        # Convert from log probabilities to probabilities
        probs = pt.exp(log_probs)
        main_prob = pt.prod(probs)  # type: ignore[no-untyped-call]

        return cast(TensorVar, main_prob)

    def log_likelihood(self, context: Context) -> TensorVar:
        """Log-space main Poisson likelihood: sum of per-bin Poisson log-pmfs.

        This is the log-space counterpart of :meth:`likelihood`.  Returning the
        sum of per-bin log-probabilities directly avoids the
        ``log(prod(exp(log_probs)))`` round-trip, whose intermediate product
        underflows float64 to ``0.0`` for channels with many bins or large
        expected counts (turning ``log_prob`` into ``-inf``).

        Args:
            context: Mapping of parameter names to PyTensor variables

        Returns:
            PyTensor expression for the summed Poisson log-probability.
        """
        # Reuse the per-bin log-probabilities cached by likelihood() (via
        # _build_main_model) for the same context, if available, to avoid
        # rebuilding the expected-rates and Poisson log-pmf subgraphs.  Only
        # trust the cache when it was populated by this exact context object
        # (see _cached_context) -- otherwise rebuild and refresh the cache so
        # a later call for this context can reuse it.
        if self._cached_context is context and self._cached_bin_log_probs is not None:
            log_probs = self._cached_bin_log_probs
        else:
            total_bins = self._get_total_bins()
            if (
                self._cached_context is context
                and self._cached_expected_rates is not None
            ):
                expected_rates = self._cached_expected_rates
            else:
                expected_rates = self._compute_expected_rates(context, total_bins)
                self._cached_expected_rates = expected_rates
                self._cached_context = context
            log_probs = self._bin_log_probs(context, expected_rates)
            self._cached_bin_log_probs = log_probs
            self._cached_context = context
        return cast(TensorVar, pt.sum(log_probs))  # type: ignore[no-untyped-call]

    def log_extended_likelihood(self, context: Context) -> TensorVar:
        """Log-space constraint sum for this channel.

        Log-space counterpart of :meth:`extended_likelihood`: returns the sum of
        ``log(constraint)`` terms (deduped by parameter exactly as in
        :meth:`extended_likelihood`) instead of their product, avoiding the
        ``log(prod(...))`` round-trip.

        Args:
            context: Mapping of parameter names to PyTensor variables

        Returns:
            PyTensor expression for the summed log-constraint contribution.
        """
        seen: set[str] = set()
        log_constraints: list[TensorVar] = []
        for dedup_key, modifier, sample_data in self.constraint_specs():
            # Skip StatErrorModifier in lite mode - constraint built at channel level
            if self.barlow_beeston_method == "lite" and isinstance(
                modifier, StatErrorModifier
            ):
                continue
            if dedup_key is not None:
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
            log_constraints.append(
                pt.log(modifier.make_constraint(context, sample_data))
            )

        # Add channel-level BB-lite constraint if in lite mode
        if self.barlow_beeston_method == "lite":
            lite_constraint = self._make_barlow_beeston_lite_constraint(context)
            if lite_constraint is not None:
                log_constraints.append(cast(TensorVar, pt.log(lite_constraint)))

        if not log_constraints:
            return cast(TensorVar, pt.constant(0.0))
        return cast(TensorVar, pt.sum(pt.stack(log_constraints)))  # type: ignore[no-untyped-call]

    def log_expression(self, context: Context) -> TensorVar:
        """Log-probability for the channel: summed Poisson log-pmf + log-constraints.

        Overrides :meth:`Distribution.log_expression` to assemble the channel
        log-probability directly in log space (sum of per-bin Poisson log-pmfs
        plus summed log-constraint terms), rather than taking ``log`` of the
        probability-space :meth:`likelihood`/:meth:`extended_likelihood` product.
        This keeps the result finite where the probability-space product would
        underflow to ``0.0``.

        Args:
            context: Mapping of parameter names to PyTensor variables

        Returns:
            PyTensor expression for the channel log-probability.
        """
        return cast(
            TensorVar,
            self.log_likelihood(context) + self.log_extended_likelihood(context),
        )

    def to_hist(self) -> Any:
        """
        Convert HistFactory channel to hist.Hist with categorical process axis.

        Creates a single histogram combining all samples using a categorical axis.
        The first axis is a categorical axis with sample names (labeled "process"),
        followed by the original binning axes.

        Returns:
            hist.Hist: Histogram with shape (n_samples, *binning_shape) where:
                - Axis 0: Categorical axis "process" with sample names
                - Remaining axes: Original binning axes from self.axes
                - Values: Sample contents (with errors as variances)

        Examples:
            >>> channel = HistFactoryDistChannel(
            ...     name="SR",
            ...     axes=[{"name": "mass", "min": 100, "max": 150, "nbins": 5}],
            ...     samples=[
            ...         {"name": "signal", "data": {"contents": [105, 106, 107, 108, 109], "errors": [0.5, 0.6, 0.7, 0.8, 0.9]}},
            ...         {"name": "background", "data": {"contents": [110, 111, 112, 113, 114], "errors": [0.05, 0.06, 0.07, 0.08, 0.09]}}
            ...     ]
            ... )
            >>> h = channel.to_hist()
            >>> h
            Hist(
              StrCategory(['signal', 'background'], name='process'),
              Regular(5, 100, 150, name='mass'),
              storage=Weight()) # Sum: WeightedSum(value=1095, variance=2.5755)
            >>> h.axes[0]
            StrCategory(['signal', 'background'], name='process')
            >>> h["signal", :]  # Get all mass bins for signal sample
            Hist(Regular(5, 100, 150, name='mass'), storage=Weight()) # Sum: WeightedSum(value=535, variance=2.55)
        """
        # First axis: categorical sample axis (use "process" since "sample" is a protected keyword in hist)
        sample_names = [sample.name for sample in self.samples]
        process_axis = hist.axis.StrCategory(sample_names, name="process")

        # Convert remaining axes to hist.axis objects
        binning_axes = [axis.to_hist() for axis in self.axes]

        # Create histogram with all axes (categorical first, then binning axes)
        h = hist.Hist(process_axis, *binning_axes, storage=hist.storage.Weight())

        # Calculate shape from axes (excluding the sample axis)
        binning_shape = tuple(axis.nbins for axis in self.axes)

        # Fill histogram by iterating over samples
        for i, sample in enumerate(self.samples):
            # Reshape contents and variances for this sample
            contents_nd = np.array(sample.data.contents).reshape(binning_shape)
            variances_nd = np.square(sample.data.errors).reshape(binning_shape)

            # Set values for this sample using integer indexing
            # The sample order matches the categorical axis order, so h[i, ...] corresponds to sample.name
            stacked = np.stack([contents_nd, variances_nd], axis=-1)
            h[i, ...] = stacked

        return h


# Registry of histfactory distributions
distributions: dict[str, type[Distribution]] = {
    "histfactory_dist": HistFactoryDistChannel,
}

# Define what should be exported from this module
__all__ = [
    "HistFactoryDistChannel",
    "distributions",
]
