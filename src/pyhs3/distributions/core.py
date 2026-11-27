"""
Core distribution classes and utilities.

Provides the base Distribution class and common utilities used by both
standard and CMS-specific distribution implementations.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Self, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pydantic import PrivateAttr
from pytensor.graph.traversal import ancestors
from pytensor.tensor.shape import specify_shape

from pyhs3.base import Evaluable
from pyhs3.context import Context
from pyhs3.normalization import gauss_legendre_integral, substitute_variable
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyhs3.distributions import Distributions


@dataclass
class LogProbTerms:
    """
    Structured per-channel contributions to :attr:`pyhs3.Model.log_prob`.

    Distributions describe *what* they contribute via
    :meth:`Distribution.log_prob_terms`; the model owns the channel-dataset
    pairing and decides *how* the pieces enter the joint log-likelihood
    (event weighting, summing over events, global constraint deduplication).

    Attributes:
        per_event: Log-density terms with the observable as a free pt.vector
            input, broadcasting to shape (N, M) for N events and parameter
            batch size M.  The model sums these over the event axis (applying
            per-event weights when present).
        channel: Terms added once per channel (e.g. the -nu yield term of an
            extended mixture): scalars, or (1, M) when built from batched
            parameters; the model combines them onto the (M,) parameter batch
            axis.
        constraints: Log-terms keyed by factor name, depending only on
            nuisance parameters (scalars, or (1, M) when batched).  The model
            adds each name exactly once globally, so constraints shared
            across channels are not double-counted.
    """

    per_event: list[TensorVar] = field(default_factory=list)
    channel: list[TensorVar] = field(default_factory=list)
    constraints: dict[str, TensorVar] = field(default_factory=dict)


class Distribution(Evaluable, ABC):
    """
    Base class for probability distributions in HS3.

    Provides the foundation for all distribution implementations,
    handling parameter management, constant generation, and symbolic
    expression evaluation using PyTensor.

    Distributions separate the main probability model (likelihood) from
    additional extended likelihood terms (e.g., constraints). The complete
    probability is the product of both terms.

    All distributions are automatically normalized over the domain of their
    observables unless explicitly opted out via _normalizable = False.

    Inherits parameter processing functionality from Evaluable.
    Subclasses must implement _expression() to define computation logic.
    """

    _normalizable: bool = PrivateAttr(default=True)

    #: Whether two distributions of this class that are equal in every field
    #: except ``name`` may share one built expression (see
    #: :meth:`structural_key`).  Classes that keep per-object state while
    #: their expression is built must set this to ``False``: an alias reuses
    #: the first object's tensors and never builds, so its own state would be
    #: missing.
    share_identical_expressions: ClassVar[bool] = True

    def structural_key(self) -> str:
        """Identity of this distribution's expression, independent of its name.

        Two distributions with the same key reference the same parameters,
        functions and constants in the same fields, so in one model they
        build identical expressions.  The key is the class name plus the
        JSON dump of every field except ``name``.

        Examples:
            Equal fields under different names give the same key:

            >>> from pyhs3.distributions import GaussianDist
            >>> a = GaussianDist(name="sig", x="m", mean="mu", sigma=1.0)
            >>> b = GaussianDist(name="bkg", x="m", mean="mu", sigma=1.0)
            >>> a.structural_key() == b.structural_key()
            True

            A different value, or a constant in place of a parameter
            reference, gives a different key:

            >>> c = GaussianDist(name="bkg", x="m", mean="mu", sigma=2.0)
            >>> d = GaussianDist(name="bkg", x="m", mean=0.0, sigma=1.0)
            >>> a.structural_key() in {c.structural_key(), d.structural_key()}
            False

        Returns:
            str: The structural key.
        """
        fields = self.model_dump(mode="json", exclude={"name"})
        return f"{type(self).__name__}:{json.dumps(fields, sort_keys=True)}"

    @abstractmethod
    def likelihood(self, context: Context) -> TensorVar:
        """
        Main probability model for the distribution.

        This is the core probability density function (PDF) for the distribution.
        For example, the Poisson probability for observed data, Gaussian PDF, etc.
        Must be implemented by all subclasses.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Main probability density

        Raises:
            TypeError: Must be implemented by subclasses
        """

    def log_likelihood(self, context: Context) -> TensorVar:
        """
        Log-space counterpart of :meth:`likelihood`.

        Default implementation takes the log of the probability-space
        likelihood. Override with an analytic log-space form wherever one
        exists (e.g. Gaussian, Poisson, LogNormal): the default round-trips
        through a probability value that can underflow to 0.0 for parameter
        points far from the mode, at which point ``pt.log(0.0)`` is ``-inf``
        even though the true log-density is finite.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Log of the main probability density
        """
        return cast(TensorVar, pt.log(self.likelihood(context)))

    def normalization_expression(
        self, _context: Context, _observable_name: str
    ) -> TensorVar | None:
        """
        Return the antiderivative expression, or None for numerical fallback.

        Override in subclasses to provide analytical normalization. The returned
        expression should be the antiderivative F(x) such that the integral
        ∫f(x)dx from a to b equals F(b) - F(a).

        Args:
            context: Mapping of names to pytensor variables
            observable_name: Name of the observable to integrate over

        Returns:
            Symbolic antiderivative expression, or None for numerical fallback.
        """
        return None

    def _normalization_integral(
        self, context: Context, obs_name: str, lower: TensorVar, upper: TensorVar
    ) -> TensorVar | None:
        """
        Evaluate normalization integral using the antiderivative expression.

        This is a private method that evaluates F(upper) - F(lower) where F is
        the antiderivative returned by normalization_expression().

        Args:
            context: Mapping of names to pytensor variables
            obs_name: Name of the observable to integrate over
            lower: Lower integration bound
            upper: Upper integration bound

        Returns:
            Symbolic integral expression, or None if normalization_expression() returns None.
        """
        expr = self.normalization_expression(context, obs_name)  # pylint: disable=assignment-from-none
        if expr is None:
            return None
        # Use the leaf (not the view) as the substitution target so the
        # replacement propagates through every ExpandDims(leaf) view in the
        # expression.
        leaf = context.parameters[obs_name]
        upper_t = pt.as_tensor_variable([upper], dtype=leaf.dtype)
        lower_t = pt.as_tensor_variable([lower], dtype=leaf.dtype)
        # substitute_variable rebuilds only the nodes downstream of x and
        # infers their types again from the (1,) bound, so the integral's
        # former event axis is statically length 1 (broadcastable,
        # https://github.com/scipp-atlas/pyhs3/issues/346), while the
        # antiderivative's parameter-only parts stay shared by both bounds.
        upper_val = substitute_variable(expr, leaf, upper_t)
        lower_val = substitute_variable(expr, leaf, lower_t)
        return cast(TensorVar, upper_val - lower_val)

    def unconstrained_parameters(self) -> list[str]:
        """Parameters this distribution scales a positive yield with but does not constrain.

        Such a parameter must be ``const`` in the parameter point an analysis
        uses, or it floats unconstrained. Only HistFactory channels have any
        (per-bin parameters of bins with zero uncertainty).
        """
        return []

    def _depends_on_observable(
        self, context: Context, observable_name: str, fields: Sequence[str]
    ) -> bool:
        """Whether any of ``fields`` is built from the observable ``observable_name``.

        An analytic antiderivative in ``x`` assumes the distribution's other
        parameters are constant in ``x``. When one of them is computed from the
        observable, :meth:`normalization_expression` should return None so the
        integral falls back to quadrature.

        Args:
            context: Mapping of names to pytensor variables.
            observable_name: Name of the observable being integrated over.
            fields: Names of this distribution's parameter fields to check.

        Returns:
            True if the observable's leaf is an ancestor of any of the fields.
        """
        leaf = context.parameters[observable_name]
        return bool(
            self._upstream_among(context, [self._parameters[f] for f in fields], {leaf})
        )

    @staticmethod
    def _upstream_among(
        context: Context,
        names: Iterable[str],
        candidates: set[TensorVar],
    ) -> set[TensorVar]:
        """
        The ``candidates`` found in the graphs of ``names``, the named variables included.

        Filtering while walking avoids materializing every ancestor of a large
        graph just to test a few variables for membership.

        Args:
            context: Mapping of names to pytensor variables.
            names: Names of the context entries whose graphs to walk.
            candidates: Variables to look for.

        Returns:
            The subset of ``candidates`` that is a named variable or one of its ancestors.
        """
        return cast(
            set[TensorVar],
            {
                variable
                for variable in ancestors([context[name] for name in names])
                if variable in candidates
            },
        )

    def _matching_observables(
        self, context: Context
    ) -> list[tuple[str, TensorVar, TensorVar]]:
        """
        Observable bounds from context that this distribution depends on.

        An observable matches when it is one of the distribution's parameters or
        when its leaf is an ancestor of a parameter's tensor, so an observable
        that reaches the density only through a function still counts. The
        ancestors are collected once for all observables.

        Shared by _apply_normalization() and log_expression() so both agree on
        whether any normalization applies without duplicating the matching logic.

        Args:
            context: Mapping of names to pytensor variables (includes observables)

        Returns:
            List of (name, lower, upper) for each observable this distribution depends on.
        """
        leaves = context.parameters
        direct = [
            (name, bounds)
            for name, bounds in context.observables.items()
            if name in self.parameters
        ]
        # Observables of data this graph never builds have no leaf.
        indirect = {
            leaves[name]: (name, bounds)
            for name, bounds in context.observables.items()
            if name not in self.parameters and name in leaves
        }
        if indirect:
            found = self._upstream_among(context, self.parameters, set(indirect))
            direct += [indirect[leaf] for leaf in found]
        return [(name, lower, upper) for name, (lower, upper) in direct]

    def _normalization_integral_for(
        self,
        raw: TensorVar,
        context: Context,
        matching: list[tuple[str, TensorVar, TensorVar]],
    ) -> TensorVar:
        """
        Compute the normalization integral for a non-empty set of matching observables.

        Attempts analytical integration first via _normalization_integral(),
        then falls back to nested Gauss-Legendre quadrature.

        Args:
            raw: Raw (unnormalized) likelihood expression, used only by the
                Gauss-Legendre quadrature fallback (the analytical branch
                integrates the antiderivative directly and does not need it).
            context: Mapping of names to pytensor variables (includes observables)
            matching: Non-empty list of (name, lower, upper), as returned by
                _matching_observables().

        Returns:
            Symbolic normalization integral.

        Raises:
            NotImplementedError: If more than one observable matches (multi-dimensional
                normalization is not yet supported).
        """
        # Single observable: try analytical integral first
        if len(matching) == 1:
            obs_name, lower, upper = matching[0]
            integral = self._normalization_integral(context, obs_name, lower, upper)
            if integral is not None:
                return integral

        if len(matching) > 1:
            obs_names = [name for name, _, _ in matching]
            msg = (
                f"Multi-dimensional normalization is not yet supported "
                f"(observables: {obs_names}). "
                f"See https://github.com/scipp-atlas/pyhs3/issues/214. "
                f"A distribution that depends on an observable only through "
                f"a function is also affected: whether it is normalized over "
                f"that observable (joint) or conditioned on it is undecided, "
                f"see https://github.com/scipp-atlas/pyhs3/issues/396"
            )
            raise NotImplementedError(msg)

        # Single observable: fall back to Gauss-Legendre quadrature.
        # Pass the leaf (not the view) so graph_replace substitutes through
        # every ExpandDims(leaf) view inside the integrand.
        obs_name, lower, upper = matching[0]
        return gauss_legendre_integral(raw, context.parameters[obs_name], lower, upper)

    def _apply_normalization(
        self,
        raw: TensorVar,
        context: Context,
    ) -> TensorVar:
        """
        Apply normalization to a raw likelihood expression.

        Normalizes a likelihood over observables present in the context.
        For a single matching observable, attempts analytical integration
        first via _normalization_integral(), then falls back to Gauss-Legendre
        quadrature. Multi-dimensional normalization is not yet supported and
        raises NotImplementedError.

        Args:
            raw: Raw (unnormalized) likelihood expression
            context: Mapping of names to pytensor variables (includes observables)

        Returns:
            Normalized likelihood expression
        """
        # Explicit opt-out for distributions that should not be normalized
        if not self._normalizable:
            return raw

        matching = self._matching_observables(context)
        if not matching:
            return raw

        integral = self._normalization_integral_for(raw, context, matching)
        return cast(TensorVar, raw / integral)

    def _expression(self, context: Context) -> TensorVar:
        """
        Complete probability combining main likelihood with extended terms.

        Returns the product of likelihood() and extended_likelihood().
        This provides the complete probability for the distribution.

        All distributions are automatically normalized over observables present
        in the context, unless explicitly opted out via _normalizable = False.

        Subclasses typically do not need to override this method - just
        implement likelihood() and optionally extended_likelihood().

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Complete probability density
        """
        raw = self.likelihood(context)

        # Apply normalization (respects _normalizable flag internally)
        raw = self._apply_normalization(raw, context)

        return cast(TensorVar, raw * self.extended_likelihood(context))

    def log_expression(self, context: Context) -> TensorVar:
        """
        Log-probability combining main likelihood with extended terms.

        Returns log_likelihood() plus log(extended_likelihood()), minus the log
        of the normalization integral where normalization applies. This is
        mathematically equivalent to log(likelihood * extended_likelihood) but
        starts from log_likelihood() so that a distribution's analytic log form
        (e.g. Gaussian, Poisson, LogNormal) is used directly instead of round-
        tripping through a probability value that can underflow to 0.0.

        Normalization itself stays probability-space: the normalization
        integral (whether analytic or Gauss-Legendre quadrature) is a sum of
        positive terms and is far less underflow-prone than the raw density,
        so log(integral) is subtracted rather than computed as its own
        analytic log form.

        All distributions are automatically normalized over observables present
        in the context, unless explicitly opted out via _normalizable = False.

        PyTensor handles optimization and simplification automatically.

        Args:
            context: Mapping of names to pytensor variables

        Returns:
            TensorVar: Log-probability density
        """
        log_raw = self.log_likelihood(context)

        # Apply normalization (respects _normalizable flag internally). Only
        # compute the raw (probability-space) likelihood when normalization
        # actually applies -- it's needed for the Gauss-Legendre fallback.
        if self._normalizable:
            matching = self._matching_observables(context)
            if matching:
                raw = self.likelihood(context)
                integral = self._normalization_integral_for(raw, context, matching)
                log_raw = log_raw - pt.log(integral)

        return cast(
            TensorVar,
            log_raw + pt.log(self.extended_likelihood(context)),
        )

    def extended_likelihood(
        self, _context: Context, _data: TensorVar | None = None
    ) -> TensorVar:
        """
        Extended likelihood contribution in normal space.

        Returns additional likelihood terms for extended ML fitting.
        Override only when the extended terms belong in the distribution's
        per-event density, like constraint terms (HistFactory).  Terms that
        enter the likelihood once per channel (e.g. the Poisson yield term
        of an extended MixtureDist, which involves the observed event count)
        must not use this hook: ``_expression()`` multiplies the result into
        the density, so ``Model.log_prob`` would count it once per event when
        summing over data.  Such terms are assembled once per channel by
        ``Model.log_prob``, which owns the channel-dataset pairing.

        Default: no contribution (returns 1.0 in normal space).

        Args:
            context: Mapping of names to pytensor variables
            data: Optional data tensor for data-dependent terms

        Returns:
            TensorVar: Likelihood contribution (default: 1.0 = no contribution)
        """
        return pt.constant(1.0)

    def log_prob_terms(
        self,
        _expressions: Mapping[str, TensorVar],
        log_expressions: Mapping[str, TensorVar],
        _distributions: Distributions,
    ) -> LogProbTerms:
        """
        Structured contributions of this distribution to the joint log-likelihood.

        Called by :attr:`pyhs3.Model.log_prob` for each channel after the
        model graph is built.  Override when the distribution's terms do not
        all enter the likelihood as a per-event log-density — e.g. once-per-
        channel yield terms (extended :class:`~pyhs3.distributions.MixtureDist`)
        or globally-deduplicated constraint factors
        (:class:`~pyhs3.distributions.ProductDist`).

        Default: a single per-event ``log(PDF)`` term sourced from the
        pre-built log-space expression so it stays finite where the
        probability-space PDF underflows.

        Args:
            _expressions: Compiled symbolic expressions for all distributions,
                keyed by name (``model.distributions``).  Unused in the base
                implementation; subclasses may reference it.
            log_expressions: Compiled log-space symbolic expressions for all
                distributions, keyed by name (``model.log_distributions``).
            _distributions: Distribution objects keyed by name, so composite
                distributions can delegate to their components' hooks.

        Returns:
            LogProbTerms: per-event, per-channel, and constraint contributions.
        """
        return LogProbTerms(per_event=[log_expressions[self.name]])


class SampleableDistribution(Distribution, ABC):
    """
    A distribution type that can draw values of one of its fields.

    :meth:`pyhs3.Model.sample_data` draws the global observables of a
    likelihood's constraint terms with it: the field that holds the global
    observable is drawn from the distribution given the values of the other
    fields at the chosen parameter point.  Drawing happens in two steps, so
    that one compiled function evaluates the parameters of every constraint
    term: :meth:`sampling_parameters` returns them as tensors, and
    :meth:`sample` draws from their values.
    """

    #: Fields whose value :meth:`sample` can draw given the others, in the
    #: order a constraint term's global observable is looked for.
    sampled_fields: ClassVar[tuple[str, ...]] = ("x",)

    def field_tensor(self, context: Context, field_name: str) -> TensorVar:
        """The tensor ``context`` holds for one of this distribution's fields."""
        return context[self._parameters[field_name]]

    def _check_sampled_field(self, field_name: str) -> None:
        """Raise if ``field_name`` is not one of :attr:`sampled_fields`."""
        if field_name not in self.sampled_fields:
            msg = (
                f"{type(self).__name__} draws only {list(self.sampled_fields)}, "
                f"not '{field_name}'."
            )
            raise ValueError(msg)

    @abstractmethod
    def sampling_parameters(
        self, context: Context, field_name: str
    ) -> dict[str, TensorVar]:
        """Parameters of the distribution of ``field_name`` given the other fields.

        Args:
            context: Mapping of names to PyTensor variables.
            field_name: One of :attr:`sampled_fields`.

        Returns:
            dict[str, TensorVar]: Tensors keyed by the names :meth:`sample`
            reads, each a scalar or of the shape of the drawn field.

        Raises:
            ValueError: If ``field_name`` is not one of :attr:`sampled_fields`.
        """

    @abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        parameters: Mapping[str, npt.NDArray[np.float64]],
        size: tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        """Draw values of the sampled field.

        Args:
            rng: Source of the random draws; the draws advance its state.
            parameters: Values of the tensors of :meth:`sampling_parameters`,
                each broadcastable to ``size``.
            size: Shape of the result, e.g. ``(n, B)`` for ``n`` draws of a
                field of shape ``(B,)``.

        Returns:
            numpy.ndarray: float64 draws of shape ``size``.
        """


class BatchableDistribution(Distribution, ABC):
    """
    A distribution type whose instances can be evaluated together, as one family.

    The distribution counterpart of
    :class:`~pyhs3.functions.BatchableFunction`: :class:`~pyhs3.Model` groups
    the instances of a batchable type that feed the same root distribution into
    one family node and builds all of them with a single call to
    :meth:`batch_log_expressions`.

    Inheriting this class asserts that the type's ``log_likelihood``,
    ``likelihood`` and ``normalization_expression`` are elementwise in every
    parameter: evaluated with each parameter replaced by a stack of the
    members' values along a new leading axis, they return the members' results
    stacked along that axis.  The default :meth:`batch_log_expressions` relies
    on that and needs no code in the subclass.

    No built-in distribution inherits this class: on large workspaces,
    batching them made the cold C compile slower (stacked parameter math needs
    C modules of its own) for gains only on the JAX backend.  See
    ``docs/explanation/batched_families.rst`` before opting a type in.
    """

    @classmethod
    def batch_log_expressions(
        cls, members: Sequence[Self], context: Context
    ) -> dict[str, TensorVar]:
        """
        Build every member's log-space expression with one evaluation of the class's math.

        Fields that name the same input for every member (typically the
        observable) are kept as they are.  Every other field is replaced by a
        stack of the members' scalar inputs placed on the parameter-batch
        axis: an observable enters as an ``(N, 1)`` view whose last axis is
        that batch axis (length 1 for scalar parameters), so the stack has
        shape ``(1, K)`` for ``K`` members and the log density comes out
        ``(N, K)``.  Normalization works unchanged, since Gauss-Legendre
        quadrature also sums over axis 0.  Members without shared tensors (a
        constraint on a parameter) stack to ``(K,)``.  One
        :meth:`log_expression` of a member whose fields name those stacks,
        normalization included, is then split along the last axis into one
        expression per member with the member's own shape: ``(N, 1)``, or a
        scalar.

        Members are first split by the non-scalar inputs their fields name
        (for example, shapes on one observable and constraints on parameters
        are batched separately), and each such group is batched on its own;
        a group of one member is simply that member's
        :meth:`log_expression`.  If a group's fields are list-valued or a
        shared input is not ``(N, 1)``-like (its last axis already batches
        parameters), each of its members is built with its own
        :meth:`log_expression` and the reason is logged at debug level.

        Args:
            members: Instances of this class with the same observables.
            context: Built tensors by name.

        Returns:
            dict[str, TensorVar]: Log-space expression per member name, equal
            to that member's ``log_expression(context)`` up to rounding.
        """
        groups: dict[tuple[tuple[str, str | None], ...], list[Self]] = {}
        for member in members:
            signature = tuple(
                (key, name if context[name].ndim != 0 else None)
                for key, name in member._parameters.items()  # pylint: disable=protected-access
            )
            groups.setdefault(signature, []).append(member)
        outputs: dict[str, TensorVar] = {}
        for group in groups.values():
            if len(group) > 1:
                outputs.update(cls._batch_log_expressions_of_one_shape(group, context))
                continue
            if len(members) > 1:
                log.debug(
                    "%s '%s' is built on its own, not batched: no other member "
                    "uses the same non-scalar inputs",
                    cls.__name__,
                    group[0].name,
                )
            outputs[group[0].name] = group[0].log_expression(context)
        return outputs

    @classmethod
    def _batch_log_expressions_of_one_shape(
        cls, members: Sequence[Self], context: Context
    ) -> dict[str, TensorVar]:
        """Batch members whose fields name the same non-scalar inputs (see :meth:`batch_log_expressions`)."""
        # batch_log_expressions groups members by their fields and the names of
        # their non-scalar inputs, so every member has the same keys, and a
        # field whose input names differ between members is scalar for each.
        first = members[0]
        keys = list(first._parameters)  # pylint: disable=protected-access
        reason = None
        if any("[" in key for key in keys):
            reason = "it has list-valued fields"
        differing: dict[str, list[str]] = {}
        shared_ndim = 0
        if reason is None:
            for key in keys:
                names = [m._parameters[key] for m in members]  # pylint: disable=protected-access
                if len(set(names)) == 1:
                    tensor = context[names[0]]
                    if tensor.ndim == 1 or (
                        tensor.ndim > 1 and not tensor.type.broadcastable[-1]
                    ):
                        reason = f"its shared '{key}' input has no length-1 batch axis"
                        break
                    shared_ndim = max(shared_ndim, tensor.ndim)
                else:
                    differing[key] = names
        if reason is not None:
            for member in members:
                log.debug(
                    "%s '%s' is built on its own, not batched: %s",
                    cls.__name__,
                    member.name,
                    reason,
                )
            return {m.name: m.log_expression(context) for m in members}
        if not differing:
            # Every field names the same inputs: the members are one expression.
            shared = first.log_expression(context)
            return {m.name: shared for m in members}

        batched_context = context.copy()
        fields = first.model_dump()
        for key, names in differing.items():
            placeholder = f"__family__{first.name}__{key}"
            stacked = pt.stack([context[name] for name in names])  # (K,)
            # (1, ..., 1, K): the members ride on the shared inputs' last
            # (parameter-batch) axis; no extra axis when nothing is shared.
            stacked = stacked.dimshuffle(*(["x"] * max(shared_ndim - 1, 0)), 0)
            batched_context.add_parameter(placeholder, cast(TensorVar, stacked))
            fields[key] = placeholder
        fields["name"] = f"__family__{first.name}"
        batched = type(first).model_validate(fields).log_expression(batched_context)
        k = len(members)
        last = batched.ndim - 1
        pieces = (
            pt.split(batched, [1] * k, n_splits=k, axis=last)  # type: ignore[no-untyped-call]
            if k > 1
            else [batched]
        )
        outputs = {}
        for member, piece in zip(members, pieces, strict=True):
            # Restore the member's own shape: (N, 1) keeps its length-1 batch
            # axis; a scalar member drops it.
            fixed = specify_shape(piece, (None,) * last + (1,))
            outputs[member.name] = cast(
                TensorVar,
                fixed.dimshuffle(tuple(range(last))) if shared_ndim == 0 else fixed,
            )
        return outputs
