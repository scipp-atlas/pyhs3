from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterable
from functools import singledispatchmethod
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pyhs3.analyses import Analyses, Analysis
from pyhs3.data import Data, DataType
from pyhs3.distributions import Distributions, DistributionType, HistFactoryDistChannel
from pyhs3.distributions.histfactory.modifiers import (
    ParametersModifier,
    SingleParamConstraint,
)
from pyhs3.domains import Domain, Domains, DomainType, ProductDomain
from pyhs3.exceptions import WorkspaceValidationError
from pyhs3.functions import Functions
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.model import Model
from pyhs3.parameter_points import ParameterPoints, ParameterSet

log = logging.getLogger(__name__)


class Workspace(BaseModel):
    """
    Workspace for managing HS3 model specifications.

    A workspace contains parameter points, distributions, domains, and functions
    that define a probabilistic model. It provides methods to construct Model
    objects with specific parameter values and domain constraints.

    Attributes:
        metadata: Required metadata containing HS3 version and optional attribution
        distributions: List of distribution configurations
        functions: List of function configurations
        domains: List of domain configurations
        parameter_points: List of parameter point configurations
        data: Data specifications for observations
        likelihoods: Likelihood specifications mapping distributions to data
        analyses: Analysis configurations for automated analyses
        misc: Arbitrary user-created information
        parameter_collection (ParameterPoints): Named parameter sets.
        distribution_set (Distributions): Available distributions.
        domain_collection (Domains): Domain constraints for parameters.
        function_set (Functions): Available functions for parameter computation.

    HS3 Reference:
        See :external+hs3:ref:`HS3 file format specification <hs3.file-format>` for the complete workspace structure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required field
    metadata: Metadata

    # Optional fields using discriminated unions
    distributions: Distributions | None = Field(
        default_factory=lambda: Distributions([])
    )
    functions: Functions | None = Field(default_factory=lambda: Functions([]))
    domains: Domains | None = Field(default_factory=lambda: Domains([]))
    parameter_points: ParameterPoints | None = Field(
        default_factory=lambda: ParameterPoints([])
    )
    data: Data | None = Field(default_factory=lambda: Data([]))
    likelihoods: Likelihoods | None = Field(default_factory=lambda: Likelihoods([]))
    analyses: Analyses | None = Field(default_factory=lambda: Analyses([]))
    misc: dict[str, Any] | None = Field(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        """Resolve foreign key references after workspace construction."""
        self._resolve_foreign_keys()

    def _resolve_foreign_keys(self) -> None:
        """Resolve string references to actual objects with referential integrity checking."""
        errors: list[str] = []

        # Resolve Likelihood fields first (analyses reference likelihoods)
        if self.likelihoods is not None:
            for likelihood in self.likelihoods:
                self._resolve_likelihood_fields(likelihood, errors)

        # Validate observable axis uniqueness after FK resolution
        if self.likelihoods is not None:
            for likelihood in self.likelihoods:
                try:
                    likelihood.validate_unique_axis_names(self)
                except ValueError as exc:
                    errors.append(str(exc))

        # Validate named HFDC references independently of whether a channel is
        # selected by a likelihood. Invalid foreign keys must fail at import,
        # not turn into implicit free parameters in an unused subgraph.
        try:
            self._validate_all_named_hfdc_constraints()
        except ValueError as exc:
            errors.append(str(exc))

        # Validate internal per-bin constraint ownership across each likelihood.
        if self.likelihoods is not None:
            for likelihood in self.likelihoods:
                try:
                    self._validate_hfdc_constraints(likelihood)
                except ValueError as exc:
                    errors.append(str(exc))

        # Resolve Analysis fields
        if self.analyses is not None:
            for analysis in self.analyses:
                self._resolve_analysis_fields(analysis, errors)

        if errors:
            msg = "Workspace has unresolved references:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise WorkspaceValidationError(msg)

    def _resolve_fk_list(
        self,
        refs: Iterable[Any],
        collection: Distributions | Data | Domains,
        parent_label: str,
        entity_label: str,
        errors: list[str],
    ) -> list[Any]:
        """Resolve string references in a list against a named collection."""
        resolved: list[Any] = []
        for ref in refs:
            if isinstance(ref, str):
                obj = collection.get(ref)
                if obj is None:
                    errors.append(
                        f"{parent_label} references unknown {entity_label} '{ref}'"
                    )
                else:
                    resolved.append(obj)
            else:
                resolved.append(ref)
        return resolved

    def _constraint_dependency_closure(
        self, target_name: str
    ) -> tuple[set[str], list[str] | None]:
        """Return transitive named dependencies and a reachable cycle, if any.

        Distribution and function references share the workspace namespace in
        the runtime dependency graph. Unmatched names are leaf parameters (or
        generated constants) and are included in the closure without further
        traversal.
        """
        entities: dict[str, object] = {}
        if self.distributions is not None:
            entities.update({dist.name: dist for dist in self.distributions})
        if self.functions is not None:
            entities.update({function.name: function for function in self.functions})

        closure: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []
        cycle: list[str] | None = None

        def visit(name: str) -> None:
            nonlocal cycle
            if cycle is not None:
                return
            closure.add(name)
            entity = entities.get(name)
            if entity is None or name in visited:
                return
            if name in stack:
                start = stack.index(name)
                cycle = [*stack[start:], name]
                return
            stack.append(name)
            for dependency in entity.parameters:  # type: ignore[attr-defined]
                visit(dependency)
            stack.pop()
            visited.add(name)

        visit(target_name)
        closure.discard(target_name)
        return closure, cycle

    def _validate_named_hfdc_constraint(
        self,
        modifier: SingleParamConstraint,
        channel: HistFactoryDistChannel,
        where: str,
        observable_names: set[str],
        dependency_cache: dict[str, tuple[set[str], list[str] | None]],
    ) -> None:
        """Validate one named modifier constraint against workspace entities."""
        constraint_name = modifier.constraint
        if constraint_name is None:
            return
        target = (
            self.distributions.get(constraint_name)
            if self.distributions is not None
            else None
        )
        if target is None:
            msg = (
                f"'{where}' references unknown constraint distribution "
                f"'{constraint_name}'"
            )
            raise ValueError(msg)
        if isinstance(target, HistFactoryDistChannel):
            msg = (
                f"'{where}' constraint '{constraint_name}' must not reference "
                "a histfactory_dist"
            )
            raise ValueError(msg)

        if constraint_name not in dependency_cache:
            dependency_cache[constraint_name] = self._constraint_dependency_closure(
                constraint_name
            )
        dependencies, cycle = dependency_cache[constraint_name]
        if cycle is not None or channel.name in dependencies:
            path = " -> ".join(
                cycle or [constraint_name, channel.name, constraint_name]
            )
            msg = f"'{where}' constraint '{constraint_name}' creates a circular dependency: {path}"
            raise ValueError(msg)

        # A composite/function-mediated target that reaches any HistFactory
        # channel is not an auxiliary constraint distribution either.
        if self.distributions is not None:
            hfdc_dependencies = sorted(
                name
                for name in dependencies
                if isinstance(self.distributions.get(name), HistFactoryDistChannel)
            )
            if hfdc_dependencies:
                msg = (
                    f"'{where}' constraint '{constraint_name}' depends on "
                    f"histfactory_dist '{hfdc_dependencies[0]}'"
                )
                raise ValueError(msg)

        if modifier.parameter not in dependencies:
            msg = (
                f"'{where}' constraint '{constraint_name}' does not depend on "
                f"modifier parameter '{modifier.parameter}'"
            )
            raise ValueError(msg)

        event_dependencies = sorted(dependencies & observable_names)
        if event_dependencies:
            msg = (
                f"'{where}' constraint '{constraint_name}' depends on event "
                f"observable(s): {', '.join(event_dependencies)}"
            )
            raise ValueError(msg)

    def _validate_all_named_hfdc_constraints(self) -> None:
        """Validate every named HistFactory reference in the workspace."""
        if self.distributions is None:
            return
        observable_names = {
            axis.name for datum in (self.data or []) for axis in (datum.axes or [])
        }
        dependency_cache: dict[str, tuple[set[str], list[str] | None]] = {}
        for channel in self.distributions:
            if not isinstance(channel, HistFactoryDistChannel):
                continue
            for sample in channel.samples:
                for modifier in sample.modifiers:
                    if not isinstance(modifier, SingleParamConstraint):
                        continue
                    where = f"{channel.name}/{sample.name}/{modifier.name}"
                    self._validate_named_hfdc_constraint(
                        modifier,
                        channel,
                        where,
                        observable_names,
                        dependency_cache,
                    )

    def _validate_hfdc_constraints(self, likelihood: Likelihood) -> None:
        """Validate named and internal constraints for one likelihood.

        Rules enforced:
        - shapesys parameters must not be shared across channels (per-channel by design).
        - staterror parameters must not be shared across channels (same reason).

        Called after FK resolution so likelihood.distributions contains objects.
        """
        shapesys_owners: dict[str, str] = {}
        staterror_owners: dict[str, str] = {}
        likelihood_distributions = [
            *likelihood.distributions,
            *(likelihood.aux_distributions or []),
        ]
        for dist_obj in likelihood_distributions:
            if isinstance(dist_obj, str) or not isinstance(
                dist_obj, HistFactoryDistChannel
            ):
                continue
            for sample in dist_obj.samples:
                for modifier in sample.modifiers:
                    if isinstance(modifier, SingleParamConstraint):
                        continue
                    multi_modifier = cast(ParametersModifier, modifier)
                    owners = (
                        shapesys_owners
                        if multi_modifier.type == "shapesys"
                        else staterror_owners
                    )
                    for param in multi_modifier.parameters:
                        if param in owners and owners[param] != dist_obj.name:
                            kind = multi_modifier.type
                            msg = (
                                f"{kind} parameter '{param}' appears in both "
                                f"'{owners[param]}' and '{dist_obj.name}'; "
                                f"{kind} is per-channel and may not be shared."
                            )
                            raise ValueError(msg)
                        owners[param] = dist_obj.name

    def _resolve_likelihood_fields(
        self, likelihood: Likelihood, errors: list[str]
    ) -> None:
        """Resolve foreign key fields on a Likelihood."""
        # Resolve distributions
        if self.distributions is not None:
            resolved = self._resolve_fk_list(
                likelihood.distributions,
                self.distributions,
                f"Likelihood '{likelihood.name}'",
                "distribution",
                errors,
            )
            likelihood.distributions = Distributions(
                cast(list[DistributionType], resolved)
            )

            if likelihood.aux_distributions is not None:
                resolved_aux = self._resolve_fk_list(
                    likelihood.aux_distributions,
                    self.distributions,
                    f"Likelihood '{likelihood.name}'",
                    "aux_distribution",
                    errors,
                )
                likelihood.aux_distributions = Distributions(
                    cast(list[DistributionType], resolved_aux)
                )
        else:
            errors.append(
                f"Likelihood '{likelihood.name}' references unknown distributions"
            )

        # Resolve data
        if self.data is not None:
            resolved = self._resolve_fk_list(
                likelihood.data,
                self.data,
                f"Likelihood '{likelihood.name}'",
                "data",
                errors,
            )
            likelihood.data = Data(cast(list[DataType], resolved))
        else:
            errors.append(f"Likelihood '{likelihood.name}' references unknown data")

    def _resolve_analysis_fields(self, analysis: Analysis, errors: list[str]) -> None:
        """Resolve foreign key fields on an Analysis."""
        # Resolve likelihood
        if self.likelihoods is not None:
            if isinstance(analysis.likelihood, str):
                lk = self.likelihoods.get(analysis.likelihood)
                if lk is None:
                    errors.append(
                        f"Analysis '{analysis.name}' references unknown likelihood '{analysis.likelihood}'"
                    )
                else:
                    analysis.likelihood = lk
        else:
            errors.append(
                f"Analysis '{analysis.name}' references unknown likelihood '{analysis.likelihood}'"
            )

        # Resolve domains
        if self.domains is not None:
            resolved = self._resolve_fk_list(
                analysis.domains,
                self.domains,
                f"Analysis '{analysis.name}'",
                "domain",
                errors,
            )
            analysis.domains = Domains(cast(list[DomainType], resolved))
        else:
            errors.append(f"Analysis '{analysis.name}' references unknown domains")

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        verbose: bool = False,
        suppress_traceback: bool = True,
    ) -> Workspace:
        """
        Load workspace from a JSON file.

        Args:
            path: Path to the JSON file containing the HS3 specification
            verbose: If True, show all errors. If False, show first 20 and summarize rest.
            suppress_traceback: If True, suppress traceback on validation errors (default True).

        Returns:
            Workspace: The loaded workspace instance
        """
        path_obj = Path(path)
        with path_obj.open("r", encoding="utf-8") as f:
            spec_dict = json.load(f)

        try:
            return cls(**spec_dict)
        except ValidationError as e:
            error_summary = cls._format_validation_error(e, path, verbose)

            if suppress_traceback:
                sys.tracebacklimit = 0
            raise WorkspaceValidationError(error_summary) from None

    @classmethod
    def _format_validation_error(
        cls,
        validation_error: ValidationError,
        path: str | os.PathLike[str],
        verbose: bool,
    ) -> str:
        """
        Format a ValidationError into a readable error summary.

        Args:
            validation_error: The ValidationError to format
            path: Path to the file that caused the error
            verbose: If True, show all errors. If False, show first 20 and summarize rest.

        Returns:
            Formatted error message string
        """
        errors = validation_error.errors()
        error_count = len(errors)
        error_types: Counter[str] = Counter()
        loc_errors: Counter[tuple[str, ...]] = Counter()

        for error in errors:
            error_types[error["type"]] += 1
            loc_errors[
                tuple("#" if isinstance(key, int) else key for key in error["loc"])
            ] += 1

        # Build error summary using list for efficient string concatenation
        parts = [
            f"Workspace validation failed with {error_count} errors from {path}\n",
            "\nError breakdown by type:\n",
        ]

        for error_type, count in error_types.most_common():
            parts.append(f"  {error_type}: {count}\n")

        parts.append("\nError breakdown by component:\n")
        for loc, count in loc_errors.most_common():
            loc_str = ".".join(loc)
            parts.append(f"  {loc_str}: {count}\n")

        # Show detailed errors with improved formatting
        errors_to_show = errors if verbose else errors[:20]
        parts.append(f"\nErrors for debugging ({'all' if verbose else 'first 20'}):\n")
        for i, error in enumerate(errors_to_show):
            # Format location more readably using list comprehension
            loc_parts = [
                f"[{part}]" if isinstance(part, int) else str(part)
                for part in error.get("loc", [])
            ]

            # Build readable location string
            if not loc_parts:
                readable_loc = ""
            else:
                readable_loc = loc_parts[0]
                for part in loc_parts[1:]:
                    if part.startswith("["):
                        readable_loc += part  # Index directly follows
                    else:
                        readable_loc += f" -> {part}"

            # Add name from input if available

            # Add name from input if available
            input_data: Any = error.get("input", {})
            if isinstance(input_data, dict) and "name" in input_data:
                name = input_data["name"]
                if readable_loc and not readable_loc.endswith("]"):
                    readable_loc += f"('{name}')"

            msg = error.get("msg", "Unknown error")
            parts.append(f"  {i + 1}. {readable_loc}: {msg}\n")

        if not verbose and error_count > 20:
            parts.append(
                f"  ... and {error_count - 20} more errors (use verbose=True to see all)\n"
            )

        return "".join(parts)

    def _compute_observables(self) -> dict[str, tuple[float, float]]:
        """
        Resolve observables for model paths built without an explicit likelihood.

        Observables are a per-likelihood concept: an axis is only an observable
        of the data actually loaded by a given likelihood, and every analysis
        references exactly one likelihood, so axes from different likelihoods
        are never combined in one calculation. The legacy ``model(int)`` and
        name-fallback ``model(str)`` paths have no likelihood to scope to, so
        observables are resolved from each likelihood independently (via
        :meth:`_extract_observables`, the same resolution the per-likelihood
        ``model(Likelihood)`` / ``model(Analysis)`` paths use) and only used
        when every likelihood implies the same result.

        Returns:
            Dictionary mapping observable names to (min, max) tuples; empty
            when the workspace has no likelihoods.

        Raises:
            ValueError: If the workspace's likelihoods imply different
                observables, since a model built without a likelihood has no
                principled way to choose between them.
        """
        if not self.likelihoods:
            return {}

        first = self.likelihoods[0]
        observables = self._extract_observables(first)
        for likelihood in self.likelihoods:
            candidate = self._extract_observables(likelihood)
            if candidate != observables:
                msg = (
                    f"Cannot determine observables for a model built without a "
                    f"likelihood: likelihood '{first.name}' implies {observables} "
                    f"but likelihood '{likelihood.name}' implies {candidate}. "
                    f"Observables are resolved per likelihood; build the model "
                    f"from a specific likelihood instead, e.g. "
                    f"workspace.model(workspace.likelihoods[{likelihood.name!r}])."
                )
                raise ValueError(msg)
        return observables

    @staticmethod
    def _extract_observables(likelihood: Likelihood) -> dict[str, tuple[float, float]]:
        """Return {axis_name: (min, max)} for all data axes in a likelihood."""
        return {
            axis.name: (axis.min, axis.max)
            for datum in likelihood.data
            if not isinstance(datum, str)
            for axis in datum.axes or []
        }

    def _select_parameterset(
        self,
        parameter_set: int | str | ParameterSet | None,
        *,
        base_parameter_set: int | str | ParameterSet | None = None,
        fallback_first: bool = True,
    ) -> ParameterSet:
        """Resolve *parameter_set* to a :class:`~pyhs3.parameter_points.ParameterSet`.

        Args:
            parameter_set: Explicit override -- a ``ParameterSet`` instance, an int
                or str index into ``self.parameter_points``, or ``None`` to fall back.
            base_parameter_set: Optional complete parameter set on which the
                selected set is overlaid. This makes partial-snapshot handling
                explicit instead of assigning implicit inheritance semantics to
                HS3 parameter points.
            fallback_first: When *parameter_set* is ``None`` and ``fallback_first``
                is ``True`` (the default), fall back to ``parameter_points[0]``.
                Pass ``False`` to return an empty default instead (used by the
                ``Analysis`` path which manages its own ``init`` fallback).
        """

        def resolve(
            reference: int | str | ParameterSet,
            argument_name: str,
        ) -> ParameterSet:
            if isinstance(reference, ParameterSet):
                return reference
            if not self.parameter_points:
                msg = (
                    f"{argument_name}={reference!r} was requested but no "
                    "parameter_points are available in this workspace"
                )
                raise ValueError(msg)
            return self.parameter_points[reference]

        selected: ParameterSet | None = None
        if parameter_set is not None:
            selected = resolve(parameter_set, "parameter_set")
        elif fallback_first and self.parameter_points:
            selected = self.parameter_points[0]

        if base_parameter_set is None:
            return (
                selected
                if selected is not None
                else ParameterSet(name="default", parameters=[])
            )

        base = resolve(base_parameter_set, "base_parameter_set")
        if selected is None or selected is base:
            return base
        return base.overlay(selected)

    def _select_domain(
        self,
        domain: int | str | Domain | None,
        default_index: int | str | None = None,
    ) -> Domain:
        """Resolve *domain* to a :class:`~pyhs3.domains.Domain`.

        Args:
            domain: Explicit override (a ``Domain`` instance, int, or str key) or
                ``None`` to use *default_index*.
            default_index: Index/key to use when *domain* is ``None``. If both are
                ``None``, ``default_domain`` is preferred when present before
                falling back to the first domain. If no domain collection exists,
                returns a default ``ProductDomain``.
        """
        if isinstance(domain, Domain):
            return domain
        if domain is not None:
            if not self.domains:
                msg = f"domain={domain!r} was requested but no domains are available in this workspace"
                raise ValueError(msg)
            return self.domains[domain]
        if default_index is not None and self.domains:
            return self.domains[default_index]
        if self.domains:
            default_domain = self.domains.get("default_domain")
            if default_domain is not None:
                return default_domain
            return self.domains[0]
        return ProductDomain(name="default")

    @singledispatchmethod
    def model(
        self,
        target: int,
        *,
        domain: int | str | Domain | None = None,
        parameter_set: int | str | ParameterSet | None = None,
        base_parameter_set: int | str | ParameterSet | None = None,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ) -> Model:
        """
        Constructs a :class:`~pyhs3.model.Model` rooted at ``target``.

        Dispatch is based on the type of ``target``:

        - :class:`~pyhs3.analyses.Analysis` — all context (domain, parameter
          set, observables) is derived from the analysis; gains access to
          :attr:`~pyhs3.model.Model.log_prob`, :attr:`~pyhs3.model.Model.data`,
          and :attr:`~pyhs3.model.Model.free_params`. Only the likelihood's
          distributions and their transitive dependencies are built.
        - :class:`~pyhs3.likelihoods.Likelihood` — observable bounds are derived
          from the likelihood's data; ``domain`` falls back to ``default_domain``
          then index 0, and ``parameter_set`` falls back to workspace defaults
          unless overridden. Only the likelihood's distributions and their
          transitive dependencies are built.
        - ``str`` — searches analyses then likelihoods by name; delegates to the
          appropriate registered path.  Falls back to legacy domain indexing if
          the name is not found in either.
        - ``int`` — legacy path: ``target`` indexes into workspace domains and
          the complete workspace graph is built.

        Args:
            target: Dispatch key.  Pass an
                :class:`~pyhs3.analyses.Analysis` or
                :class:`~pyhs3.likelihoods.Likelihood` for the modern paths,
                a name string to search analyses/likelihoods, or an ``int``
                domain index for the legacy path.
            domain: Override domain (legacy and Likelihood paths only).
            parameter_set: Override parameter set. It may be partial when
                ``base_parameter_set`` is also supplied.
            base_parameter_set: Complete parameter set beneath ``parameter_set``.
                Parameters omitted by the override are inherited from this base.
            progress: Whether to show a progress bar during graph construction.
            mode: PyTensor compilation mode (default ``"FAST_RUN"``).

        Returns:
            :class:`~pyhs3.model.Model`: The constructed model.
        """
        # Legacy int path: target indexes into workspace domains.
        selected_domain = self._select_domain(domain, default_index=target)
        parameterset = self._select_parameterset(
            parameter_set, base_parameter_set=base_parameter_set
        )
        return Model(
            parameterset=parameterset,
            distributions=self.distributions or Distributions(),
            domain=selected_domain or Domain(name="default", type="unknown"),
            functions=self.functions or Functions(),
            progress=progress,
            mode=mode,
            observables=self._compute_observables(),
            likelihood=None,
        )

    @model.register
    def _(
        self,
        target: Analysis,
        *,
        parameter_set: int | str | ParameterSet | None = None,
        base_parameter_set: int | str | ParameterSet | None = None,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ) -> Model:
        # _resolve_foreign_keys guarantees both are resolved objects by construction.
        likelihood_obj = cast(Likelihood, target.likelihood)
        domains = cast(Domains, target.domains)

        if len(domains) == 1:
            analysis_domain: Domain = domains[0]
        else:
            # Merge all domain axes into one ProductDomain
            all_axes = [ax for d in domains for ax in getattr(d, "axes", [])]
            analysis_domain = ProductDomain(name=f"{target.name}_merged", axes=all_axes)  # type: ignore[arg-type]

        if target.init:
            if self.parameter_points is None:
                msg = f"Analysis '{target.name}' requires parameter set '{target.init}' but workspace has no parameter_points"
                raise ValueError(msg)
            param_set = self.parameter_points.get(target.init)
            if param_set is None:
                msg = f"Analysis '{target.name}' references unknown parameter set '{target.init}'"
                raise ValueError(msg)
        else:
            param_set = None

        # Explicit override takes priority; otherwise use analysis.init param_set or empty default.
        # Do NOT fall back to parameter_points[0] when neither init nor override was given —
        # that would silently impose workspace defaults that the caller did not request.
        selected_parameter_set = (
            parameter_set if parameter_set is not None else param_set
        )
        parameterset = self._select_parameterset(
            selected_parameter_set,
            base_parameter_set=base_parameter_set,
            fallback_first=False,
        )

        return Model(
            parameterset=parameterset,
            distributions=self.distributions or Distributions(),
            domain=analysis_domain,
            functions=self.functions or Functions(),
            progress=progress,
            mode=mode,
            observables=self._extract_observables(likelihood_obj),
            likelihood=likelihood_obj,
        )

    @model.register
    def _(
        self,
        target: Likelihood,
        *,
        domain: int | str | Domain | None = None,
        parameter_set: int | str | ParameterSet | None = None,
        base_parameter_set: int | str | ParameterSet | None = None,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ) -> Model:
        selected_domain = self._select_domain(domain)
        parameterset = self._select_parameterset(
            parameter_set, base_parameter_set=base_parameter_set
        )
        return Model(
            parameterset=parameterset,
            distributions=self.distributions or Distributions(),
            domain=selected_domain or Domain(name="default", type="unknown"),
            functions=self.functions or Functions(),
            progress=progress,
            mode=mode,
            observables=self._extract_observables(target),
            likelihood=target,
        )

    @model.register
    def _(
        self,
        target: str,
        *,
        domain: int | str | Domain | None = None,
        parameter_set: int | str | ParameterSet | None = None,
        base_parameter_set: int | str | ParameterSet | None = None,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ) -> Model:
        # Search analyses first, then likelihoods; fall back to legacy domain indexing.
        if self.analyses:
            analysis = self.analyses.get(target)
            if analysis is not None:
                if domain is not None:
                    msg = "domain override not supported when target resolves to an analysis"
                    raise ValueError(msg)
                return self.model(
                    analysis,
                    parameter_set=parameter_set,
                    base_parameter_set=base_parameter_set,
                    progress=progress,
                    mode=mode,
                )
        if self.likelihoods:
            likelihood = self.likelihoods.get(target)
            if likelihood is not None:
                return self.model(
                    likelihood,
                    domain=domain,
                    parameter_set=parameter_set,
                    base_parameter_set=base_parameter_set,
                    progress=progress,
                    mode=mode,
                )
        # Legacy fallback: treat target as a domain name.
        selected_domain = self._select_domain(domain, default_index=target)
        parameterset = self._select_parameterset(
            parameter_set, base_parameter_set=base_parameter_set
        )
        return Model(
            parameterset=parameterset,
            distributions=self.distributions or Distributions(),
            domain=selected_domain or Domain(name="default", type="unknown"),
            functions=self.functions or Functions(),
            progress=progress,
            mode=mode,
            observables=self._compute_observables(),
            likelihood=None,
        )
