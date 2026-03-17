from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pyhs3.analyses import Analyses, Analysis
from pyhs3.data import Data, DataType, PointData
from pyhs3.distributions import Distributions, DistributionType
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
        See :hs3:label:`HS3 file format specification <hs3.file-format>` for the complete workspace structure.
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

    @staticmethod
    def _get_root_version_hint(spec_dict: dict[str, Any]) -> str | None:
        """
        Extract ROOT version hint from workspace spec_dict metadata.

        Args:
            spec_dict: The workspace specification dictionary

        Returns:
            Hint string if ROOT version is older than required, None otherwise
        """
        try:
            metadata = Metadata.model_validate(spec_dict.get("metadata", {}))
            return metadata.root_version_hint()
        except ValidationError:
            return None

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
            hint = cls._get_root_version_hint(spec_dict)
            if hint:
                error_summary += f"\n\n{hint}"

            if suppress_traceback:
                sys.tracebacklimit = 0
            raise WorkspaceValidationError(error_summary) from None
        except WorkspaceValidationError as e:
            hint = cls._get_root_version_hint(spec_dict)
            if hint:
                error_with_hint = f"{e}\n\n{hint}"
                raise WorkspaceValidationError(error_with_hint) from e
            raise

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
        Extract observable names and bounds from likelihoods + data + domain.

        Walks likelihoods to find distribution-data pairings. For each dataset axis,
        gets bounds from the data axis itself (axis.min/max). Propagates observable
        info through composite distributions (MixtureDist, ProductDist).

        Returns:
            Dictionary mapping observable names to (min, max) tuples
        """
        observables: dict[str, tuple[float, float]] = {}

        if not self.likelihoods or not self.data:
            return observables

        # For each likelihood, extract observable axes from paired data
        for likelihood in self.likelihoods:
            for data_item in likelihood.data:
                # FK resolution guarantees data items are resolved objects
                datum = (
                    data_item
                    if not isinstance(data_item, str)
                    else self.data[data_item]
                )

                # PointData axes are optional; UnbinnedData/BinnedData always have them
                if isinstance(datum, PointData) and datum.axes is None:
                    log.warning(
                        "The likelihood '%s' references a PointData '%s' without axes. This cannot be used to normalize any distribution.",
                        likelihood.name,
                        datum.name,
                    )
                    continue

                # For each axis, extract bounds
                for axis in datum.axes or []:
                    observables[axis.name] = (axis.min, axis.max)

        return observables

    def model(
        self,
        *,
        domain: int | str | Domain = 0,
        parameter_set: int | str | ParameterSet = 0,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ) -> Model:
        """
        Constructs a `Model` object using the provided domain and parameter set.

        Args:
            domain (int | str | Domain): Identifier or object specifying the domain to use.
            parameter_set (int | str | ParameterSet): Identifier or object specifying the parameter values to use.
            progress (bool): Whether to show progress bar during dependency graph construction. Defaults to True.
            mode (str): PyTensor compilation mode. Defaults to "FAST_RUN".
                       Options: "FAST_RUN" (apply all rewrites, use C implementations),
                       "FAST_COMPILE" (few rewrites, Python implementations),
                       "NUMBA" (compile using Numba), "JAX" (compile using JAX),
                       "PYTORCH" (compile using PyTorch), "DebugMode" (debugging),
                       "NanGuardMode" (NaN detection).

        Returns:
            Model: The constructed model object.
        """

        selected_domain = (
            domain
            if isinstance(domain, Domain)
            else self.domains[domain]
            if self.domains
            else ProductDomain(name="default")
        )
        parameterset = (
            parameter_set
            if isinstance(parameter_set, ParameterSet)
            else self.parameter_points[parameter_set]
            if self.parameter_points
            else ParameterSet(name="default", parameters=[])
        )

        # Compute observables from likelihoods + data + domain
        observables = self._compute_observables()

        return Model(
            parameterset=parameterset or ParameterSet(name="default"),
            distributions=self.distributions or Distributions(),
            domain=selected_domain or Domain(name="default", type="unknown"),
            functions=self.functions or Functions(),
            progress=progress,
            mode=mode,
            observables=observables,
        )
