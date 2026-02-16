from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pytensor.compile.function import function
from pytensor.graph.traversal import applys_between, explicit_graph_inputs
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pyhs3.analyses import Analyses, Analysis
from pyhs3.context import Context
from pyhs3.data import Data, DataType
from pyhs3.distributions import Distributions, DistributionType
from pyhs3.domains import Domain, Domains, DomainType, ProductDomain
from pyhs3.exceptions import WorkspaceValidationError
from pyhs3.functions import Functions
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.networks import build_dependency_graph
from pyhs3.parameter_points import ParameterPoints, ParameterSet
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)

TDefault = TypeVar("TDefault")

Axis: TypeAlias = tuple[float | None, float | None]


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
            raise ValueError(msg)

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

    def _resolve_analysis_fields(self, analysis: Analysis, errors: list[str]) -> None:
        """Resolve foreign key fields on an Analysis."""
        # Resolve likelihood
        if isinstance(analysis.likelihood, str) and self.likelihoods is not None:
            lk = self.likelihoods.get(analysis.likelihood)
            if lk is None:
                errors.append(
                    f"Analysis '{analysis.name}' references unknown likelihood '{analysis.likelihood}'"
                )
            else:
                analysis.likelihood = lk

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

        return Model(
            parameterset=parameterset or ParameterSet(name="default"),
            distributions=self.distributions or Distributions(),
            domain=selected_domain or Domain(name="default", type="unknown"),
            functions=self.functions or Functions(),
            progress=progress,
            mode=mode,
        )


class Model:
    """
    Probabilistic model with compiled tensor operations.

    A model represents a specific instantiation of a workspace with concrete
    parameter values and domain constraints. It builds symbolic computation
    graphs for distributions and functions, with optional compilation for
    performance optimization.

    The model handles dependency resolution between parameters, functions,
    and distributions, ensuring proper evaluation order through topological
    sorting of the computation graph.

    HS3 Reference:
        Models are computational representations of :hs3:label:`HS3 workspaces <hs3.file-format>`.
    """

    def __init__(
        self,
        *,
        parameterset: ParameterSet,
        distributions: Distributions,
        domain: Domain,
        functions: Functions,
        progress: bool = True,
        mode: str = "FAST_RUN",
    ):
        """
        Represents a probabilistic model composed of parameters, domains, distributions, and functions.

        Args:
            parameterset (ParameterSet): The parameter set used in the model.
            distributions (Distributions): Set of distributions to include.
            domain (Domain): Domain constraints for parameters.
            functions (Functions): Set of functions that compute parameter values.
            progress (bool): Whether to show progress bar during dependency graph construction.
            mode (str): PyTensor compilation mode. Defaults to "FAST_RUN".
                       Options: "FAST_RUN" (apply all rewrites, use C implementations),
                       "FAST_COMPILE" (few rewrites, Python implementations),
                       "NUMBA" (compile using Numba), "JAX" (compile using JAX),
                       "PYTORCH" (compile using PyTorch), "DebugMode" (debugging),
                       "NanGuardMode" (NaN detection).

        Attributes:
            domain (Domain): The original domain with constraints for parameters.
            parameterset (ParameterSet): The original parameter set with parameter values.
            distributions (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic distribution expressions.
            parameters (dict[str, pytensor.tensor.variable.TensorVariable]): Symbolic parameter variables.
            functions (dict[str, pytensor.tensor.variable.TensorVariable]): Computed function values.
            mode (str): PyTensor compilation mode.
            _compiled_functions (dict[str, Callable[..., npt.NDArray[np.float64]]]): Cache of compiled PyTensor functions.
        """
        self.parameterset = parameterset
        self.domain = domain
        self._distribution_objects = (
            distributions  # Store original distribution objects
        )
        self._function_objects = functions  # Store original function objects
        self.parameters: dict[str, TensorVar] = {}
        self.functions: dict[str, TensorVar] = {}
        self.distributions: dict[str, TensorVar] = {}
        self.modifiers: dict[str, TensorVar] = {}
        self.mode = mode
        self._compiled_functions: dict[str, Callable[..., npt.NDArray[np.float64]]] = {}
        self._compiled_inputs: dict[str, list[TensorVar]] = {}

        # Build dependency graph with proper entity identification
        self._build_dependency_graph(functions, distributions, progress)

    @staticmethod
    def _ensure_array(
        value: float | list[float] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Ensure a value is a numpy array with dtype float64.

        Converts scalars to 0-d arrays and lists to 1-d arrays.
        Existing numpy arrays are converted to float64 dtype if needed.

        Args:
            value: Input value (scalar, list, or array)

        Returns:
            NumPy array with dtype float64
        """
        return np.asarray(value, dtype=np.float64)

    def _build_dependency_graph(
        self,
        functions: Functions,
        distributions: Distributions,
        progress: bool = True,
    ) -> None:
        """
        Build and evaluate dependency graph for functions and distributions.

        This method properly handles cross-references between functions, distributions,
        and parameters by building a complete dependency graph first, then evaluating
        in topological order.
        """
        # Build dependency graph using the networks module
        graph, constants_map, modifiers_map = build_dependency_graph(
            self.parameterset, functions, distributions
        )

        # Get topological order (handles cycle detection internally)
        sorted_nodes = graph.topological_sort()

        # Evaluate nodes in topological order with optional progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", style="cyan"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
            transient=True,  # Progress bar disappears when finished
            disable=not progress,  # Disable progress bar if progress=False
        ) as progress_bar:
            task = progress_bar.add_task(
                "Building expressions...", total=len(sorted_nodes)
            )

            for node_idx in sorted_nodes:
                node_data = graph[node_idx]
                node_type: Literal[
                    "parameter", "constant", "function", "distribution", "modifier"
                ] = node_data["type"]
                node_name = node_data["name"]

                # Truncate long names to prevent jumpiness
                max_name_length = 60
                display_name = node_name
                if len(node_name) > max_name_length:
                    display_name = node_name[: max_name_length - 3] + "..."

                # Update progress description with current entity (fixed width)
                progress_bar.update(
                    task,
                    description=f"Building {node_type:<12}: {display_name:<{max_name_length}}",
                )

                # Build context with all currently available entities
                context_data = {
                    **self.parameters,
                    **self.functions,
                    **self.distributions,
                    **self.modifiers,
                }
                context = Context(parameters=context_data)

                if node_type == "parameter":
                    # Create parameter tensor with domain bounds applied
                    domain_bounds = (
                        self.domain.get(node_name, (None, None))
                        if self.domain
                        else (None, None)
                    )
                    param_point = (
                        self.parameterset.get(node_name) if self.parameterset else None
                    )
                    # Default to vector for observed data parameters, scalar for others
                    if param_point:
                        param_kind = param_point.kind
                    elif "_observed" in node_name:
                        param_kind = pt.vector
                    else:
                        param_kind = pt.scalar
                    self.parameters[node_name] = create_bounded_tensor(
                        node_name, domain_bounds, param_kind
                    )

                elif node_type == "constant":
                    # Constants are pre-created by distributions - add to parameters
                    self.parameters[node_name] = constants_map[node_name]

                elif node_type == "function":
                    # Functions are evaluated by design
                    self.functions[node_name] = functions[node_name].expression(context)

                elif node_type == "modifier":
                    # Modifiers are evaluated and stored for later use by distributions
                    # Use pre-built modifiers map for efficient O(1) lookup
                    self.modifiers[node_name] = modifiers_map[node_name].expression(
                        context
                    )

                else:  # node_type == "distribution"
                    # Distributions are evaluated by design
                    self.distributions[node_name] = distributions[node_name].expression(
                        context
                    )

                # Advance progress
                progress_bar.advance(task)

    def _get_compiled_function(
        self, name: str
    ) -> Callable[..., npt.NDArray[np.float64]]:
        """
        Get or create a compiled PyTensor function for the specified distribution.

        The distribution expression already includes both the main likelihood
        and extended likelihood terms, so no additional combination is needed.

        Args:
            name (str): Name of the distribution.

        Returns:
            Callable: Compiled PyTensor function.
        """
        if name not in self._compiled_functions:
            # Get the distribution expression (already includes extended_likelihood)
            dist_expression = self.distributions[name]

            inputs = [
                var
                for var in explicit_graph_inputs([dist_expression])
                if var.name is not None
            ]

            # Cache the inputs list for consistent ordering
            self._compiled_inputs[name] = cast(list[TensorVar], inputs)

            # Use the specified PyTensor mode
            compilation_mode = self.mode

            self._compiled_functions[name] = cast(
                Callable[..., npt.NDArray[np.float64]],
                function(
                    inputs=inputs,
                    outputs=dist_expression,
                    mode=compilation_mode,
                    on_unused_input="ignore",
                    name=name,
                    trust_input=True,
                ),
            )
        return self._compiled_functions[name]

    def pdf_unsafe(
        self,
        name: str,
        **parametervalues: float | list[float] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the PDF with automatic type conversion (convenience method).

        This method automatically converts parameter values to numpy arrays before
        evaluation. Use this for convenience in testing or interactive use.

        For performance-critical code, prefer :meth:`pdf` with pre-converted numpy arrays.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues: Values for each parameter (floats, lists, or arrays).

        Returns:
            npt.NDArray[np.float64]: The evaluated PDF value.

        See Also:
            :meth:`pdf`: Type-safe version requiring numpy arrays
            :meth:`logpdf_unsafe`: Log PDF with automatic type conversion

        Example:
            >>> model.pdf_unsafe("gauss", x=1.5, mu=0.0, sigma=1.0)  # floats ok  # doctest: +SKIP
            >>> model.pdf_unsafe("gauss", x=[1.5], mu=0.0, sigma=1.0)  # lists ok  # doctest: +SKIP
        """
        # Convert all parameter values to numpy arrays
        converted_params = {
            key: self._ensure_array(value) for key, value in parametervalues.items()
        }
        return self.pdf(name, **converted_params)

    def pdf(
        self, name: str, **parametervalues: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the probability density function of the specified distribution.

        This method requires all parameter values to be numpy arrays with dtype float64.
        For automatic type conversion, use :meth:`pdf_unsafe` instead.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues: Values for each parameter as numpy arrays.

        Returns:
            npt.NDArray[np.float64]: The evaluated PDF value.

        Raises:
            TypeError: If any parameter value is not a numpy array.

        See Also:
            :meth:`pdf_unsafe`: Convenience version with automatic type conversion
            :meth:`logpdf`: Log PDF with strict type checking

        Example:
            >>> import numpy as np
            >>> model.pdf("gauss", x=np.array(1.5), mu=np.array(0.0), sigma=np.array(1.0))  # doctest: +SKIP
        """
        # Use compiled function for better performance
        func = self._get_compiled_function(name)
        positional_values = self._reorder_params(name, parametervalues)
        return func(*positional_values)

    def logpdf_unsafe(
        self,
        name: str,
        **parametervalues: float | list[float] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the log PDF with automatic type conversion (convenience method).

        This method automatically converts parameter values to numpy arrays before
        evaluation. Use this for convenience in testing or interactive use.

        For performance-critical code, prefer :meth:`logpdf` with pre-converted numpy arrays.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues: Values for each parameter (floats, lists, or arrays).

        Returns:
            npt.NDArray[np.float64]: The log of the PDF.

        See Also:
            :meth:`logpdf`: Type-safe version requiring numpy arrays
            :meth:`pdf_unsafe`: PDF with automatic type conversion

        Example:
            >>> model.logpdf_unsafe("gauss", x=1.5, mu=0.0, sigma=1.0)  # floats ok  # doctest: +SKIP
        """
        return np.log(self.pdf_unsafe(name, **parametervalues))

    def logpdf(
        self, name: str, **parametervalues: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the natural logarithm of the PDF.

        This method requires all parameter values to be numpy arrays with dtype float64.
        For automatic type conversion, use :meth:`logpdf_unsafe` instead.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues: Values for each parameter as numpy arrays.

        Returns:
            npt.NDArray[np.float64]: The log of the PDF.

        Raises:
            TypeError: If any parameter value is not a numpy array.

        See Also:
            :meth:`logpdf_unsafe`: Convenience version with automatic type conversion
            :meth:`pdf`: PDF with strict type checking

        Example:
            >>> import numpy as np
            >>> model.logpdf("gauss", x=np.array(1.5), mu=np.array(0.0), sigma=np.array(1.0))  # doctest: +SKIP
        """
        return np.log(self.pdf(name, **parametervalues))

    def pars(self, name: str) -> list[str]:
        """
        Get the ordered list of input parameter names for a distribution.

        This method returns the parameter names in the exact order expected
        by the compiled PDF function. This is useful when you need to know
        the order of parameters for programmatic access.

        Args:
            name: Distribution name

        Returns:
            List of parameter names in the order expected by pdf()

        Example:
            >>> model.pars("model_singlechannel") # doctest: +SKIP
            ['uncorr_bkguncrt_1', 'uncorr_bkguncrt_0', 'model_singlechannel_observed', 'mu', 'Lumi']
        """
        if name not in self._compiled_inputs:
            # Trigger compilation to populate cache
            self._get_compiled_function(name)
        return [var.name for var in self._compiled_inputs[name] if var.name is not None]

    def parsort(self, name: str, names: list[str]) -> list[int]:
        """
        Similar to numpy's argsort, returns the indices that would sort the parameters.

        Args:
            name: Distribution name
            names: Parameter names to sort

        Returns:
            List of indices that would sort the parameters

        Example:
            >>> model.parsort("model_singlechannel", ["mu", "Lumi", "uncorr_bkguncrt_0", "uncorr_bkguncrt_1", "model_singlechannel_observed"]) # doctest: +SKIP
            [3, 2, 4, 0, 1]

        """
        return [names.index(par) for par in self.pars(name)]

    def _reorder_params(
        self,
        name: str,
        params: Mapping[str, npt.NDArray[np.float64]],
    ) -> list[npt.NDArray[np.float64]]:
        """
        Reorder parameters to match the expected input order for a distribution.

        Args:
            name: Distribution name
            params: Dictionary of parameter values (numpy arrays)

        Returns:
            List of values in the correct order for the compiled function
        """
        input_order = self.pars(name)
        return [params[param_name] for param_name in input_order]

    def visualize_graph(
        self,
        name: str,
        fmt: str = "svg",
        outfile: str | None = None,
        path: str | None = None,
    ) -> str:
        """
        Visualize the computation graph for a distribution.

        Args:
            name (str): Distribution name.
            fmt (str): Output format ('svg', 'png', 'pdf'). Defaults to 'svg'.
            outfile (str | None): Output filename. If None, uses '{name}_graph.{fmt}'.
            path (str | None): Directory path for output. If None, uses current working directory.

        Returns:
            str: Path to the generated visualization file.

        Raises:
            ImportError: If pydot is not installed.
        """
        try:
            from pytensor.printing import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
                pydotprint,
            )
        except ImportError as e:
            msg = "Graph visualization requires pydot. Install with: pip install pydot"
            raise ImportError(msg) from e

        if name not in self.distributions:
            msg = f"Distribution '{name}' not found in model"
            raise ValueError(msg)

        dist = self.distributions[name]

        if outfile is not None:
            filename = outfile
        else:
            base_filename = f"{name}_graph.{fmt}"
            if path is not None:
                filename = str(Path(path) / base_filename)
            else:
                filename = base_filename

        pydotprint(
            dist, outfile=filename, format=fmt, with_ids=True, high_contrast=True
        )
        return filename

    def __repr__(self) -> str:
        """Provide a concise overview of the model structure."""
        param_names = list(self.parameters.keys())
        dist_names = list(self.distributions.keys())
        func_names = list(self.functions.keys())

        param_display = ", ".join(param_names[:5]) + (
            "..." if len(param_names) > 5 else ""
        )
        dist_display = ", ".join(dist_names[:3]) + (
            "..." if len(dist_names) > 3 else ""
        )
        func_display = ", ".join(func_names[:3]) + (
            "..." if len(func_names) > 3 else ""
        )

        mode_status = self.mode

        return f"""Model(
    mode: {mode_status}
    parameters: {len(param_names)} ({param_display})
    distributions: {len(dist_names)} ({dist_display})
    functions: {len(func_names)} ({func_display})
)"""

    def graph_summary(self, name: str) -> str:
        """
        Get a summary of the computation graph structure.

        Args:
            name (str): Distribution name.

        Returns:
            str: Summary of the graph structure.
        """
        if name not in self.distributions:
            msg = f"Distribution '{name}' not found in model"
            raise ValueError(msg)

        dist = self.distributions[name]
        inputs = list(explicit_graph_inputs([dist]))

        # Count different types of operations
        applies = list(applys_between(inputs, [dist]))

        op_types: dict[str, int] = {}
        for apply in applies:
            op_name = type(apply.op).__name__
            op_types[op_name] = op_types.get(op_name, 0) + 1

        compile_info = f"\n    Mode: {self.mode}\n    Compiled: {'Yes' if self.mode != 'FAST_COMPILE' and name in self._compiled_functions else 'No'}"

        return f"""Distribution '{name}':
    Input variables: {len(inputs)}
    Graph operations: {len(applies)}
    Operation types: {dict(sorted(op_types.items()))}{compile_info}
"""


def create_bounded_tensor(
    name: str, domain: Axis, kind: Callable[..., TensorVar] = pt.scalar
) -> TensorVar:
    """
    Creates a tensor variable with optional domain constraints.

    Args:
        name: Name of the parameter.
        domain (tuple): Tuple specifying (min, max) range. Use None for unbounded sides.
                       For example: (0.0, None) for lower bound only, (None, 1.0) for upper bound only.
                       If both bounds are None, returns an unbounded tensor.
        kind: pt.scalar for scalars, pt.vector for vectors (default: pt.scalar).

    Returns:
        pytensor.tensor.variable.TensorVariable: The tensor variable, clipped to domain if bounds exist.

    Examples:
        >>> sigma = create_bounded_tensor("sigma", (0.0, None))  # sigma >= 0 (scalar)
        >>> fraction = create_bounded_tensor("fraction", (0.0, 1.0))  # 0 <= fraction <= 1 (scalar)
        >>> temperatures = create_bounded_tensor("temperatures", (None, 100.0), pt.vector)  # vector <= 100
        >>> unbounded = create_bounded_tensor("unbounded", (None, None))  # no bounds applied
    """
    min_bound, max_bound = domain

    # Create the base tensor
    tensor = kind(name)

    # If both bounds are None, return unbounded tensor
    if min_bound is None and max_bound is None:
        return tensor

    # Use infinity constants for unbounded sides
    min_val = pt.constant(-np.inf) if min_bound is None else pt.constant(min_bound)
    max_val = pt.constant(np.inf) if max_bound is None else pt.constant(max_bound)

    clipped = pt.clip(tensor, min_val, max_val)
    clipped.name = tensor.name  # Preserve the original name
    return cast(TensorVar, clipped)
