from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pytensor.compile.function import function
from pytensor.graph.traversal import applys_between, graph_inputs

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pyhs3.analyses import Analyses
from pyhs3.context import Context
from pyhs3.data import Data
from pyhs3.distributions import Distributions
from pyhs3.domains import Domain, Domains, ProductDomain
from pyhs3.exceptions import WorkspaceValidationError
from pyhs3.functions import Functions
from pyhs3.likelihoods import Likelihoods
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
        error_count = len(validation_error.errors())
        error_types: Counter[str] = Counter()
        loc_errors: Counter[tuple[str, ...]] = Counter()

        for error in validation_error.errors():
            error_types[error["type"]] += 1
            loc_errors[
                tuple("#" if isinstance(key, int) else key for key in error["loc"])
            ] += 1

        # Create a concise error message
        error_summary = (
            f"Workspace validation failed with {error_count} errors from {path}\n"
        )
        error_summary += "\nError breakdown by type:\n"
        for error_type, count in error_types.most_common():
            error_summary += f"  {error_type}: {count}\n"

        error_summary += "\nError breakdown by component:\n"
        for loc, count in loc_errors.most_common():
            loc_str = ".".join(loc)
            error_summary += f"  {loc_str}: {count}\n"

        # Show detailed errors with improved formatting
        errors_to_show = (
            validation_error.errors() if verbose else validation_error.errors()[:20]
        )
        error_summary += (
            f"\nErrors for debugging ({'all' if verbose else 'first 20'}):\n"
        )
        for i, error in enumerate(errors_to_show):
            # Format location more readably
            loc_parts = []
            for part in error.get("loc", []):
                if isinstance(part, int):
                    loc_parts.append(f"[{part}]")
                else:
                    loc_parts.append(str(part))

            # Build readable location string
            readable_loc = ""
            for j, part in enumerate(loc_parts):
                if j == 0:
                    readable_loc = part
                elif part.startswith("["):
                    readable_loc += part  # Index directly follows
                else:
                    readable_loc += f" -> {part}"

            # Add name from input if available
            input_data: Any = error.get("input", {})
            if isinstance(input_data, dict) and "name" in input_data:
                name = input_data["name"]
                if readable_loc and not readable_loc.endswith("]"):
                    readable_loc += f"('{name}')"

            msg = error.get("msg", "Unknown error")
            error_summary += f"  {i + 1}. {readable_loc}: {msg}\n"

        if not verbose and error_count > 20:
            error_summary += f"  ... and {error_count - 20} more errors (use verbose=True to see all)\n"

        return error_summary

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
        self.parameters: dict[str, TensorVar] = {}
        self.functions: dict[str, TensorVar] = {}
        self.distributions: dict[str, TensorVar] = {}
        self.mode = mode
        self._compiled_functions: dict[str, Callable[..., npt.NDArray[np.float64]]] = {}

        # Build dependency graph with proper entity identification
        self._build_dependency_graph(functions, distributions, progress)

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
        graph, constants_map = build_dependency_graph(
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
                    "parameter", "constant", "function", "distribution"
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
                }
                context = Context(context_data)

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
                    param_kind = param_point.kind if param_point else pt.scalar
                    self.parameters[node_name] = create_bounded_tensor(
                        node_name, domain_bounds, param_kind
                    )

                elif node_type == "constant":
                    # Constants are pre-created by distributions - add to parameters
                    self.parameters[node_name] = constants_map[node_name]

                elif node_type == "function":
                    # Functions are evaluated by design
                    self.functions[node_name] = functions[node_name].expression(context)

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

        Args:
            name (str): Name of the distribution.

        Returns:
            Callable: Compiled PyTensor function.
        """
        if name not in self._compiled_functions:
            dist = self.distributions[name]
            inputs = [var for var in graph_inputs([dist]) if var.name is not None]

            # Use the specified PyTensor mode
            compilation_mode = self.mode

            self._compiled_functions[name] = cast(
                Callable[..., npt.NDArray[np.float64]],
                function(
                    inputs=inputs,
                    outputs=dist,
                    mode=compilation_mode,
                    on_unused_input="ignore",
                ),  # type: ignore[no-untyped-call]
            )
        return self._compiled_functions[name]

    def pdf(self, name: str, **parametervalues: float) -> npt.NDArray[np.float64]:
        """
        Evaluates the probability density function of the specified distribution.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (float): Values for each distribution parameter.

        Returns:
            npt.NDArray[np.float64]: The evaluated PDF value.
        """
        if self.mode != "FAST_COMPILE":
            # Use compiled function for better performance
            func = self._get_compiled_function(name)
            inputs = [
                var
                for var in graph_inputs([self.distributions[name]])
                if var.name is not None
            ]
            positional_values = []
            for var in inputs:
                assert var.name is not None
                positional_values.append(parametervalues[var.name])
            return func(*positional_values)
        # Use original uncompiled approach
        dist = self.distributions[name]
        inputs = [var for var in graph_inputs([dist]) if var.name is not None]
        keyword_values: dict[str, float] = {}
        for var in inputs:
            assert var.name is not None
            keyword_values[var.name] = parametervalues[var.name]

        func = cast(
            Callable[..., npt.NDArray[np.float64]],
            function(inputs=inputs, outputs=dist),  # type: ignore[no-untyped-call]
        )
        return func(**keyword_values)

    def logpdf(self, name: str, **parametervalues: float) -> npt.NDArray[np.float64]:
        """
        Evaluates the natural logarithm of the PDF.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (float): Values for each distribution parameter.

        Returns:
            npt.NDArray[np.float64]: The log of the PDF.
        """
        return np.log(self.pdf(name, **parametervalues))

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

        pydotprint(  # type: ignore[no-untyped-call]
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
        inputs = list(graph_inputs([dist]))

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
