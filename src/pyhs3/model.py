from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
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

from pyhs3.context import Context
from pyhs3.distributions import Distributions
from pyhs3.domains import Domain
from pyhs3.functions import Functions
from pyhs3.networks import build_dependency_graph
from pyhs3.parameter_points import ParameterSet
from pyhs3.typing.aliases import DomainBounds, TensorVar

if TYPE_CHECKING:
    from pyhs3.likelihoods import Likelihood

log = logging.getLogger(__name__)


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
        observables: dict[str, tuple[float, float]] | None = None,
        likelihood: Likelihood | None = None,
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
            observables (dict[str, tuple[float, float]] | None): Dictionary mapping observable names to (lower, upper) bounds.

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
        self._observables = {
            name: (pt.constant(lower), pt.constant(upper))
            for name, (lower, upper) in (observables or {}).items()
        }
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
        self._likelihood = likelihood

        # Build dependency graph with proper entity identification
        self._build_dependency_graph(functions, distributions, progress)

    @property
    def data(self) -> dict[str, npt.NDArray[np.float64]]:
        """Observed data arrays from the workspace, keyed by observable name.

        Only available when the model was built via ``ws.model(analysis)`` or
        ``ws.model(likelihood)``.  Raises ``RuntimeError`` otherwise.

        Returns a dict suitable for passing directly to a compiled or JAX
        function alongside :attr:`free_params`::

            jg = pyhs3.jaxify(model.log_prob)
            jg(**model.data, **model.free_params)
        """
        if self._likelihood is None:
            msg = "data requires a likelihood context; build via ws.model(analysis)"
            raise RuntimeError(msg)
        return self._likelihood.data_arrays()

    @property
    def nominal_params(self) -> dict[str, float]:
        """Default parameter values from the workspace parameter set.

        Returns all parameters, including those marked ``const=True`` (which
        are baked as :func:`pytensor.tensor.constant` in the symbolic graph
        and are therefore not free inputs to a jaxified expression).

        Use :attr:`free_params` when passing parameters to a jaxified callable
        to avoid supplying spurious keyword arguments.
        """
        result: dict[str, float] = {}
        for pp in self.parameterset:
            result[pp.name] = float(pp.value)
        return result

    @property
    def free_params(self) -> dict[str, float]:
        """Non-constant parameter values from the workspace parameter set.

        Like :attr:`nominal_params` but excludes parameters whose
        ``ParameterPoint.const`` flag is ``True``.  These are the parameters
        that remain as free symbolic inputs after model construction, making
        this dict the correct one to pass to a jaxified callable::

            jg = pyhs3.jaxify(model.log_prob)
            jg(**model.data, **model.free_params)
        """
        result: dict[str, float] = {}
        for pp in self.parameterset:
            if not pp.const:
                result[pp.name] = float(pp.value)
        return result

    @property
    def log_prob(self) -> TensorVar:
        """Symbolic joint log-probability expression for the full likelihood.

        Returned as a 1-D PyTensor ``TensorVar`` of shape ``(M,)``, where
        ``M`` is the parameter batch size.  For all-scalar (non-vectorised)
        parameters ``M = 1``; for a profile scan over ``M`` points the shape
        is ``(M,)``.  Observable data and parameters listed in
        :attr:`free_params` are symbolic free inputs; parameters with
        ``const=True`` are baked as compile-time constants and do not appear
        as free inputs.  The expression is suitable for JAX transpilation,
        gradient computation, or direct PyTensor compilation.

        Normalization denominators are fixed constants (axis bounds baked at
        ``Model`` construction time).  For unweighted data, the same compiled/JAX
        function can be evaluated against different event arrays.  **Weighted**
        ``UnbinnedData`` entries bake the weights as constants at construction time;
        to use different weights, a new ``Model`` must be built.

        The workspace defaults for evaluation are available via :attr:`data`
        and :attr:`free_params`.

        Only available when the model was built via ``ws.model(analysis)`` or
        ``ws.model(likelihood)``.  Raises ``RuntimeError`` otherwise.

        Example::

            model = ws.model(ws.analyses["CombinedPdf_combData"])
            nll = -2 * model.log_prob
            jg = pyhs3.jaxify(nll)
            val = jg(**model.data, **model.free_params)
        """
        if self._likelihood is None:
            msg = "log_prob requires a likelihood context; build via ws.model(analysis)"
            raise RuntimeError(msg)

        # Accumulate with + so shapes broadcast correctly across the parameter axis.
        # Summing over axis 0 (events) yields shape (M,) — the parameter batch size.
        lp: TensorVar = pt.constant(np.float64(0.0))

        for dist_obj, datum in zip(
            self._likelihood.distributions, self._likelihood.data, strict=True
        ):
            if isinstance(datum, str):
                continue
            entries = getattr(datum, "entries", None)
            if entries is None:
                continue

            dist_name = dist_obj if isinstance(dist_obj, str) else dist_obj.name

            # model.distributions[name] is the normalized PDF expression with
            # observables as symbolic pt.vector free inputs.
            log_pdf: TensorVar = pt.log(self.distributions[dist_name])

            weights = getattr(datum, "weights", None)
            if weights is not None:
                warnings.warn(
                    f"'{datum.name}' has per-event weights; weights are baked as "
                    f"pt.constant and cannot be changed without rebuilding the model.",
                    UserWarning,
                    stacklevel=2,
                )
                # (N,) → (N, 1) so it broadcasts correctly against (N, M) log_pdf
                weights_t = pt.constant(np.asarray(weights, dtype=np.float64))[:, None]
                lp = lp + pt.sum(weights_t * log_pdf, axis=0)  # type: ignore[no-untyped-call]
            else:
                lp = lp + pt.sum(log_pdf, axis=0)  # type: ignore[no-untyped-call]

        # Auxiliary distributions (constraint terms) are scalars; they broadcast
        # onto the parameter-scan axis when non-scalar params are present.
        if self._likelihood.aux_distributions:
            for aux_name in self._likelihood.aux_distributions:
                if aux_name in self.distributions:
                    lp = lp + pt.log(self.distributions[aux_name])

        return lp

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

    def _build_parameter_node(self, node_name: str, context: Context) -> TensorVar:
        """Build a parameter node: baked constant (const=True) or bounded free variable."""
        param_point = self.parameterset.get(node_name) if self.parameterset else None
        domain_bounds = (
            self.domain.get(node_name, (None, None)) if self.domain else (None, None)
        )
        if param_point and param_point.const:
            # Bake as a compile-time constant so it is invisible to
            # explicit_graph_inputs and JAX transpilation.
            val = np.float64(param_point.value)
            lower, upper = domain_bounds
            if (lower is not None and val < lower) or (
                upper is not None and val > upper
            ):
                warnings.warn(
                    f"Parameter '{node_name}' has const=True with value"
                    f" {val} outside domain [{lower}, {upper}];"
                    f" using the specified value as-is.",
                    stacklevel=2,
                )
            return pt.constant(val, name=node_name)

        is_observable = "_observed" in node_name or node_name in context.observables

        # Free variable: determine default kind (vector for observables, scalar otherwise)
        default_kind: Callable[..., TensorVar] = (
            pt.vector if is_observable else pt.scalar
        )

        # Allow explicit override from ParameterPoint.kind
        if param_point and param_point.kind is not None:
            param_kind = param_point.kind
            if param_kind is not default_kind:
                warnings.warn(
                    f"Parameter '{node_name}' has kind override"
                    f" {param_kind.__name__} (default would be"
                    f" {default_kind.__name__})",
                    stacklevel=2,
                )
        else:
            param_kind = default_kind

        tensor = create_bounded_tensor(node_name, domain_bounds, param_kind)

        # Shape convention for vector parameters so broadcasting is unambiguous:
        #   observables → (N, 1): events on the first axis
        #   non-observable overrides → (1, N): scan dimension on the second axis
        # Scalars have no axes and broadcast trivially — no reshaping needed.
        if param_kind is pt.vector:
            # (N,) → (N, 1) if is_observable else (N,) → (1, N)
            tensor = tensor[:, None] if is_observable else tensor[None, :]
            tensor.name = node_name  # propagate name through shape op

        return tensor

    def _build_constant_node(
        self, node_name: str, constants_map: dict[str, TensorVar]
    ) -> TensorVar:
        """Build a constant node from the pre-computed constants map."""
        return constants_map[node_name]

    def _build_function_node(
        self, node_name: str, functions: Functions, context: Context
    ) -> TensorVar:
        """Build a function node by evaluating its symbolic expression."""
        return functions[node_name].expression(context)

    def _build_modifier_node(
        self, node_name: str, modifiers_map: dict[str, Any], context: Context
    ) -> TensorVar:
        """Build a modifier node by evaluating its symbolic expression."""
        return cast(TensorVar, modifiers_map[node_name].expression(context))

    def _build_distribution_node(
        self, node_name: str, distributions: Distributions, context: Context
    ) -> TensorVar:
        """Build a distribution node by evaluating its symbolic expression."""
        return distributions[node_name].expression(context)

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
        graph, constants_map, modifiers_map = build_dependency_graph(
            self.parameterset, functions, distributions
        )

        sorted_nodes = graph.topological_sort()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", style="cyan"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
            transient=True,
            disable=not progress,
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

                max_name_length = 60
                display_name = (
                    node_name[: max_name_length - 3] + "..."
                    if len(node_name) > max_name_length
                    else node_name
                )
                progress_bar.update(
                    task,
                    description=f"Building {node_type:<12}: {display_name:<{max_name_length}}",
                )

                context = Context(
                    parameters={
                        **self.parameters,
                        **self.functions,
                        **self.distributions,
                        **self.modifiers,
                    },
                    observables=self._observables,
                )

                if node_type == "parameter":
                    self.parameters[node_name] = self._build_parameter_node(
                        node_name, context
                    )
                elif node_type == "constant":
                    self.parameters[node_name] = self._build_constant_node(
                        node_name, constants_map
                    )
                elif node_type == "function":
                    self.functions[node_name] = self._build_function_node(
                        node_name, functions, context
                    )
                elif node_type == "modifier":
                    self.modifiers[node_name] = self._build_modifier_node(
                        node_name, modifiers_map, context
                    )
                else:  # node_type == "distribution"
                    self.distributions[node_name] = self._build_distribution_node(
                        node_name, distributions, context
                    )

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
    name: str, domain: DomainBounds, kind: Callable[..., TensorVar] = pt.scalar
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
