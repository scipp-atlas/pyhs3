# pylint: disable=too-many-lines
from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pytensor.compile.maker import function
from pytensor.compile.mode import Mode
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

from pyhs3.compiled import CompiledNLL, compile_nll
from pyhs3.context import Context
from pyhs3.data import BinnedData
from pyhs3.distributions import Distribution, Distributions, HistFactoryDistChannel
from pyhs3.distributions.core import BatchableDistribution
from pyhs3.domains import Domain
from pyhs3.functions import Functions
from pyhs3.functions.core import BatchableFunction, Function
from pyhs3.graph_viz import (
    DEFAULT_OP_PARAMS,
    ConstArrayMode,
    OpParamMode,
    render_graph,
)
from pyhs3.networks import Family, build_dependency_graph, plan_families
from pyhs3.parameter_points import ParameterSet
from pyhs3.tensorutils import (
    create_bounded_tensor,
    ensure_array,
    merge_duplicate_nodes,
)
from pyhs3.typing.aliases import EntityType, TensorVar

if TYPE_CHECKING:
    from pyhs3.likelihoods import Likelihood

log = logging.getLogger(__name__)

# PyTensor mode a :class:`Model` uses unless built with another one.
_DEFAULT_MODE = "CVM"


def _drop_view_axis(term: TensorVar) -> TensorVar:
    """Return a likelihood term as a scalar or an ``(M,)`` vector.

    Batched non-observable parameters reach distributions as ``(1, M)``
    views so that they broadcast against ``(N, 1)`` observables.  Per-event
    terms lose that leading axis when summed over events, but terms that are
    not summed (once-per-channel yields, constraints) keep it as ``(1, M)``,
    and adding them to an ``(M,)`` term would give ``(1, M)``.  Dropping the
    length-1 axis lets every term broadcast to the ``(M,)`` result; ``()``
    and ``(M,)`` terms are returned unchanged.
    """
    if term.ndim == 2:
        # Fails at graph construction if the leading axis is not length 1.
        return cast(TensorVar, pt.squeeze(term, axis=0))  # type: ignore[no-untyped-call]
    return term


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

    **Single log-space pipeline.** Every distribution node is built exactly
    once, in log space, as ``self.log_distributions[name]``; the
    probability-space counterpart ``self.distributions[name]`` is always
    derived as ``pt.exp(self.log_distributions[name])`` rather than built
    independently, so the two can never disagree on normalization or on their
    free inputs -- :meth:`pars` is therefore valid for both :meth:`pdf` and
    :meth:`logpdf` by construction.

    **HFDC constraint storage.** For ``HistFactoryDistChannel`` distributions,
    ``self.log_distributions[name]`` stores the full per-channel log
    expression (summed Poisson log-pmf plus summed log-constraints) so that
    ``logpdf(name, **params)`` matches pyhf/cabinetry semantics for callers
    asking about a single channel's probability; ``self.distributions[name]``
    is ``pt.exp`` of that.  ``self._hfdc_log_poisson[name]`` stores only the
    log-space Poisson term; ``log_prob`` uses it to assemble the joint NLL
    without double-counting constraint factors when multiple channels share a
    nuisance parameter.  Log-space constraint expressions are appended to
    ``self._hfdc_log_constraints`` exactly once per unique dedup key across
    all channels: single-parameter modifiers (``normsys``, ``histosys``) are
    deduped by parameter name using ``self._hfdc_constraint_params_seen``;
    multi-parameter modifiers (``shapesys``, ``staterror``) are channel-local
    by workspace validation and always emitted as-is.  ``log_prob`` sums
    ``self._hfdc_log_constraints`` directly, so it stays finite where a
    probability-space constraint would underflow to 0.0.

    HS3 Reference:
        Models are computational representations of :external+hs3:ref:`HS3 workspaces <hs3.file-format>`.
    """

    def __init__(
        self,
        *,
        parameterset: ParameterSet,
        distributions: Distributions,
        domain: Domain,
        functions: Functions,
        progress: bool = True,
        mode: str = _DEFAULT_MODE,
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
            mode (str): PyTensor compilation mode. Defaults to "CVM".
                       Options: "CVM" (apply all rewrites, use C implementations),
                       "FAST_RUN" (all rewrites; with PyTensor's config.linker
                       "auto" (the default) or "numba" it compiles with Numba, which
                       does not finish compiling large workspaces, and with "cvm"
                       it is "CVM"),
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
            modifiers (dict[str, pytensor.tensor.variable.TensorVariable]): HistFactory modifier terms
                (normalization factors, shape systematics, and similar) discovered while building the graph.
            log_distributions (dict[str, pytensor.tensor.variable.TensorVariable]): Log-space expression for
                each distribution, keyed by name; what :meth:`logpdf`/:meth:`logpdf_unsafe` evaluate.
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
        # Compiled-function caches: one per output (probability-space vs
        # log-space), since each compiles a different expression down to its
        # own Callable.  But the two expressions differ only by the trailing
        # pt.exp() wrapper (self.distributions[name] is always
        # pt.exp(self.log_distributions[name])), so their free inputs are
        # identical by construction -- _compiled_inputs/_compiled_input_names
        # are therefore a single cache shared by both compile paths (see
        # _compile_expression), populated by whichever of pdf()/logpdf()/
        # pars() compiles first.
        self._compiled_functions: dict[str, Callable[..., npt.NDArray[np.float64]]] = {}
        self._compiled_log_functions: dict[
            str, Callable[..., npt.NDArray[np.float64]]
        ] = {}
        self._compiled_inputs: dict[str, list[TensorVar]] = {}
        # Ordered input names cached alongside _compiled_inputs so that pars()
        # and _reorder_params() avoid rebuilding the list on every pdf/logpdf call.
        self._compiled_input_names: dict[str, list[str]] = {}
        # Log-space distribution expressions (dist.log_expression(context)), the
        # only expression built per distribution node.  self.distributions is
        # always pt.exp() of the corresponding entry here (see
        # _build_distribution_node).  logpdf and log_prob use these to
        # evaluate in log space so that log(prod(exp(...))) stays collapsed
        # and does not underflow to -inf.
        self.log_distributions: dict[str, TensorVar] = {}
        self._likelihood = likelihood
        # Views used internally for broadcasting: leaf[:, None] for observables,
        # leaf[None, :] for non-observable vector overrides.  Distributions see
        # these via Context; model.parameters[name] always holds the leaf.
        self._views: dict[str, TensorVar] = {}
        # Pre-built HFDC log-space constraint expressions, collected during
        # graph construction. ParameterModifier constraints are deduped by
        # parameter name across channels; ParametersModifier constraints
        # (shapesys/staterror) are emitted per-channel.
        self._hfdc_log_constraints: list[TensorVar] = []
        self._hfdc_constraint_params_seen: set[str] = set()
        # Entity type of every user-facing name, for entity_type().  Family
        # nodes are excluded: their names are generated, not user-defined.
        self._entity_types: dict[str, EntityType] = {}
        # Log-space Poisson-only (no constraint sum) expression for each HFDC
        # channel: the summed per-bin Poisson log-pmf. log_prob uses this
        # instead of log(exp(...)) so the joint NLL stays finite where the
        # probability-space product would underflow.
        self._hfdc_log_poisson: dict[str, TensorVar] = {}
        # (log-space, probability-space) expressions of each distinct
        # distribution, keyed by Distribution.structural_key(), so that
        # distributions identical apart from their name are built once.
        self._shared_expressions: dict[str, tuple[TensorVar, TensorVar]] = {}
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
        are baked as ``pytensor.tensor.constant`` in the symbolic graph
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
        ``ParameterPoint.const`` flag is ``True`` and observables (whose values
        come from :attr:`data`).  These are the parameters
        that remain as free symbolic inputs after model construction, making
        this dict the correct one to pass to a jaxified callable::

            jg = pyhs3.jaxify(model.log_prob)
            jg(**model.data, **model.free_params)
        """
        result: dict[str, float] = {}
        for pp in self.parameterset:
            if not pp.const and pp.name not in self._observables:
                result[pp.name] = float(pp.value)
        return result

    def compile_nll(
        self,
        *,
        gradient: bool = True,
        backend: Literal["c", "jax"] = "c",
        fuse: bool = False,
        mode: Mode | str | None = None,
        cache: bool | str | os.PathLike[str] | None = None,
        batched: bool = False,
    ) -> CompiledNLL:
        """Compile the negative log-likelihood for a minimizer.

        Returns a :class:`~pyhs3.compiled.CompiledNLL`: ``-sum(log_prob)``
        and, by default, its gradient, compiled once as a function of a single
        1-D vector of the free parameters (ordered as
        :attr:`~pyhs3.compiled.CompiledNLL.parameter_names`).  The observed
        data and every ``const`` parameter are baked into the compiled
        function, so each call passes only the free parameters.  The result
        also carries the starting point (:attr:`free_params`) and the domain
        bounds of each parameter, ready for SciPy or iminuit.

        This is the recommended way to evaluate a likelihood repeatedly; see
        :doc:`/howto/fit_model`.

        Args:
            gradient: Also compile the gradient with respect to the free
                parameters.  The value and gradient then come from one
                compiled function.
            backend: ``"c"`` (default) compiles with PyTensor's C linker.
                ``"jax"`` transpiles the graph to JAX and compiles it with
                XLA, which needs the optional JAX dependency
                (``pip install pyhs3[jax]``) and enables JAX's float64 mode
                (a process-wide JAX setting).  On large workspaces it
                compiles faster than the C backend and evaluates the value
                and gradient several times faster.
            fuse: Keep PyTensor's elemwise-fusion rewrites (C backend only).  Fused kernels
                evaluate faster but make the first compile of a large
                workspace several times slower; see
                :func:`~pyhs3.compiled.default_nll_mode`.  Use ``fuse=True``
                for long fitting campaigns, ideally with ``cache=True``.
            mode: PyTensor compile mode, replacing the default
                ``default_nll_mode(fuse=fuse)``; e.g. ``"CVM"`` (C
                backend only).  Cannot be combined with ``fuse=True``.
                With ``mode=None``, the C backend uses the ``mode`` the
                model was built with if that is not the default ``"CVM"``
                and ``fuse`` is ``False``; otherwise it uses
                ``default_nll_mode(fuse=fuse)``.  The JAX backend ignores
                the model's ``mode``.
            cache: Cache the compiled function on disk and reuse it when the
                same likelihood is compiled again with the same options,
                pyhs3 and PyTensor versions.  ``True`` uses
                ``$XDG_CACHE_HOME/pyhs3`` (``~/.cache/pyhs3``); a path uses
                that directory; ``None`` or ``False`` disables caching.  The
                cached file refers to PyTensor's compiled C modules, so it is
                only valid on the machine (and PyTensor compile directory)
                that wrote it.  With ``backend="jax"`` the cache holds a
                serialized ``jax.export`` module, whose loading skips the
                PyTensor-to-JAX transpilation, and JAX's persistent
                compilation cache in its ``xla`` subdirectory holds the XLA
                executable.  JAX's cache directory is a process-wide setting:
                pyhs3 sets ``jax_compilation_cache_dir`` only if it is unset,
                so a directory you configured yourself is kept.
            batched: Evaluate many parameter vectors per call
                (``backend="jax"`` only).  Returns a
                :class:`~pyhs3.compiled.BatchedCompiledNLL`, whose methods take
                a ``(K, P)`` array with one parameter vector per row and
                return values ``(K,)`` and gradients ``(K, P)``; XLA compiles
                once per batch size ``K``, at the first call with that size.

        Returns:
            CompiledNLL: The compiled likelihood.

        Raises:
            RuntimeError: If the model was built without a likelihood.
            ValueError: If a free parameter has no starting value in the
                parameter set (the message lists every such name).  Also if a
                free parameter is batched (``kind=pt.vector``);
                the parameter vector has one entry per scalar parameter.  Also
                for an unknown backend, for ``fuse=True`` combined with
                ``mode=``, for ``fuse``/``mode`` with ``backend="jax"``, and
                for ``batched=True`` without ``backend="jax"``.
            ImportError: If ``backend="jax"`` and JAX is not installed.
        """
        if mode is None and backend == "c" and not fuse and self.mode != _DEFAULT_MODE:
            mode = self.mode
        log_prob = self.log_prob
        data = self.data
        start = {**self.nominal_params, **self.free_params}
        free: list[TensorVar] = []
        baked: dict[TensorVar, npt.NDArray[np.float64]] = {}
        names: list[str] = []
        inputs = [
            cast(TensorVar, var)
            for var in explicit_graph_inputs([log_prob])
            if var.name is not None
        ]
        for var in inputs:
            if var.name in data:
                baked[var] = data[var.name]
            elif var.ndim == 0:
                free.append(var)
                names.append(cast(str, var.name))
            else:
                msg = (
                    f"Free parameter '{var.name}' is batched (ndim={var.ndim}); "
                    "compile_nll needs scalar free parameters, one entry each "
                    "in the parameter vector."
                )
                raise ValueError(msg)
        if missing := sorted(set(names) - start.keys()):
            msg = (
                f"No starting value for free parameter(s): {', '.join(missing)}. "
                "Set them in the parameter point, or pass a complete "
                "parameter_set= to Workspace.model."
            )
            raise ValueError(msg)
        bounds = [self.domain.get(name, (None, None)) for name in names]
        return compile_nll(
            log_prob,
            free,
            baked,
            [start[name] for name in names],
            bounds,
            gradient=gradient,
            backend=backend,
            fuse=fuse,
            mode=mode,
            cache=cache,
            batched=batched,
        )

    @cached_property
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

        # Collect per-channel and aux terms into a flat list, then combine with a
        # single n-ary pt.add so that PyTensor sees one Elemwise{add} node rather
        # than a left-deep chain of binary Adds.  Summing over axis 0 (events)
        # yields shape (M,) per term — the parameter batch size.
        terms: list[TensorVar] = []

        # Track constraint factor names already contributed so that constraints
        # shared across multiple channels are only counted once globally.
        # This prevents the N-fold overcounting that occurs when constraint PDFs
        # (which do not depend on the observable) appear in a ProductDist alongside
        # the shape PDF: naively summing log(shape x Π_j constr_j) over N events
        # multiplies each constr_j term by N rather than counting it once.
        # Once-per-likelihood counting matches RooFit but is not specified by
        # HS3; see https://github.com/scipp-atlas/pyhs3/issues/240.
        seen_constraint_factors: set[str] = set()

        for dist_obj, datum in zip(
            self._likelihood.distributions, self._likelihood.data, strict=True
        ):
            if isinstance(datum, str):
                continue
            entries = getattr(datum, "entries", None)
            # zip() joins str and Distribution to object for mypy; narrow on the class.
            dist_name = (
                dist_obj.name if isinstance(dist_obj, Distribution) else str(dist_obj)
            )

            if entries is None:
                # HFDC: log_prob uses the Poisson-only term here; deduplicated
                # constraint terms are added after the loop.  Non-HFDC distributions
                # paired with BinnedData raise: HS3 does not define the
                # likelihood of a continuous pdf evaluated on binned data
                # (bin centers vs bin integrals), and ROOT exports do not record
                # which convention the original fit used.  Native support is
                # tracked in https://github.com/scipp-atlas/pyhs3/issues/242
                # (spec: hep-statistics-serialization-standard issue #93,
                # export: root-project/root issue #22598).
                if dist_name in self._hfdc_log_poisson:
                    # Log-space Poisson term: sum of per-bin log-pmfs, finite even
                    # where the probability-space product underflows to 0.0.
                    terms.append(self._hfdc_log_poisson[dist_name])
                elif isinstance(datum, BinnedData):
                    msg = (
                        f"Distribution '{dist_name}' is paired with binned data "
                        f"'{datum.name}', but HS3 does not define the likelihood "
                        f"of a non-HistFactory distribution on binned data "
                        f"(see https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/issues/93). "
                        f"Convert the binned data to unbinned data instead: "
                        f"entries = bin centres, weights = bin contents."
                    )
                    raise ValueError(msg)
                else:
                    # Only PointData remains: UnbinnedData always has entries, and
                    # PointData only has entries when axes name the variate.
                    msg = (
                        f"Distribution '{dist_name}' is paired with point data "
                        f"'{datum.name}', but pyhs3 cannot determine the "
                        f"distribution's single variate. Give the point data "
                        f"'axes' naming the variate."
                    )
                    raise ValueError(msg)
                continue

            # Resolve weight tensor once for this datum.
            weights = getattr(datum, "weights", None)
            weights_t: TensorVar | None = None
            if weights is not None:
                warnings.warn(
                    f"'{datum.name}' has per-event weights; weights are baked as "
                    f"pt.constant and cannot be changed without rebuilding the model.",
                    UserWarning,
                    stacklevel=2,
                )
                weights_arr = np.asarray(weights, dtype=np.float64)
                total_weight = float(np.sum(weights_arr))
                if not np.isfinite(total_weight) or np.isclose(total_weight, 0.0):
                    msg = (
                        f"'{datum.name}' has invalid total weight ({total_weight}); "
                        "sum of weights must be finite and non-zero."
                    )
                    raise ValueError(msg)
                # (N,) → (N, 1) so it broadcasts against (N, M) log_pdf.
                weights_t = pt.constant(weights_arr)[:, None]

            # Each distribution describes its own structured contributions
            # (per-event log-density, once-per-channel scalars, named
            # constraints) via Distribution.log_prob_terms; the model owns
            # the channel-dataset pairing and applies event weights, sums
            # over events, and deduplicates constraints globally.
            contrib = self._distribution_objects[dist_name].log_prob_terms(
                self.distributions,
                self.log_distributions,
                self._distribution_objects,
            )

            # Sum the per-event log-density over events.  Weighted:
            # Σᵢ wᵢ log f(xᵢ); unweighted: Σᵢ log f(xᵢ).  No /total_weight:
            # that would give an average NLL (wrong scale).  The weighted form
            # follows RooFit; HS3 does not define the weighted likelihood — see
            # https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/issues/92
            # and https://github.com/scipp-atlas/pyhs3/issues/241.
            if contrib.per_event:
                log_pdf: TensorVar = (
                    pt.add(*contrib.per_event)
                    if len(contrib.per_event) > 1
                    else contrib.per_event[0]
                )
                if weights_t is not None:
                    terms.append(pt.sum(weights_t * log_pdf, axis=0))  # type: ignore[no-untyped-call]
                else:
                    terms.append(pt.sum(log_pdf, axis=0))  # type: ignore[no-untyped-call]

            # Once-per-channel scalar terms (e.g. the -nu yield term of an
            # extended mixture).
            terms.extend(contrib.channel)

            # Add each unique constraint exactly once globally; constraints
            # shared with an earlier channel are skipped.
            for factor_name, constraint_expr in contrib.constraints.items():
                if factor_name not in seen_constraint_factors:
                    seen_constraint_factors.add(factor_name)
                    terms.append(constraint_expr)

        # Auxiliary distributions (constraint terms) are scalars; they broadcast
        # onto the parameter-scan axis when non-scalar params are present.
        if self._likelihood.aux_distributions:
            terms.extend(
                self.log_distributions[
                    aux_name if isinstance(aux_name, str) else aux_name.name
                ]
                for aux_name in self._likelihood.aux_distributions
            )

        # HFDC constraint terms: collected once per unique nuisance parameter
        # across all channels during graph construction. These are log-space
        # terms built directly (not pt.log() of a probability-space product),
        # so this stays finite where a probability-space constraint would
        # underflow to 0.0.
        terms.extend(self._hfdc_log_constraints)

        if not terms:
            return pt.constant(np.float64(0.0))
        terms = [_drop_view_axis(term) for term in terms]
        joint = (
            terms[0] if len(terms) == 1 else cast(TensorVar, pt.add(*terms))  # pylint: disable=no-value-for-parameter
        )
        # Distributions that share parameters build identical subgraphs (e.g.
        # mixture summands with the same shape); merge them here so that
        # PyTensor's rewriter does not have to.
        return merge_duplicate_nodes(joint)

    def _try_bake_hfdc_observed(self, node_name: str) -> TensorVar | None:
        """Return a baked constant for an HFDC ``{name}_observed`` parameter, or None.

        When a likelihood pairs an HFDC distribution with a BinnedData object,
        the observed bin counts are baked in as compile-time constants so that
        they are invisible to ``explicit_graph_inputs`` and JAX transpilation.
        """
        if self._likelihood is None:
            return None
        datum = self._likelihood.binned_observed_inputs().get(node_name)
        if datum is None:
            return None
        return pt.constant(np.asarray(datum.contents, dtype=np.float64), name=node_name)

    def _build_parameter_node(self, node_name: str, context: Context) -> TensorVar:
        """Build a parameter node: baked constant (const=True) or bounded free variable."""
        baked = self._try_bake_hfdc_observed(node_name)
        if baked is not None:
            return baked

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

        is_observable = (
            node_name.endswith("_observed") or node_name in context.observables
        )

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

        # For vector parameters, cache the ExpandDims view used internally for
        # broadcasting, but return the leaf so model.parameters[name] is 1-D:
        #   observables → leaf[:, None] cached as (N, 1) view
        #   non-observable overrides → leaf[None, :] cached as (1, N) view
        # Scalars have no axes and broadcast trivially — no reshaping needed.
        if param_kind is pt.vector:
            view: TensorVar = tensor[:, None] if is_observable else tensor[None, :]
            view.name = node_name  # propagate name through shape op for readability
            self._views[node_name] = view

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
        """Build a distribution node from its log-space expression.

        Only the log-space expression is built per node; the probability-
        space counterpart stored in ``self.distributions`` is always
        ``pt.exp()`` of it, so the two agree by construction (no separate
        probability-space graph to keep in sync, and identical free inputs
        for :meth:`pars`).

        HistFactory channels are assembled by
        :meth:`_build_histfactory_channel_node`; every other distribution by
        :meth:`_build_generic_distribution_node`.
        """
        dist = distributions[node_name]
        if isinstance(dist, HistFactoryDistChannel):
            return self._build_histfactory_channel_node(node_name, dist, context)
        return self._build_generic_distribution_node(node_name, dist, context)

    def _build_generic_distribution_node(
        self, node_name: str, dist: Distribution, context: Context
    ) -> TensorVar:
        """Build a non-HistFactory distribution node, sharing identical ones.

        A distribution equal to an already built one in every field but its
        name (same :meth:`~pyhs3.distributions.Distribution.structural_key`)
        builds the identical expression, so it reuses those tensors instead
        of building (and later merging) a second copy.  The shared tensors
        keep the first distribution's name.
        """
        key = dist.structural_key() if dist.share_identical_expressions else None
        if key is not None and key in self._shared_expressions:
            log_expr, full = self._shared_expressions[key]
            self.log_distributions[node_name] = log_expr
            return full
        log_expr = dist.log_expression(context)
        full = self._register_log_expression(node_name, log_expr)
        if key is not None:
            self._shared_expressions[key] = (log_expr, full)
        return full

    def _register_log_expression(
        self, node_name: str, log_expr: TensorVar
    ) -> TensorVar:
        """Record a distribution's log-space expression and return ``pt.exp`` of it."""
        self.log_distributions[node_name] = log_expr
        full = cast(TensorVar, pt.exp(log_expr))
        full.name = node_name  # Evaluable.expression sets the name
        return full

    def _build_histfactory_channel_node(
        self, node_name: str, dist: HistFactoryDistChannel, context: Context
    ) -> TensorVar:
        """Build a HistFactory channel node directly in log space.

        The full expression is assembled in log space and stored in
        ``self.log_distributions`` so that :meth:`logpdf` continues to work as
        before.  In addition, the Poisson-only log term is stored in
        ``self._hfdc_log_poisson`` and the deduplicated log-constraint terms
        are appended to ``self._hfdc_log_constraints`` so that
        :attr:`log_prob` can combine them without double-counting when
        multiple channels share a nuisance parameter.
        """
        # Build the log-space Poisson term and each log-space constraint
        # factor exactly once, then assemble the model-level deduped
        # constraint list and the full per-channel log expression from those
        # shared pieces.  This mirrors HistFactoryDistChannel.log_expression/
        # log_extended_likelihood without rebuilding the Poisson subgraph or
        # re-running log_constraint a second time.  The probability-space
        # Poisson term and per-modifier make_constraint() are never called
        # here: the probability-space channel expression is derived below as
        # pt.exp() of the log expression instead.
        try:
            log_poisson = dist.log_likelihood(context)
        finally:
            # The captured TensorVar owns everything this Model needs. Do not
            # leave the shared workspace channel retaining the Context (and
            # therefore the complete model graph).
            dist._clear_expression_cache()  # pylint: disable=protected-access
        self._hfdc_log_poisson[node_name] = log_poisson

        channel_log_constraints = self._collect_histfactory_log_constraints(
            node_name, dist, context
        )

        # Assemble the full per-channel log expression: summed Poisson log-pmf
        # plus summed log-constraints (mirrors HistFactoryDistChannel.log_expression
        # and log_extended_likelihood, reusing the pieces built above instead of
        # calling dist.log_expression(context), which would rebuild log_poisson
        # and re-run log_constraint for every spec).
        if not channel_log_constraints:
            log_extended: TensorVar = cast(TensorVar, pt.constant(0.0))
        else:
            log_extended = cast(
                TensorVar,
                pt.sum(pt.stack(channel_log_constraints)),  # type: ignore[no-untyped-call]
            )
        log_full = cast(TensorVar, log_poisson + log_extended)
        self.log_distributions[node_name] = log_full

        # Probability-space per-channel expression, derived rather than
        # rebuilt from the (unused) probability-space Poisson/constraint terms.
        full = cast(TensorVar, pt.exp(log_full))
        full.name = dist.name  # Evaluable.expression sets the name
        return full

    def _collect_histfactory_log_constraints(
        self, node_name: str, dist: HistFactoryDistChannel, context: Context
    ) -> list[TensorVar]:
        """Build a channel's log-constraint terms, each exactly once.

        Returns the channel's own (per-channel deduplicated) log-constraint
        terms for its full log expression.  When the channel is part of the
        active likelihood, the terms not already contributed by another
        channel are also appended to ``self._hfdc_log_constraints`` for
        :attr:`log_prob`.
        """
        # Whether this channel participates in the active likelihood; only then
        # do its constraints feed the joint log_prob. Legacy models build all
        # channels, while likelihood-rooted models build only the active
        # dependency closure selected in _build_dependency_graph.
        in_likelihood = self._in_likelihood(node_name)

        # Single pass over constraint specs builds each log-constraint factor
        # exactly once.
        # - channel_seen dedups by parameter for the per-channel sum (matches
        #   log_extended_likelihood's local dedup),
        # - self._hfdc_constraint_params_seen dedups across channels for log_prob.
        channel_seen: set[str] = set()
        channel_log_constraints: list[TensorVar] = []
        for dedup_key, modifier, sample_data in dist.constraint_specs():
            log_constraint = modifier.log_constraint(context, sample_data)

            if dedup_key is None or dedup_key not in channel_seen:
                if dedup_key is not None:
                    channel_seen.add(dedup_key)
                channel_log_constraints.append(log_constraint)

            if in_likelihood:
                self._add_global_histfactory_constraint(dedup_key, log_constraint)

        # BB-lite mode's channel-level constraint (shared gamma parameters
        # combining all samples' statistical uncertainties) is not a
        # per-modifier spec, so constraint_specs() cannot yield it -- mirror
        # log_extended_likelihood's channel-level addition here.  It is
        # channel-local (like shapesys/staterror dedup_key=None specs above),
        # so it is always appended, with no cross-channel dedup key.
        if dist.barlow_beeston_method == "lite":
            lite_log_constraint = dist._make_barlow_beeston_lite_log_constraint(  # pylint: disable=protected-access
                context
            )
            if lite_log_constraint is not None:
                channel_log_constraints.append(lite_log_constraint)
                if in_likelihood:
                    self._hfdc_log_constraints.append(lite_log_constraint)

        return channel_log_constraints

    def _in_likelihood(self, node_name: str) -> bool:
        """Whether a distribution is paired with data in the active likelihood.

        Models built without a likelihood treat every distribution as active.
        """
        return self._likelihood is None or any(
            (d if isinstance(d, str) else d.name) == node_name
            for d in self._likelihood.distributions
        )

    def _add_global_histfactory_constraint(
        self, dedup_key: str | None, log_constraint: TensorVar
    ) -> None:
        """Contribute a HistFactory log-constraint to log_prob at most once.

        Constraints keyed by a nuisance parameter (``dedup_key``) are added the
        first time any channel builds them; channel-local constraints
        (``dedup_key is None``) are always added.
        """
        if dedup_key is not None:
            if dedup_key in self._hfdc_constraint_params_seen:
                return
            self._hfdc_constraint_params_seen.add(dedup_key)
        self._hfdc_log_constraints.append(log_constraint)

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

        likelihood_roots: list[str] = []
        if self._likelihood is not None:
            likelihood_roots = self._likelihood.distribution_names()

        # Batchable functions or distributions of one type that feed the same
        # root distribution become one family node, built by a single
        # batch_expressions / batch_log_expressions call, so the graph does not
        # grow with the number of such entities.  Families are scoped by root
        # (the likelihood's distributions, else every distribution) so that one
        # distribution's pdf only evaluates its own channel's family.
        families = {
            family.name: family
            for family in plan_families(
                graph,
                [*functions, *distributions],
                likelihood_roots or [dist.name for dist in distributions],
            )
        }
        functions_by_name = {function.name: function for function in functions}
        family_outputs: dict[str, TensorVar] = {}

        sorted_nodes = graph.topological_sort()
        # A likelihood-rooted model only needs the distributions that
        # participate in that likelihood and their transitive dependencies.
        # The complete structural graph is still built first so dependency
        # resolution and cycle detection retain their existing behavior.
        # Legacy models without a likelihood continue to build every node.
        if self._likelihood is not None:
            # Workspace construction resolves and validates all likelihood
            # references. Direct Model construction may intentionally leave
            # string references unresolved, however; preserve its historical
            # behavior by pruning from roots that actually exist in the graph.
            resolved_roots = [name for name in likelihood_roots if name in graph]
            active_nodes = graph.ancestor_closure(resolved_roots)
            sorted_nodes = [idx for idx in sorted_nodes if idx in active_nodes]

        for node_idx in sorted_nodes:
            if graph[node_idx]["type"] != "family":
                self._entity_types[graph[node_idx]["name"]] = graph[node_idx]["type"]

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

            # A single long-lived context is grown incrementally as nodes are
            # built.  Topological order guarantees every dependency is already
            # present before a node that uses it is built, so there is no need
            # to reconstruct (and re-validate) a fresh merged context per node.
            context = Context(
                parameters={},
                observables=self._observables,
                views=self._views,
            )

            for node_idx in sorted_nodes:
                node_data = graph[node_idx]
                node_type: Literal[
                    "parameter",
                    "constant",
                    "function",
                    "distribution",
                    "modifier",
                    "family",
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

                if node_type == "parameter":
                    built = self._build_parameter_node(node_name, context)
                    self.parameters[node_name] = built
                    # Vector parameters register a broadcasting view in
                    # self._views during the build; propagate it so later
                    # nodes resolve context[node_name] to the view.
                    if node_name in self._views:
                        context.add_view(node_name, self._views[node_name])
                elif node_type == "constant":
                    built = self._build_constant_node(node_name, constants_map)
                    self.parameters[node_name] = built
                elif node_type == "family":
                    # Builds every member; each member node below picks up
                    # its own expression.  A family is not itself a value, so
                    # it is not added to the context.
                    family = families[node_name]
                    family_outputs.update(
                        self._build_function_family(family, context, functions_by_name)
                        if issubclass(family.kind, BatchableFunction)
                        else self._build_distribution_family(
                            family, context, distributions
                        )
                    )
                    progress_bar.advance(task)
                    continue
                elif node_type == "function":
                    family_built = family_outputs.pop(node_name, None)
                    built = (
                        self._build_function_node(node_name, functions, context)
                        if family_built is None
                        else family_built
                    )
                    self.functions[node_name] = built
                elif node_type == "modifier":
                    built = self._build_modifier_node(node_name, modifiers_map, context)
                    self.modifiers[node_name] = built
                else:  # node_type == "distribution"
                    # A family member's log-space expression is already
                    # recorded by _build_distribution_family.
                    family_built = family_outputs.pop(node_name, None)
                    built = (
                        self._build_distribution_node(node_name, distributions, context)
                        if family_built is None
                        else family_built
                    )
                    self.distributions[node_name] = built
                    # Composite distributions built later read their
                    # components in log space via context.log_density().
                    context.add_log_density(
                        node_name, self.log_distributions[node_name]
                    )

                # Make the freshly built node available to later nodes (any
                # broadcasting view was propagated above for parameter nodes).
                context.add_parameter(node_name, built)

                progress_bar.advance(task)

    @staticmethod
    def _check_family_outputs(
        family: Family, outputs: Mapping[str, TensorVar], members: Sequence[str]
    ) -> None:
        """Raise if a family's batch call left out any of ``members``."""
        missing = [name for name in members if name not in outputs]
        if missing:
            msg = (
                f"{family.kind.__name__} family returned no expression for "
                f"{missing}; it must return one per member."
            )
            raise RuntimeError(msg)

    def _build_function_family(
        self,
        family: Family,
        context: Context,
        functions_by_name: Mapping[str, Function],
    ) -> dict[str, TensorVar]:
        """Build a function family with one ``batch_expressions`` call."""
        outputs = cast(type[BatchableFunction], family.kind).batch_expressions(
            [cast(BatchableFunction, functions_by_name[n]) for n in family.members],
            context,
            functions_by_name,
        )
        self._check_family_outputs(family, outputs, family.members)
        for name in family.members:
            # Named like Evaluable.expression names its result.
            outputs[name].name = name
        return {name: outputs[name] for name in family.members}

    def _build_distribution_family(
        self, family: Family, context: Context, distributions: Distributions
    ) -> dict[str, TensorVar]:
        """Build a distribution family with one ``batch_log_expressions`` call.

        Structural sharing applies as in :meth:`_build_generic_distribution_node`:
        a member identical to an already built distribution, or to an earlier
        member, reuses that distribution's tensors (and name) instead of
        becoming another row of the batch.  Records each member's log-space
        expression and returns the probability-space ones (``pt.exp`` of it).
        """
        built: dict[str, TensorVar] = {}
        first_with_key: dict[str, str] = {}
        aliases: dict[str, str] = {}
        rows: list[BatchableDistribution] = []
        keys: dict[str, str | None] = {}
        for name in family.members:
            dist = cast(BatchableDistribution, distributions[name])
            key = dist.structural_key() if dist.share_identical_expressions else None
            keys[name] = key
            if key is not None and key in self._shared_expressions:
                log_expr, built[name] = self._shared_expressions[key]
                self.log_distributions[name] = log_expr
            elif key is not None and key in first_with_key:
                aliases[name] = first_with_key[key]
            else:
                if key is not None:
                    first_with_key[key] = name
                rows.append(dist)
        if rows:
            outputs = cast(
                type[BatchableDistribution], family.kind
            ).batch_log_expressions(rows, context)
            self._check_family_outputs(family, outputs, [d.name for d in rows])
            for dist in rows:
                log_expr = outputs[dist.name]
                log_expr.name = dist.name
                built[dist.name] = self._register_log_expression(dist.name, log_expr)
                key = keys[dist.name]
                if key is not None:
                    self._shared_expressions[key] = (log_expr, built[dist.name])
        for name, first in aliases.items():
            self.log_distributions[name] = self.log_distributions[first]
            built[name] = built[first]
        return built

    def _compile_expression(
        self,
        name: str,
        expression: TensorVar,
        compiled_functions: dict[str, Callable[..., npt.NDArray[np.float64]]],
        fn_name: str,
    ) -> Callable[..., npt.NDArray[np.float64]]:
        """
        Get or create a compiled PyTensor function for a single expression.

        Shared by :meth:`_get_compiled_function` and
        :meth:`_get_compiled_log_function`, which differ only in which
        expression (probability-space ``self.distributions[name]`` or
        log-space ``self.log_distributions[name]``) they compile and which
        function cache they populate.

        The probability-space expression is always ``pt.exp()`` of the
        log-space one (see :meth:`_build_distribution_node`), so the two
        graphs share identical free inputs by construction.
        ``self._compiled_inputs``/``self._compiled_input_names`` are therefore
        a single cache shared by both compile paths, populated by whichever
        one compiles first; the second path validates (rather than
        recomputes) the ordering against that shared cache.

        Args:
            name (str): Key into ``compiled_functions`` and the shared input caches.
            expression: The distribution expression to compile.
            compiled_functions: Cache of compiled functions to read and populate
                (``self._compiled_functions`` or ``self._compiled_log_functions``).
            fn_name (str): Name to give the compiled PyTensor function.

        Returns:
            Callable: Compiled PyTensor function.

        Raises:
            RuntimeError: If this expression's free inputs do not match the
                already-cached ordering from the other compile path -- this
                would violate the pt.exp()-derivation invariant above.
        """
        if name not in compiled_functions:
            inputs = [
                var
                for var in explicit_graph_inputs([expression])
                if var.name is not None
            ]
            input_names = [var.name for var in inputs if var.name is not None]

            if name in self._compiled_input_names:
                cached_names = self._compiled_input_names[name]
                if input_names != cached_names:
                    msg = (
                        f"pdf/logpdf input mismatch for distribution '{name}': "
                        f"{input_names} vs already-cached {cached_names}. "
                        "The probability- and log-space expressions are expected "
                        "to share identical free inputs by construction."
                    )
                    raise RuntimeError(msg)
            else:
                # Cache the inputs list (and their names) for consistent ordering
                # so pars() and _reorder_params() don't rebuild the name list per
                # evaluation, and so the other compile path can validate against it.
                self._compiled_inputs[name] = cast(list[TensorVar], inputs)
                self._compiled_input_names[name] = input_names

            compiled_functions[name] = cast(
                Callable[..., npt.NDArray[np.float64]],
                function(
                    inputs=inputs,
                    outputs=expression,
                    mode=self.mode,
                    on_unused_input="ignore",
                    name=fn_name,
                    trust_input=True,
                ),
            )
        return compiled_functions[name]

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
        return self._compile_expression(
            name,
            self.distributions[name],
            self._compiled_functions,
            name,
        )

    def _get_compiled_log_function(
        self, name: str
    ) -> Callable[..., npt.NDArray[np.float64]]:
        """
        Get or create a compiled PyTensor function for a distribution's log-PDF.

        Mirrors :meth:`_get_compiled_function` but compiles the log-space
        expression (``self.log_distributions[name]``) instead of the
        probability-space one.  Evaluating in log space keeps
        ``log(prod(exp(...)))`` collapsed so it does not underflow to ``-inf``
        for HistFactory channels with many bins or large expected counts.

        Args:
            name (str): Name of the distribution.

        Returns:
            Callable: Compiled PyTensor function returning the log-PDF.
        """
        return self._compile_expression(
            name,
            self.log_distributions[name],
            self._compiled_log_functions,
            f"log_{name}",
        )

    def pdf_unsafe(
        self,
        name: str,
        **parametervalues: float | list[float] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the PDF with automatic type conversion (convenience method).

        This method automatically converts parameter values to numpy arrays before
        evaluation and, unlike :meth:`pdf`, does not check that each array has
        the dimensionality of its compiled input.  An array passed for a scalar
        parameter is then not rejected: the default C backend reads only its
        first element.  Use this for convenience in testing or interactive use.

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
            key: ensure_array(value) for key, value in parametervalues.items()
        }
        func = self._get_compiled_function(name)
        return func(*self._reorder_params(name, converted_params))

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
            TypeError: If any parameter value is not a numpy array of the
                compiled input's dtype (float64).
            ValueError: If an array's dimensionality differs from its
                compiled input's (e.g. a vector passed for a scalar parameter).

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
        self._check_inputs(name, positional_values)
        return func(*positional_values)

    def logpdf_unsafe(
        self,
        name: str,
        **parametervalues: float | list[float] | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the log PDF with automatic type conversion (convenience method).

        This method automatically converts parameter values to numpy arrays before
        evaluation and, unlike :meth:`logpdf`, does not check that each array
        has the dimensionality of its compiled input.  An array passed for a
        scalar parameter is then not rejected: the default C backend reads only
        its first element.  Use this for convenience in testing or interactive
        use.

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
        # Convert all parameter values to numpy arrays
        converted_params = {
            key: ensure_array(value) for key, value in parametervalues.items()
        }
        func = self._get_compiled_log_function(name)
        return func(*self._reorder_params(name, converted_params))

    def logpdf(
        self, name: str, **parametervalues: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluates the natural logarithm of the PDF.

        This method requires all parameter values to be numpy arrays with dtype float64.
        For automatic type conversion, use :meth:`logpdf_unsafe` instead.

        The logarithm is evaluated in log space via a compiled log-PDF function
        (``log(prod(exp(...)))`` stays collapsed), so the result remains finite
        for HistFactory channels where the probability-space ``pdf`` underflows
        to ``0.0``.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues: Values for each parameter as numpy arrays.

        Returns:
            npt.NDArray[np.float64]: The log of the PDF.

        Raises:
            TypeError: If any parameter value is not a numpy array of the
                compiled input's dtype (float64).
            ValueError: If an array's dimensionality differs from its
                compiled input's (e.g. a vector passed for a scalar parameter).

        See Also:
            :meth:`logpdf_unsafe`: Convenience version with automatic type conversion
            :meth:`pdf`: PDF with strict type checking

        Example:
            >>> import numpy as np
            >>> model.logpdf("gauss", x=np.array(1.5), mu=np.array(0.0), sigma=np.array(1.0))  # doctest: +SKIP
        """
        func = self._get_compiled_log_function(name)
        positional_values = self._reorder_params(name, parametervalues)
        self._check_inputs(name, positional_values)
        return func(*positional_values)

    def pars(self, name: str) -> list[str]:
        """
        Get the ordered list of input parameter names for a distribution.

        This method returns the parameter names in the exact order expected
        by the compiled PDF function. Since the probability- and log-space
        expressions for a distribution share identical free inputs by
        construction (:meth:`pdf` evaluates ``pt.exp()`` of exactly what
        :meth:`logpdf` evaluates), this same ordering is valid for both
        :meth:`pdf` and :meth:`logpdf`.

        Args:
            name: Distribution name

        Returns:
            List of parameter names in the order expected by pdf() and logpdf()

        Example:
            >>> model.pars("model_singlechannel") # doctest: +SKIP
            ['uncorr_bkguncrt_1', 'uncorr_bkguncrt_0', 'model_singlechannel_observed', 'mu', 'Lumi']
        """
        if name not in self._compiled_input_names:
            # Trigger compilation to populate the shared cache.
            self._get_compiled_function(name)
        return self._compiled_input_names[name]

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

    def _check_inputs(self, name: str, values: Sequence[object]) -> None:
        """Raise if an argument is not an array of its compiled input's dtype and ndim.

        The compiled functions skip PyTensor's input validation
        (``trust_input=True``): the C backend reads only the first element of
        an array passed for a scalar input instead of raising, and rejects
        other types with messages that do not name the parameter.

        Args:
            name: Distribution name; its function must already be compiled.
            values: Parameter values in the order of :meth:`pars`.

        Raises:
            TypeError: If an argument is not a numpy array of the input's dtype.
            ValueError: If an array's dimensionality differs from the input's.
        """
        for var, value in zip(self._compiled_inputs[name], values, strict=True):
            if not isinstance(value, np.ndarray) or value.dtype != var.dtype:
                got = (
                    f"{value.dtype} array"
                    if isinstance(value, np.ndarray)
                    else type(value).__name__
                )
                msg = (
                    f"Parameter '{var.name}' of '{name}' expects a numpy array of "
                    f"dtype {var.dtype}, got {got}; use np.asarray(value, "
                    f"dtype=np.{var.dtype}) or the _unsafe variant."
                )
                raise TypeError(msg)
            if value.ndim != var.ndim:
                msg = (
                    f"Parameter '{var.name}' of '{name}' expects a "
                    f"{var.ndim}-dimensional array, got shape {value.shape}."
                )
                raise ValueError(msg)

    def _reorder_params(
        self,
        name: str,
        params: Mapping[str, npt.NDArray[np.float64]],
    ) -> list[npt.NDArray[np.float64]]:
        """
        Reorder parameters to match the expected input order for a distribution.

        Used by both :meth:`pdf` and :meth:`logpdf`: :meth:`pars` returns the
        same ordering for either compiled function.

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
        *,
        op_params: OpParamMode | Mapping[str, OpParamMode] = DEFAULT_OP_PARAMS,
        show_id: bool = False,
        show_dtype: bool = False,
        show_shape: bool = False,
        const_arrays: ConstArrayMode = "elide",
        const_array_threshold: int = 8,
    ) -> str:
        """
        Visualize the computation graph for a distribution.

        The defaults favor a figure that's readable at a glance over one that
        exposes every implementation detail: dtype/shape annotations and
        toposort-index ids are hidden, single-input broadcasting plumbing ops
        (``ExpandDims``, ``DimShuffle``) are spliced out of the drawing, and
        constant arrays larger than ``const_array_threshold`` collapse to a
        ``const[shape]`` placeholder. Pass the corresponding kwarg to bring any
        of that detail back.

        Args:
            name (str): Distribution name.
            fmt (str): Output format ('svg', 'png', 'pdf'). Defaults to 'svg'.
            outfile (str | None): Output filename. If None, uses '{name}_graph.{fmt}'.
            path (str | None): Directory path for output. If None, uses current working directory.
            op_params: Per-op-name display mode, or one mode applied to every
                op. ``"orig"`` keeps the op's full label (e.g. ``Sum{axis=0}``);
                ``"none"`` drops the parameter block (``Sum``); ``"elide"``
                removes the node from the drawing entirely, rewiring its
                consumers to its own (sole) input - only sound for
                single-input ops. Op names absent from a mapping default to
                ``"orig"``. Defaults to eliding ``ExpandDims``/``DimShuffle``
                only.
            show_id: Append each op's toposort index to its label
                (e.g. ``Add id=9``). Defaults to ``False``.
            show_dtype: Keep the dtype portion of PyTensor's type annotations
                (e.g. ``Matrix(float32)``). Defaults to ``False``.
            show_shape: Keep the shape portion of PyTensor's type annotations
                (e.g. ``Matrix(shape=(1, 1))``). Defaults to ``False``.
            const_arrays: How to render constant arrays larger than
                ``const_array_threshold`` elements. ``"orig"`` leaves
                pydotprint's own (possibly mid-token-truncated) value dump
                untouched; ``"truncate"`` shows a deliberate
                ``[v0, v1, v2, ...] (shape)`` preview; ``"elide"`` collapses
                it to ``const[shape]``. Defaults to ``"elide"``.
            const_array_threshold: Element count above which ``const_arrays``
                applies. Defaults to 8.

        Returns:
            str: Path to the generated visualization file.

        Raises:
            ImportError: If pydot is not installed.
        """
        try:
            from pytensor.printing import (  # noqa: F401, PLC0415  # pylint: disable=import-outside-toplevel,unused-import
                pydotprint as _pydotprint_availability_probe,
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

        return render_graph(
            dist,
            filename,
            fmt,
            op_params=op_params,
            show_id=show_id,
            show_dtype=show_dtype,
            show_shape=show_shape,
            const_arrays=const_arrays,
            const_array_threshold=const_array_threshold,
        )

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

    def entity_type(self, name: str) -> EntityType:
        """
        Get the kind of entity a name refers to.

        Args:
            name (str): Name of a distribution, function, parameter, constant,
                or HistFactory modifier in the model.

        Returns:
            EntityType: One of ``"distribution"``, ``"function"``,
            ``"parameter"``, ``"constant"``, or ``"modifier"``.

        Raises:
            ValueError: If the model has no entity with this name.

        Examples:
            >>> import pyhs3
            >>> ws = pyhs3.Workspace(
            ...     metadata={"hs3_version": "0.2"},
            ...     distributions=[
            ...         {
            ...             "name": "gauss",
            ...             "type": "gaussian_dist",
            ...             "x": "x",
            ...             "mean": "mu",
            ...             "sigma": 1.0,
            ...         }
            ...     ],
            ...     parameter_points=[
            ...         {
            ...             "name": "default_values",
            ...             "parameters": [
            ...                 {"name": "x", "value": 0.0},
            ...                 {"name": "mu", "value": 0.0},
            ...             ],
            ...         }
            ...     ],
            ... )
            >>> model = ws.model(0, progress=False)
            >>> model.entity_type("gauss")
            'distribution'
            >>> model.entity_type("mu")
            'parameter'
        """
        try:
            return self._entity_types[name]
        except KeyError:
            msg = f"Entity '{name}' not found in model"
            raise ValueError(msg) from None

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

        compile_info = f"\n    Mode: {self.mode}\n    Compiled: {'Yes' if name in self._compiled_functions else 'No'}"

        return f"""Distribution '{name}':
    Input variables: {len(inputs)}
    Graph operations: {len(applies)}
    Operation types: {dict(sorted(op_types.items()))}{compile_info}
"""
