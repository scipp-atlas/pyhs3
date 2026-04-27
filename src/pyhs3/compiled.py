"""
Joint compiled NLL for multi-channel HS3 analyses.

Produced by :meth:`~pyhs3.analyses.Analysis.compile`. The
:class:`CompiledLikelihood` holds a single PyTensor scalar ``nll_expr``
(``-2 log L``) with observed data frozen as ``pt.constant`` nodes, leaving
only physics and nuisance parameters as symbolic graph inputs suitable for
gradient-based optimisation or JAX transpilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import explicit_graph_inputs

from pyhs3.context import Context
from pyhs3.data import UnbinnedData
from pyhs3.distributions import Distributions
from pyhs3.functions import Functions
from pyhs3.model import Model
from pyhs3.parameter_points import ParameterSet
from pyhs3.typing.aliases import TensorVar

if TYPE_CHECKING:
    from pyhs3.analyses import Analysis
    from pyhs3.transpile import JaxifiedGraph


@dataclass(frozen=True)
class CompiledLikelihood:
    """Joint compiled NLL for a multi-channel analysis.

    Built by :meth:`~pyhs3.analyses.Analysis.compile`. The ``nll_expr`` is a
    PyTensor scalar (``-2 log L``) with observed data frozen as
    ``pt.constant`` nodes; only physics/nuisance parameters remain as
    symbolic graph inputs.

    Attributes:
        nll_expr: The ``-2 log L`` PyTensor expression.
        free_parameters: Names of parameters not marked ``const`` in the
            analysis's parameter point.
        fixed_parameters: Names of parameters marked ``const``.
    """

    nll_expr: TensorVar
    free_parameters: tuple[str, ...]
    fixed_parameters: tuple[str, ...]

    def to_jax(self) -> JaxifiedGraph:
        """Transpile ``nll_expr`` to a JAX-callable :class:`~pyhs3.transpile.JaxifiedGraph`."""
        from pyhs3.transpile import jaxify  # noqa: PLC0415

        return jaxify(self.nll_expr)

    @classmethod
    def from_analysis(cls, analysis: Analysis) -> CompiledLikelihood:
        """Build from an FK-resolved :class:`~pyhs3.analyses.Analysis`.

        The analysis must have been loaded through a
        :class:`~pyhs3.workspace.Workspace` so that its ``_workspace``
        backref is set.
        """
        ws = analysis._workspace
        if ws is None:
            msg = "Analysis.compile() requires a workspace backref; load via Workspace.load()"
            raise RuntimeError(msg)

        likelihood = analysis.likelihood
        if isinstance(likelihood, str):
            msg = "Analysis likelihood FK must be resolved; load via Workspace.load()"
            raise RuntimeError(msg)

        # Collect observable bounds from all likelihood data channels.
        observables_bounds: dict[str, tuple[float, float]] = {}
        for datum in likelihood.data:
            if isinstance(datum, UnbinnedData):
                for axis in datum.axes:
                    observables_bounds[axis.name] = (axis.min, axis.max)

        # Resolve the parameter set for init values / const flags.
        param_set: ParameterSet | None = None
        if analysis.init and ws.parameter_points:
            param_set = ws.parameter_points.get(analysis.init)

        # Build Model to handle dependency-graph topological sort for
        # functions/modifiers.  This assumes function expressions do not
        # reference observable variables (true for all standard HistFactory
        # and HS3 analyses where functions depend only on nuisance parameters).
        domain = analysis.domains[0]
        if isinstance(domain, str):
            msg = "Analysis domains must be FK-resolved before compile()"
            raise RuntimeError(msg)
        model = Model(
            parameterset=param_set or ParameterSet(name="default", parameters=[]),
            distributions=ws.distributions or Distributions([]),
            domain=domain,
            functions=ws.functions or Functions([]),
            progress=False,
            observables=observables_bounds,
        )

        lp_terms: list[TensorVar] = []

        for dist, datum in zip(likelihood.distributions, likelihood.data, strict=False):
            if isinstance(dist, str) or not isinstance(datum, UnbinnedData):
                continue

            entries_arr = np.asarray(datum.entries, dtype=np.float64)

            # Keep observables SYMBOLIC when calling log_expression so that
            # the Gauss-Legendre normalization quadrature (which evaluates at
            # 64 points) can shape-infer freely.  After we have the full
            # symbolic expression, substitute the symbolic observable vectors
            # with frozen pt.constant nodes carrying the actual event data.
            context_data: dict[str, TensorVar] = {
                **model.parameters,
                **model.functions,
                **model.modifiers,
            }
            obs_bounds_channel: dict[str, tuple[TensorVar, TensorVar]] = {}
            for axis in datum.axes:
                obs_bounds_channel[axis.name] = (
                    pt.constant(axis.min),
                    pt.constant(axis.max),
                )

            context = Context(parameters=context_data, observables=obs_bounds_channel)
            # Per-event log-probs (shape: (n_events,)) — observable still symbolic.
            log_probs_sym: TensorVar = dist.log_expression(context)

            # Substitute symbolic observable vectors with frozen event data.
            replacements: dict[TensorVar, TensorVar] = {
                model.parameters[axis.name]: pt.constant(
                    entries_arr[:, ax_idx], dtype="float64"
                )
                for ax_idx, axis in enumerate(datum.axes)
                if axis.name in model.parameters
            }
            (log_probs,) = graph_replace([log_probs_sym], replacements)  # type: ignore[arg-type]

            if datum.weights is not None:
                weights = pt.constant(np.asarray(datum.weights, dtype=np.float64))
                lp_terms.append(pt.sum(weights * log_probs))  # type: ignore[no-untyped-call]
            else:
                lp_terms.append(pt.sum(log_probs))  # type: ignore[no-untyped-call]

        # Auxiliary distributions are constraint scalars — no event summing.
        if likelihood.aux_distributions:
            for aux_name in likelihood.aux_distributions:
                if aux_name in model.distributions:
                    lp_terms.append(pt.log(model.distributions[aux_name]))

        if lp_terms:
            total_log_prob: TensorVar = lp_terms[0]
            for term in lp_terms[1:]:
                total_log_prob = total_log_prob + term
        else:
            total_log_prob = pt.constant(np.float64(0.0))

        nll_expr: TensorVar = -2.0 * total_log_prob

        # Classify symbolic inputs as free vs fixed (const flag in ParameterSet).
        free_params: list[str] = []
        fixed_params: list[str] = []
        for var in explicit_graph_inputs([nll_expr]):
            name = var.name
            if name is None:
                continue
            pp = param_set.get(name) if param_set else None
            if pp and pp.const:
                fixed_params.append(name)
            else:
                free_params.append(name)

        return cls(
            nll_expr=nll_expr,
            free_parameters=tuple(sorted(free_params)),
            fixed_parameters=tuple(sorted(fixed_params)),
        )
