"""
Parameter context of a model.

A :class:`ParameterContext` is everything a :class:`~pyhs3.Model` needs to know
about its parameters, derived from one analysis (or from an explicit parameter
set and domain): the referenced parameter point (starting values and which
parameters are ``const``), the domain bounds, and the parameters of interest.
Build one with :meth:`pyhs3.Workspace.parameter_context`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyhs3.axes import DomainCoordinateAxis
from pyhs3.domains import Domain, ProductDomain
from pyhs3.parameter_points import ParameterSet
from pyhs3.typing.aliases import DomainBounds


@dataclass(frozen=True)
class ParameterContext:
    """Resolved parameter context: starting values, ``const`` set, bounds and POIs.

    The context is fixed when a model is built: ``const`` parameters become
    graph constants, every other parameter a graph input.

    Attributes:
        parameter_set: The referenced parameter point (an analysis ``init``
            or an explicit override); gives the starting values and the
            ``const`` flags.
        domain: The domain giving the parameter bounds (for an analysis, its
            domains merged into one).
        parameters_of_interest: Names of the parameters of interest, in the
            order the analysis lists them; empty when there is no analysis or
            it lists none.

    Examples:
        >>> from pyhs3.domains import ProductDomain
        >>> from pyhs3.parameter_points import ParameterPoint, ParameterSet
        >>> context = ParameterContext(
        ...     parameter_set=ParameterSet(
        ...         name="init",
        ...         parameters=[
        ...             ParameterPoint(name="mu", value=1.0),
        ...             ParameterPoint(name="lumi", value=1.0, const=True),
        ...         ],
        ...     ),
        ...     domain=ProductDomain(
        ...         name="dom", axes=[{"name": "mu", "min": 0.0, "max": 5.0}]
        ...     ),
        ...     parameters_of_interest=("mu",),
        ... )
        >>> context.values
        {'mu': 1.0, 'lumi': 1.0}
        >>> sorted(context.const)
        ['lumi']
        >>> context.bounds
        {'mu': (0.0, 5.0)}
    """

    parameter_set: ParameterSet
    domain: Domain = field(default_factory=lambda: ProductDomain(name="default"))
    parameters_of_interest: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """Name of the referenced parameter point."""
        return self.parameter_set.name

    @property
    def values(self) -> dict[str, float]:
        """Starting value of every parameter in the parameter point, ``const`` ones included."""
        return {point.name: float(point.value) for point in self.parameter_set}

    @property
    def const(self) -> frozenset[str]:
        """Names of the parameters that are ``const`` in the parameter point."""
        return frozenset(point.name for point in self.parameter_set if point.const)

    @property
    def bounds(self) -> dict[str, DomainBounds]:
        """``(lower, upper)`` of every coordinate axis of the domain; ``-inf``/``inf`` for an open side.

        Constant axes have no bounds and are left out.
        """
        if not isinstance(self.domain, ProductDomain):
            return {}
        return {
            axis.name: (axis.min, axis.max)
            for axis in self.domain.axes
            if isinstance(axis, DomainCoordinateAxis)
        }
