"""
HS3 Axis implementations.

Provides Pydantic classes for handling HS3 axis specifications
including unbinned axes (with min/max bounds) and binned axes
(with regular or irregular binning).
"""

from __future__ import annotations

from itertools import pairwise
from math import isfinite
from typing import Annotated, Any, Literal

import hist
import numpy as np
from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

from pyhs3.collections import NamedCollection, NamedModel
from pyhs3.exceptions import custom_error_msg


class Axis(NamedModel):
    """
    Base axis specification for data coordinates.

    Attributes:
        name: Name of the axis/variable
    """


class BoundedAxis(Axis):
    """
    Axis with required finite min/max bounds.

    Attributes:
        name: Name of the axis/variable
        min: Minimum value (required)
        max: Maximum value (required)
    """

    min: float = Field(..., repr=False)
    max: float = Field(..., repr=False)

    @model_validator(mode="after")
    def check_min_le_max(self) -> BoundedAxis:
        """Validate that max >= min."""
        if self.max < self.min:
            msg = f"{type(self).__name__} '{self.name}': max ({self.max}) must be >= min ({self.min})"
            raise ValueError(msg)
        return self


class UnbinnedAxis(BoundedAxis):
    """
    Axis for unbinned data with required finite bounds.

    Attributes:
        name: Name of the axis/variable
        min: Minimum value (required, inherited from BoundedAxis)
        max: Maximum value (required, inherited from BoundedAxis)
    """

    def to_hist(self) -> Any:
        """
        Convert this axis to a hist.axis object.

        This is a base implementation that should be overridden by subclasses
        that have specific binning information (like BinnedAxis).

        Returns:
            A hist.axis object

        Raises:
            ValueError: If axis has insufficient binning information
        """
        msg = f"UnbinnedAxis '{self.name}' does not have binning information for histogram conversion"
        raise ValueError(msg)


class ConstantAxis(Axis):
    """
    Axis for constant data.

    Alias for Axis with required const field.

    As a domain axis, this currently has no effect on model construction:
    :meth:`~pyhs3.domains.Domain.get` returns the unbounded default for any
    axis that isn't a :class:`DomainCoordinateAxis`, so a parameter named by
    a ``ConstantAxis`` is treated as unbounded, the same as one the domain
    doesn't name at all. To fix a parameter's value, set ``const=True`` on
    its :class:`~pyhs3.parameter_points.ParameterPoint` instead.

    Attributes:
        name: Name of the axis/variable
        const: true (required)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    const: Literal[True] = Field(True, repr=False, init=False)

    @field_validator("const", mode="before")
    @classmethod
    def validate_const(cls, value: Any) -> Any:
        """Require the JSON boolean ``true``, not a truthy numeric value."""
        if value is not True or not isinstance(value, bool):
            msg = "const must be the boolean true"
            raise ValueError(msg)
        return value


class RegularAxis(BoundedAxis):
    """
    Attributes:
        name: Name of the axis/variable
        min: Minimum value (inherited from BoundedAxis)
        max: Maximum value (inherited from BoundedAxis)
        nbins: Number of bins (for regular binning)
    """

    model_config = ConfigDict(extra="forbid")

    nbins: int = Field(repr=False)

    @field_validator("min", "max", mode="before")
    @classmethod
    def validate_finite_bounds(cls, value: Any) -> Any:
        """Require finite numeric regular-binning bounds."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            msg = "regular axis bounds must be finite numeric values"
            raise ValueError(msg)
        return value

    @field_validator("nbins", mode="before")
    @classmethod
    def validate_nbins(cls, value: Any) -> int:
        """Require a positive, non-boolean integer bin count.

        ROOT also accepts integer-valued JSON floats, so values such as
        ``4.0`` remain valid while booleans, strings, fractions and non-finite
        numbers are rejected.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = "nbins must be a positive integer"
            raise ValueError(msg)
        if not isfinite(value) or value != int(value) or value > np.iinfo(np.int32).max:
            msg = "nbins must be a positive integer"
            raise ValueError(msg)
        return int(value)

    @model_validator(mode="after")
    def check_binning(self) -> RegularAxis:
        """Validate finite, increasing bounds and a positive bin count."""
        if self.nbins <= 0:
            msg = f"RegularAxis '{self.name}' must have positive number of bins, got {self.nbins}"
            raise ValueError(msg)
        if not isfinite(self.min) or not isfinite(self.max) or self.max <= self.min:
            msg = (
                f"RegularAxis '{self.name}' bounds must be finite and increasing, "
                f"got min={self.min}, max={self.max}"
            )
            raise ValueError(msg)
        return self

    @property
    def edges(self) -> list[float]:
        """Get the bin edges for this axis.

        Returns:
            List of bin edges. Generates edges using linspace.
        """
        return list(np.linspace(self.min, self.max, self.nbins + 1))

    def to_hist(self) -> hist.axis.Regular:
        """
        Convert this axis to a hist.axis object.

        Returns:
            A hist.axis.Regular object
        """
        return hist.axis.Regular(self.nbins, self.min, self.max, name=self.name)


class IrregularAxis(Axis):
    """
    Attributes:
        name: Name of the axis/variable
        edges: Bin edges array (length n+1)
    """

    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    edges: list[float] = Field(repr=False)
    v_min: float | None = Field(
        default=None, alias="min", repr=False, exclude_if=lambda value: value is None
    )
    v_max: float | None = Field(
        default=None, alias="max", repr=False, exclude_if=lambda value: value is None
    )

    @field_validator("v_min", "v_max", mode="before")
    @classmethod
    def validate_redundant_bound(cls, value: Any) -> Any:
        """Require redundant ROOT bounds to be finite JSON numbers."""
        if value is None:
            return value
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            msg = "irregular axis bounds must be finite numeric values"
            raise ValueError(msg)
        return value

    @field_validator("edges", mode="before")
    @classmethod
    def validate_finite_edges(cls, value: Any) -> Any:
        """Reject non-numeric, boolean, and non-finite bin edges."""
        if isinstance(value, (list, tuple)):
            for edge in value:
                if (
                    isinstance(edge, bool)
                    or not isinstance(edge, (int, float))
                    or not isfinite(edge)
                ):
                    msg = "edges must contain only finite numeric values"
                    raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_binning(self) -> IrregularAxis:
        """Ensure proper binning specification for binned data."""
        if len(self.edges) < 2:
            msg = f"IrregularAxis '{self.name}' must have at least 2 edges"
            raise ValueError(msg)
        # Check that edges are in ascending order
        for prev, curr in pairwise(self.edges):
            if curr <= prev:
                msg = f"IrregularAxis '{self.name}' edges must be in ascending order"
                raise ValueError(msg)
        if self.v_min is not None and self.v_min != self.edges[0]:
            msg = (
                f"IrregularAxis '{self.name}' min ({self.v_min}) must match "
                f"its first edge ({self.edges[0]})"
            )
            raise ValueError(msg)
        if self.v_max is not None and self.v_max != self.edges[-1]:
            msg = (
                f"IrregularAxis '{self.name}' max ({self.v_max}) must match "
                f"its last edge ({self.edges[-1]})"
            )
            raise ValueError(msg)
        return self

    @property
    def min(self) -> float:
        """Return lower edge."""
        return self.edges[0]

    @property
    def max(self) -> float:
        """Return upper edge."""
        return self.edges[-1]

    @property
    def nbins(self) -> int:
        """Get the nbins for this axis.

        Returns:
            Number of bins.
        """
        return len(self.edges) - 1

    def to_hist(self) -> hist.axis.Variable:
        """
        Convert this axis to a hist.axis object.

        Returns:
            A hist.axis.Variable object
        """
        return hist.axis.Variable(self.edges, name=self.name)


class DomainCoordinateAxis(Axis):
    """
    Axis for domain coordinates with optional bounds.

    Represents a coordinate axis in a parameter domain, where bounds
    may be fully specified, partially specified, or unbounded (infinite).

    Attributes:
        name: Name of the axis/variable
        min: Minimum value (optional, defaults to -inf)
        max: Maximum value (optional, defaults to +inf)

    Examples:
        Create an unbounded domain axis:

        >>> from pyhs3.axes import DomainCoordinateAxis
        >>> axis = DomainCoordinateAxis(name="x")
        >>> axis
        DomainCoordinateAxis(x ∈ (-∞, +∞))

        Create a lower-bounded domain:

        >>> axis = DomainCoordinateAxis(name="x", min=-5)
        >>> axis
        DomainCoordinateAxis(x ∈ [-5, +∞))

        Create an upper-bounded domain:

        >>> axis = DomainCoordinateAxis(name="x", max=10)
        >>> axis
        DomainCoordinateAxis(x ∈ (-∞, 10])

        Create a fully bounded domain:

        >>> axis = DomainCoordinateAxis(name="x", min=0, max=1)
        >>> axis
        DomainCoordinateAxis(x ∈ [0, 1])

        Integers are displayed without trailing .0:

        >>> axis = DomainCoordinateAxis(name="x", min=0.0, max=5.0)
        >>> axis
        DomainCoordinateAxis(x ∈ [0, 5])
    """

    model_config = ConfigDict(serialize_by_alias=True, extra="forbid")

    v_min: float | None = Field(
        default=None, alias="min", repr=False, exclude_if=lambda v: v is None
    )
    v_max: float | None = Field(
        default=None, alias="max", repr=False, exclude_if=lambda v: v is None
    )

    @field_validator("v_min", "v_max", mode="before")
    @classmethod
    def validate_finite_bound(cls, value: Any) -> Any:
        """Require explicitly supplied bounds to be finite JSON numbers."""
        if value is None:
            return value
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            msg = "domain axis bounds must be finite numeric values or null"
            raise ValueError(msg)
        return value

    @property
    def min(self) -> float:
        """Returns defined minimum or (negative) :data:`np.inf <numpy.inf>`"""
        return -np.inf if self.v_min is None else self.v_min

    @property
    def max(self) -> float:
        """Returns defined maximum or (positive) :data:`np.inf <numpy.inf>`"""
        return np.inf if self.v_max is None else self.v_max

    @model_validator(mode="after")
    def check_min_le_max(self) -> DomainCoordinateAxis:
        """Validate that max >= min when both are provided."""
        if self.max < self.min:
            msg = f"DomainCoordinateAxis '{self.name}': max ({self.max}) must be >= min ({self.min})"
            raise ValueError(msg)
        return self

    def __repr__(self) -> str:
        # Determine interval brackets
        left_bracket = "[" if self.v_min is not None else "("
        right_bracket = "]" if self.v_max is not None else ")"

        # Determine displayed bounds
        min_str = f"{self.v_min:g}" if self.v_min is not None else "-∞"
        max_str = f"{self.v_max:g}" if self.v_max is not None else "+∞"

        return f"DomainCoordinateAxis({self.name} ∈ {left_bracket}{min_str}, {max_str}{right_bracket})"


def _binned_axis_discriminator(v: Any) -> str | None:
    if isinstance(v, dict):
        if "edges" in v and "nbins" not in v:
            return "irregular"
        if "nbins" in v and "edges" not in v:
            return "regular"
        return None

    # Already-constructed model case
    if isinstance(v, IrregularAxis):
        return "irregular"
    if isinstance(v, RegularAxis):
        return "regular"

    return None


BinnedAxis = Annotated[
    (
        Annotated[RegularAxis, Tag("regular")]
        | Annotated[IrregularAxis, Tag("irregular")]
    ),
    Discriminator(_binned_axis_discriminator),
    custom_error_msg(
        {
            "union_tag_not_found": "Unknown axis {input}'. You must specify either regular binning (nbins/min/max) or irregular binning (edges).",
            "missing": "{input_value['name']} is missing {loc}",
        }
    ),
]


class BinnedAxes(NamedCollection[BinnedAxis]):
    """
    Collection of binned axis.
    """

    def get_total_bins(self) -> int:
        """Calculate total number of bins across all axes."""
        total = 1
        for axis in self:
            total *= axis.nbins
        return total


def _domain_axis_discriminator(v: Any) -> str | None:
    """Select a product-domain axis variant from its structural wire fields."""
    if isinstance(v, dict):
        tags = [
            tag
            for field, tag in (
                ("const", "constant"),
                ("nbins", "regular"),
                ("edges", "irregular"),
            )
            if field in v
        ]
        if len(tags) > 1:
            return None
        return tags[0] if tags else "coordinate"

    if isinstance(v, ConstantAxis):
        return "constant"
    if isinstance(v, RegularAxis):
        return "regular"
    if isinstance(v, IrregularAxis):
        return "irregular"
    if isinstance(v, DomainCoordinateAxis):
        return "coordinate"

    return None


DomainAxis = Annotated[
    (
        Annotated[DomainCoordinateAxis, Tag("coordinate")]
        | Annotated[ConstantAxis, Tag("constant")]
        | Annotated[RegularAxis, Tag("regular")]
        | Annotated[IrregularAxis, Tag("irregular")]
    ),
    Discriminator(_domain_axis_discriminator),
    custom_error_msg(
        {
            "union_tag_not_found": (
                "Unknown product-domain axis {input}. Axis definitions must use "
                "exactly one of const, nbins, edges, or coordinate bounds."
            )
        }
    ),
]


class UnbinnedAxes(NamedCollection[UnbinnedAxis]):
    """
    Collection of UnbinnedAxis.
    """

    root: list[UnbinnedAxis] = Field(default_factory=list)


class Axes(NamedCollection[BinnedAxis | UnbinnedAxis]):
    """
    Collection of BinnedAxis | UnbinnedAxis.
    """

    root: list[BinnedAxis | UnbinnedAxis] = Field(default_factory=list)


class DomainAxes(NamedCollection[DomainAxis]):
    """
    Collection of BinnedAxis | UnbinnedAxis.
    """

    root: list[DomainAxis] = Field(default_factory=list)
