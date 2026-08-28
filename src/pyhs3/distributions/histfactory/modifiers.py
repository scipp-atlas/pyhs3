from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, Literal, cast

import numpy as np
import pytensor.tensor as pt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    model_validator,
)
from pytensor.configdefaults import config

from pyhs3.context import Context
from pyhs3.distributions.basic import GaussianDist, PoissonDist
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.histfactory import interpolations
from pyhs3.distributions.histfactory.data import SampleData

# Import existing distributions for constraint terms
from pyhs3.exceptions import custom_error_msg
from pyhs3.networks import HasDependencies
from pyhs3.typing.aliases import TensorVar


class ModifierData(BaseModel):
    """Base class for modifier data."""


class NormSysData(ModifierData):
    """Data for normsys modifier."""

    model_config = ConfigDict(extra="forbid")

    hi: float
    lo: float
    # Narrow HS3 0.2 import adapter. The channel validator translates this to
    # a structured modifier descriptor and this legacy field is never emitted.
    interpolation: Literal["code1", "code4"] | None = Field(
        default=None,
        exclude=True,
    )


class HistoSysDataContents(BaseModel):
    """Contents data for histosys modifier."""

    contents: list[float]


class HistoSysData(ModifierData):
    """Data for histosys modifier."""

    model_config = ConfigDict(extra="forbid")

    hi: HistoSysDataContents
    lo: HistoSysDataContents
    # Narrow HS3 0.2 import adapter; see NormSysData.interpolation.
    interpolation: Literal["code0", "code2", "code4p"] | None = Field(
        default=None,
        exclude=True,
    )

    @model_validator(mode="after")
    def validate_lengths(self) -> HistoSysData:
        """Validate that hi and lo contents have the same length."""
        if len(self.hi.contents) != len(self.lo.contents):
            msg = f"histosys data contents for hi ({len(self.hi.contents)}) and lo ({len(self.lo.contents)}) must have same length"
            raise ValueError(msg)
        return self


class ShapeSysData(ModifierData):
    """Data for shapesys modifier."""

    vals: list[float]


class StatErrorData(ModifierData):
    """Data for staterror modifier."""

    uncertainties: list[float]


# base modifier
class Modifier(BaseModel, HasDependencies, ABC):
    """Base class for modifier effects (multiplicative or additive)."""

    name: str
    type: str
    application: Literal["additive", "multiplicative"] = Field(exclude=True)

    @property
    def is_multiplicative(self) -> bool:
        """Whether this modifier applies multiplicatively to rates."""
        return self.application == "multiplicative"

    @property
    def is_additive(self) -> bool:
        """Whether this modifier applies additively to rates."""
        return self.application == "additive"

    @property
    @abstractmethod
    def dependencies(self) -> set[str]:
        """Return parameter names this modifier depends on."""

    @abstractmethod
    def expression(self, context: Context) -> TensorVar:
        """Return the modifier's contribution (additive term or multiplicative factor)."""


class HasConstraint(ABC):
    """Base class for modifiers that can have constraint terms."""

    constraint: str | None

    @abstractmethod
    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term for this modifier (probability space)."""

    @abstractmethod
    def log_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term for this modifier (log space).

        Log-space counterpart of :meth:`make_constraint`: evaluates the same
        constraint distribution(s) via their analytic ``log_likelihood``
        instead of taking ``pt.log`` of the probability-space result, so the
        constraint stays finite where the probability-space value would
        underflow to 0.0.
        """


class SingleParamConstraint(HasConstraint, ABC):
    """Mixin for modifiers constrained by a named workspace distribution.

    The HS3 field is a foreign-key-like reference, not a distribution-family
    selector.  The referenced distribution is built as an ordinary dependency
    of the HistFactory channel.  :class:`pyhs3.model.Model` reuses its analytic
    ``log_expression`` when assembling a model; the direct methods below use
    the probability-space expression available in a bare :class:`Context`.
    """

    name: str
    parameter: str
    constraint: str | None

    @property
    def dependencies(self) -> set[str]:
        """Return the nuisance parameter and optional constraint distribution."""
        result = {self.parameter}
        if self.constraint is not None:
            result.add(self.constraint)
        return result

    def make_constraint(self, context: Context, _: SampleData) -> TensorVar:
        """Return the referenced distribution expression in probability space."""
        if self.constraint is None:
            return pt.constant(1.0)
        return context[self.constraint]

    def log_constraint(self, context: Context, _: SampleData) -> TensorVar:
        """Return the log of the referenced expression for direct channel use.

        Model construction replaces this with the referenced distribution's
        pre-built analytic log expression, avoiding probability underflow.
        """
        if self.constraint is None:
            return pt.constant(0.0)
        return cast("TensorVar", pt.log(context[self.constraint]))


# Parameterized modifier base (single parameter)
class ParameterModifier(Modifier, ABC):
    """Base for modifiers that use a single parameter name."""

    parameter: str

    @property
    def dependencies(self) -> set[str]:
        """Return parameter names this modifier depends on."""
        return {self.parameter}

    @property
    @abstractmethod
    def auxdata(self) -> float:
        """Auxiliary data value associated with this modifier (single float)."""

    @abstractmethod
    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply this modifier to the given rates tensor."""


# Multi-parameter modifier base (per-bin parameters)
class ParametersModifier(Modifier, ABC):
    """Base for modifiers that use multiple parameter names (one per bin)."""

    parameters: list[str]

    @property
    def dependencies(self) -> set[str]:
        """Return parameter names this modifier depends on."""
        return set(self.parameters)

    @property
    @abstractmethod
    def auxdata(self) -> list[float]:
        """Auxiliary data values associated with this modifier (list of floats)."""

    def expression(self, context: Context) -> TensorVar:
        """Return stacked tensor of per-bin parameter values."""
        return cast("TensorVar", pt.stack([context[name] for name in self.parameters]))

    @abstractmethod
    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply this modifier to the given rates tensor."""


class NormFactorModifier(SingleParamConstraint, ParameterModifier):
    """Normalization factor modifier (simple scaling by parameter value)."""

    type: Literal["normfactor"] = "normfactor"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    constraint: str | None = Field(default=None)

    @property
    def auxdata(self) -> float:
        """Auxiliary data value for normfactor (always 0.0)."""
        # normfactor has no auxiliary measurement associated
        # return a neutral value (not used by constraint builders)
        return 0.0

    def expression(self, context: Context) -> TensorVar:
        """Return multiplicative factor for normfactor."""
        return context[self.parameter]

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply normfactor modifier (simple scaling by parameter)."""
        return cast("TensorVar", rates * self.expression(context))


class NormSysModifier(SingleParamConstraint, ParameterModifier):
    """Normalization systematic modifier (with hi/lo interpolation)."""

    type: Literal["normsys"] = "normsys"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    constraint: str | None = None
    interpolation: interpolations.InterpolationDescriptor | None = None
    data: NormSysData
    _nominal_factor: TensorVar = PrivateAttr()
    _hi_factor_tensor: TensorVar = PrivateAttr()
    _lo_factor_tensor: TensorVar = PrivateAttr()
    _resolved_interpolation: interpolations.InterpolationDescriptor | None = (
        PrivateAttr(default=None)
    )

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        # HS3/ROOT numeric payloads are doubles.  Make that explicit instead
        # of letting PyTensor's Python-scalar cast policy narrow them to
        # float32 before interpolation.
        self._nominal_factor = pt.constant(np.asarray(1.0, dtype=config.floatX))
        self._materialize_anchors(self.data.hi, self.data.lo)
        if self.interpolation is not None:
            self.resolve_interpolation(self.interpolation)

    def _materialize_anchors(self, hi: float, lo: float) -> None:
        """Materialize dtype-matched anchor tensors once per resolved form."""
        dtype = self._nominal_factor.dtype
        self._hi_factor_tensor = pt.constant(hi, dtype=dtype)
        self._lo_factor_tensor = pt.constant(lo, dtype=dtype)

    def resolve_interpolation(
        self, descriptor: interpolations.InterpolationDescriptor
    ) -> None:
        """Store the effective FlexibleInterpVar descriptor for evaluation.

        ROOT protects only the polynomial/exponential Flexible form from
        non-positive anchors.  Keep the serialized values untouched and use
        separate epsilon-adjusted tensors for the computation graph.
        """
        descriptor = interpolations.validate_interpolation_for_class(
            descriptor, "flexible"
        )
        self._resolved_interpolation = descriptor
        hi = self.data.hi
        lo = self.data.lo
        if descriptor.key == ("mult", "poly6", "exp"):
            epsilon = np.finfo(np.dtype(self._nominal_factor.dtype)).eps
            hi = epsilon if hi <= 0 else hi
            lo = epsilon if lo <= 0 else lo
        self._materialize_anchors(hi, lo)

    def compatibility_interpolation(self) -> interpolations.InterpolationDescriptor:
        """Translate the former pyhs3/ROOT-0.2 code to a structured form."""
        if self.data.interpolation == "code1":
            return interpolations.InterpolationDescriptor.model_validate(
                {"type": "mult", "in": "exp", "out": None}
            )
        # Both an explicit legacy code4 and the historical omitted default map
        # to FlexibleInterpVar's polynomial/exponential form.
        return interpolations.InterpolationDescriptor.model_validate(
            {"type": "mult", "in": "poly6", "out": "exp"}
        )

    @property
    def resolved_interpolation(self) -> interpolations.InterpolationDescriptor:
        """Return the channel-resolved descriptor or fail before graph building."""
        if self._resolved_interpolation is None:
            msg = f"normsys modifier '{self.name}' has no structured interpolation"
            raise ValueError(msg)
        return self._resolved_interpolation

    def interpolate_group(self, context: Context, current: TensorVar) -> TensorVar:
        """Apply this entry to a running FlexibleInterpVar result."""
        return interpolations.apply_interpolation_descriptor(
            self.resolved_interpolation,
            context[self.parameter],
            self._nominal_factor,
            self._hi_factor_tensor,
            self._lo_factor_tensor,
            current=current,
        )

    def interpolation_inputs(
        self, context: Context
    ) -> tuple[TensorVar, TensorVar, TensorVar]:
        """Return the parameter and materialized anchors for grouped evaluation."""
        return context[self.parameter], self._hi_factor_tensor, self._lo_factor_tensor

    @property
    def auxdata(self) -> float:
        """Auxiliary data value for normsys (always 0.0)."""
        # For normsys with Gaussian constraint the aux data is typically 0.
        # Keep this simple and return 0.0 (the constraint builder will
        # interpret as needed).
        return 0.0

    def expression(self, context: Context) -> TensorVar:
        """Return multiplicative factor for normsys."""
        return self.interpolate_group(context, self._nominal_factor)

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply normsys modifier (systematic with hi/lo interpolation)."""
        return cast("TensorVar", rates * self.expression(context))


class HistoSysModifier(SingleParamConstraint, ParameterModifier):
    """Additive correlated shape systematic modifier."""

    type: Literal["histosys"] = "histosys"
    application: Literal["additive"] = Field("additive", exclude=True)
    constraint: str | None = None
    interpolation: interpolations.InterpolationDescriptor | None = None
    data: HistoSysData
    _hi_tensor: TensorVar = PrivateAttr()
    _lo_tensor: TensorVar = PrivateAttr()
    _resolved_interpolation: interpolations.InterpolationDescriptor | None = (
        PrivateAttr(default=None)
    )

    def model_post_init(self, __context: Any, /) -> None:
        """Materialize absolute templates and any modifier-level descriptor."""
        self._hi_tensor = pt.as_tensor_variable(
            np.asarray(self.data.hi.contents, dtype=config.floatX)
        )
        self._lo_tensor = pt.as_tensor_variable(
            np.asarray(self.data.lo.contents, dtype=config.floatX)
        )
        if self.interpolation is not None:
            self.resolve_interpolation(self.interpolation)

    def resolve_interpolation(
        self, descriptor: interpolations.InterpolationDescriptor
    ) -> None:
        """Store the effective PiecewiseInterpolation descriptor."""
        self._resolved_interpolation = interpolations.validate_interpolation_for_class(
            descriptor, "piecewise"
        )

    def compatibility_interpolation(self) -> interpolations.InterpolationDescriptor:
        """Translate the former pyhs3/ROOT-0.2 code to a structured form."""
        legacy = self.data.interpolation
        if legacy == "code0":
            raw: dict[str, str | None] = {
                "type": "add",
                "in": "poly1",
                "out": None,
            }
        elif legacy == "code2":
            raw = {"type": "add", "in": "poly2", "out": "poly1"}
        else:
            # Explicit code4p and the former omitted default are ROOT
            # PiecewiseInterpolation's additive polynomial/linear form.
            raw = {"type": "add", "in": "poly6", "out": "poly1"}
        return interpolations.InterpolationDescriptor.model_validate(raw)

    @property
    def resolved_interpolation(self) -> interpolations.InterpolationDescriptor:
        """Return the channel-resolved descriptor or fail before graph building."""
        if self._resolved_interpolation is None:
            msg = f"histosys modifier '{self.name}' has no structured interpolation"
            raise ValueError(msg)
        return self._resolved_interpolation

    def interpolate_group(
        self,
        context: Context,
        nominal: TensorVar,
        current: TensorVar,
    ) -> TensorVar:
        """Apply this entry to a running PiecewiseInterpolation result."""
        return interpolations.apply_interpolation_descriptor(
            self.resolved_interpolation,
            context[self.parameter],
            nominal,
            self._hi_tensor,
            self._lo_tensor,
            current=current,
        )

    def interpolation_inputs(
        self, context: Context
    ) -> tuple[TensorVar, TensorVar, TensorVar]:
        """Return the parameter and materialized templates for grouped evaluation."""
        return context[self.parameter], self._hi_tensor, self._lo_tensor

    @property
    def auxdata(self) -> float:
        """Auxiliary data value for histosys (always 0.0)."""
        # histosys typical auxiliary measurement around 0
        return 0.0

    def expression(self, context: Context) -> TensorVar:
        """Return the histosys parameter value for dependency graph evaluation.

        For histosys modifiers, the actual additive variation calculation happens in apply()
        since it depends on the nominal rates. This method returns just the parameter value
        for the dependency graph to track parameter dependencies correctly.
        """
        return context[self.parameter]

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply one histosys entry with *rates* as nominal and running value."""
        return self.interpolate_group(context, rates, rates)


class ShapeFactorModifier(ParametersModifier):
    """Uncorrelated multiplicative bin-by-bin scaling modifier."""

    type: Literal["shapefactor"] = "shapefactor"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    parameters: list[str]

    @property
    def auxdata(self) -> list[float]:
        """Auxiliary data values for shapefactor (empty list)."""
        # shapefactor doesn't produce aux measurements per se; return empty list
        return []

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply shapefactor modifier (uncorrelated bin-by-bin scaling)."""
        return cast("TensorVar", rates * self.expression(context))


class ShapeSysModifier(HasConstraint, ParametersModifier):
    """Uncorrelated shape systematic with Poisson constraints."""

    type: Literal["shapesys"] = "shapesys"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    constraint: Literal["Poisson"] = "Poisson"
    data: ShapeSysData

    @property
    def auxdata(self) -> list[float]:
        """Auxiliary data values for shapesys (from data vals)."""
        # shapesys typically uses auxiliary counts derived from the sample data and uncertainties.
        return self.data.vals

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply shapesys modifier (shape systematic with constraints)."""
        return cast("TensorVar", rates * self.expression(context))

    def _build_bin_constraint(
        self, context: Context, sample_data: SampleData
    ) -> tuple[Distribution, Context]:
        """Construct a single vectorized Poisson constraint distribution.

        Stacks the B per-bin nuisance parameters into one length-B tensor and
        builds ONE :class:`PoissonDist` over vector-valued ``x``/``mean``,
        instead of one scalar :class:`PoissonDist` per bin. ``PoissonDist``'s
        ``likelihood``/``log_likelihood`` are elementwise pytensor expressions,
        so a vector input produces a vector output; :meth:`make_constraint`
        and :meth:`log_constraint` reduce the resulting (B,) vector with
        ``pt.prod``/``pt.sum``. Shared by both so the parametrization is
        defined exactly once.
        """
        name = f"constraint_{self.name}"

        # (sigma_b)^{-2} = (nominal / vals)^2, evaluated on concrete floats.
        # Shape: (B,) -- one rate per bin, in self.parameters order.
        rates = (
            np.asarray(sample_data.contents, dtype=np.float64)
            / np.asarray(self.data.vals, dtype=np.float64)
        ) ** 2

        # gamma: (B,) stacked per-bin nuisance-parameter tensors.
        gamma = pt.stack([context[parameter] for parameter in self.parameters])
        scaled_name = f"{name}_scaled"
        x_name = f"{name}_x"

        augmented_context = dict(context)
        augmented_context[x_name] = pt.constant(rates)
        augmented_context[scaled_name] = gamma * pt.constant(rates)

        dist: Distribution = PoissonDist(name=name, x=x_name, mean=scaled_name)
        return dist, Context({**augmented_context, **dist.constants})

    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations."""
        dist, augmented_context = self._build_bin_constraint(context, sample_data)
        return cast(TensorVar, pt.prod(dist.expression(augmented_context)))  # type: ignore[no-untyped-call]

    def log_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term as the sum of per-bin Poisson log-probabilities."""
        dist, augmented_context = self._build_bin_constraint(context, sample_data)
        return cast(TensorVar, pt.sum(dist.log_expression(augmented_context)))  # type: ignore[no-untyped-call]


class StatErrorModifier(HasConstraint, ParametersModifier):
    """Statistical uncertainty modifier (Barlow-Beeston method)."""

    type: Literal["staterror"] = "staterror"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    parameters: list[str] = Field(default_factory=list)
    constraint: Literal["Gauss", "Poisson"] = "Gauss"
    data: StatErrorData | None = None

    @property
    def auxdata(self) -> list[float]:
        """Auxiliary data values for staterror (list of 1.0)."""
        # For staterror, each auxiliary measurement is typically 1.0 (or derived).
        return [1.0] * len(self.parameters)

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply staterror modifier (Barlow-Beeston statistical uncertainties)."""
        return cast("TensorVar", rates * self.expression(context))

    def _build_bin_constraint(
        self, context: Context, sample_data: SampleData, data: StatErrorData
    ) -> tuple[Distribution, Context] | None:
        """Construct a single vectorized Gauss/Poisson constraint distribution.

        Only used in BB-full mode. In BB-lite mode, constraints are built at
        channel level. Stacks the per-bin nuisance parameters into one
        length-B tensor and builds ONE GaussianDist/PoissonDist over
        vector-valued inputs, instead of one scalar distribution per bin
        (mirroring :meth:`ShapeSysModifier._build_bin_constraint`).

        Returns ``None`` when there is nothing to constrain: either
        ``self.parameters`` is empty, or (Poisson only) every bin has
        ``nominal_yield <= 0`` and was excluded. Gauss bins with a
        nonpositive yield are NOT excluded -- they fall back to
        ``sigma_value = 1.0``, matching the pre-vectorization per-bin
        behavior exactly.

        Shared by :meth:`make_constraint` (product of per-bin probabilities)
        and :meth:`log_constraint` (sum of per-bin log-probabilities) so the
        parametrization is defined exactly once. ``data`` is taken as an
        explicit argument (rather than reading ``self.data``) so callers
        narrow the ``StatErrorData | None`` type by raising before calling,
        instead of asserting it here.
        """
        # Truncate to the shortest of parameters/uncertainties, mirroring the
        # zip(..., strict=False) pairing used before vectorization.
        n = min(len(self.parameters), len(data.uncertainties))
        if n == 0:
            return None

        name = f"constraint_{self.name}"
        # Shape (B,): concrete per-bin nominal yields and uncertainties.
        nominal_yields = np.asarray(sample_data.contents[:n], dtype=np.float64)
        uncertainties = np.asarray(data.uncertainties[:n], dtype=np.float64)
        augmented_context = dict(context)

        dist: Distribution
        if self.constraint == "Poisson":
            # Skip zero-yield bins: tau = (nu/sigma)^2 is undefined when nu <= 0.
            keep = nominal_yields > 0
            parameters = [
                p for p, k in zip(self.parameters[:n], keep, strict=True) if k
            ]
            if not parameters:
                return None
            # tau = (nominal/uncertainty)^2 = 1/sigma_value^2. Shape: (B',).
            sigma_value = uncertainties[keep] / nominal_yields[keep]
            tau = 1.0 / sigma_value**2
            gamma = pt.stack([context[parameter] for parameter in parameters])
            scaled_name = f"{name}_scaled"
            x_name = f"{name}_x"
            augmented_context[x_name] = pt.constant(tau)
            augmented_context[scaled_name] = gamma * pt.constant(tau)
            dist = PoissonDist(name=name, x=x_name, mean=scaled_name)
        else:  # "Gauss"
            # sigma_value is the relative uncertainty = uncertainty / nominal_yield,
            # falling back to 1.0 for nonpositive yields (bin kept, not skipped).
            # Shape: (B,).
            sigma_value = np.ones_like(nominal_yields)
            positive = nominal_yields > 0
            sigma_value[positive] = uncertainties[positive] / nominal_yields[positive]
            gamma = pt.stack([context[parameter] for parameter in self.parameters[:n]])
            mean_name = f"{name}_mean"
            sigma_name = f"{name}_sigma"
            augmented_context[mean_name] = gamma
            augmented_context[sigma_name] = pt.constant(sigma_value)
            dist = GaussianDist(name=name, x=1.0, mean=mean_name, sigma=sigma_name)

        return dist, Context({**augmented_context, **dist.constants})

    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations.

        Only used in BB-full mode. In BB-lite mode, constraints are built at channel level.
        """
        if self.data is None:
            msg = (
                "StatErrorModifier.data is required for BB-full mode (make_constraint)"
            )
            raise ValueError(msg)

        pair = self._build_bin_constraint(context, sample_data, self.data)
        if pair is None:
            return pt.constant(1.0)

        dist, augmented_context = pair
        return cast(TensorVar, pt.prod(dist.expression(augmented_context)))  # type: ignore[no-untyped-call]

    def log_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term as the sum of per-bin Gauss/Poisson log-probabilities.

        Only used in BB-full mode. In BB-lite mode, constraints are built at channel level.
        """
        if self.data is None:
            msg = "StatErrorModifier.data is required for BB-full mode (log_constraint)"
            raise ValueError(msg)

        pair = self._build_bin_constraint(context, sample_data, self.data)
        if pair is None:
            return pt.constant(0.0)

        dist, augmented_context = pair
        return cast(TensorVar, pt.sum(dist.log_expression(augmented_context)))  # type: ignore[no-untyped-call]


# Discriminated union of all modifier types.
ModifierType = Annotated[
    NormFactorModifier
    | NormSysModifier
    | HistoSysModifier
    | ShapeFactorModifier
    | ShapeSysModifier
    | StatErrorModifier,
    Field(discriminator="type"),
    custom_error_msg(
        {
            "union_tag_invalid": "Unknown modifier type '{tag}' does not match any supported modifier types"
        }
    ),
]


class Modifiers(RootModel[list[ModifierType]]):
    """
    Collection of modifiers for a HistFactory sample.

    Manages a set of modifier instances, providing list-like access and
    validation. Handles modifier creation from configuration dictionaries
    and maintains type safety through discriminated unions.
    """

    root: list[ModifierType] = Field(default_factory=list)

    def __iter__(self) -> Iterator[ModifierType]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, index: int) -> ModifierType:
        return self.root[index]
