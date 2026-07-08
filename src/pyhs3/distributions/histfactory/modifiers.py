from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, Literal, cast

import numpy as np
import pytensor.tensor as pt
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    RootModel,
    model_validator,
)

from pyhs3.context import Context
from pyhs3.distributions.basic import GaussianDist, LogNormalDist, PoissonDist
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

    hi: float
    lo: float
    interpolation: Literal["code1", "code4"] = Field(default="code4")


class HistoSysDataContents(BaseModel):
    """Contents data for histosys modifier."""

    contents: list[float]


class HistoSysData(ModifierData):
    """Data for histosys modifier."""

    hi: HistoSysDataContents
    lo: HistoSysDataContents
    interpolation: Literal["code0", "code2", "code4p"] = Field(default="code4p")

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

    constraint: Literal["Gauss", "Poisson", "LogNormal"] | None

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
    """Mixin for single-parameter modifiers that use a standard Gauss/Poisson/LogNormal constraint."""

    name: str
    parameter: str

    def _build_constraint(self, context: Context) -> tuple[Distribution, Context]:
        """Construct the Gauss/Poisson/LogNormal constraint distribution.

        Shared by :meth:`make_constraint` and :meth:`log_constraint` so the
        parametrization (which distribution, and with which parameter/constant
        values) is defined exactly once.
        """
        name = f"constraint_{self.name}"
        constraint_dist: Distribution

        if self.constraint == "Gauss":
            constraint_dist = GaussianDist(
                name=name, x=0.0, mean=self.parameter, sigma=1.0
            )
        elif self.constraint == "Poisson":
            constraint_dist = PoissonDist(name=name, x=1.0, mean=self.parameter)
        else:  # self.constraint == "LogNormal"
            constraint_dist = LogNormalDist(
                name=name, x=1.0, mu=self.parameter, sigma=1.0
            )

        augmented_context = {**context, **constraint_dist.constants}
        return constraint_dist, Context(augmented_context)

    def make_constraint(self, context: Context, _: SampleData) -> TensorVar:
        """Create constraint term using a Gauss, Poisson, or LogNormal distribution."""
        constraint_dist, augmented_context = self._build_constraint(context)
        return constraint_dist.expression(augmented_context)

    def log_constraint(self, context: Context, _: SampleData) -> TensorVar:
        """Create constraint term using the analytic log form of the same distribution."""
        constraint_dist, augmented_context = self._build_constraint(context)
        return constraint_dist.log_expression(augmented_context)


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


class NormFactorModifier(ParameterModifier):
    """Normalization factor modifier (simple scaling by parameter value)."""

    type: Literal["normfactor"] = "normfactor"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    # NormFactor purposely has no constraint by default (keeps it None)
    constraint: None = Field(default=None)

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
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: NormSysData
    _nominal_factor: TensorVar = PrivateAttr()
    _hi_factor_tensor: TensorVar = PrivateAttr()
    _lo_factor_tensor: TensorVar = PrivateAttr()

    def model_post_init(self, __context: Any, /) -> None:
        """Initialize computed collections after Pydantic validation."""
        self._nominal_factor = pt.constant(1.0)
        self._hi_factor_tensor = pt.constant(self.data.hi)
        self._lo_factor_tensor = pt.constant(self.data.lo)

    @property
    def auxdata(self) -> float:
        """Auxiliary data value for normsys (always 0.0)."""
        # For normsys with Gaussian constraint the aux data is typically 0.
        # Keep this simple and return 0.0 (the constraint builder will
        # interpret as needed).
        return 0.0

    def expression(self, context: Context) -> TensorVar:
        """Return multiplicative factor for normsys."""
        alpha = context[self.parameter]
        return interpolations.apply_interpolation(
            self.data.interpolation,
            alpha,
            self._nominal_factor,
            self._hi_factor_tensor,
            self._lo_factor_tensor,
        )

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply normsys modifier (systematic with hi/lo interpolation)."""
        return cast("TensorVar", rates * self.expression(context))


class HistoSysModifier(SingleParamConstraint, ParameterModifier):
    """Additive correlated shape systematic modifier."""

    type: Literal["histosys"] = "histosys"
    application: Literal["additive"] = Field("additive", exclude=True)
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: HistoSysData

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
        """Apply histosys (additive systematic) modifier."""
        alpha = context[self.parameter]

        # Get hi/lo absolute values
        hi_contents = self.data.hi.contents
        lo_contents = self.data.lo.contents

        # Convert absolute values to differences from nominal (current rates)
        hi_absolute = pt.as_tensor_variable(hi_contents)
        lo_absolute = pt.as_tensor_variable(lo_contents)
        hi_variation = hi_absolute - rates  # difference from nominal
        lo_variation = lo_absolute - rates  # difference from nominal
        zero_variation = pt.zeros_like(hi_variation)  # type: ignore[no-untyped-call]

        # Apply interpolation method to the differences
        interpolation = self.data.interpolation
        variation = interpolations.apply_interpolation(
            interpolation, alpha, zero_variation, hi_variation, lo_variation
        )

        return cast("TensorVar", rates + variation)


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

    def _build_bin_constraints(
        self, context: Context, sample_data: SampleData
    ) -> list[tuple[Distribution, Context]]:
        """Construct the per-bin Poisson constraint distributions.

        Shared by :meth:`make_constraint` (product of per-bin probabilities)
        and :meth:`log_constraint` (sum of per-bin log-probabilities) so the
        parametrization is defined exactly once.
        """
        name = f"constraint_{self.name}"

        # (sigma_b)^{-2} = (nominal / vals)^2, evaluated on concrete floats.
        rates = (
            np.asarray(sample_data.contents, dtype=np.float64)
            / np.asarray(self.data.vals, dtype=np.float64)
        ) ** 2

        # Use augmented context pattern for parameter * rate scaling
        augmented_context = dict(context)
        dists = []

        for parameter, rate in zip(self.parameters, rates, strict=False):
            # Create scaled parameter in augmented context
            scaled_param_name = f"{parameter}_scaled"
            augmented_context[scaled_param_name] = context[parameter] * rate

            # Create Poisson distribution with scaled parameter
            dist: Distribution = PoissonDist(
                name=f"{name}_{parameter}", x=rate, mean=scaled_param_name
            )
            dists.append(dist)

        # Pair each distribution with a context augmented by its own constants.
        return [
            (dist, Context({**augmented_context, **dist.constants})) for dist in dists
        ]

    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations."""
        pairs = self._build_bin_constraints(context, sample_data)
        factors = [dist.expression(ctx) for dist, ctx in pairs]
        return cast(TensorVar, pt.prod(pt.stack(factors), axis=0))  # type: ignore[no-untyped-call]

    def log_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term as the sum of per-bin Poisson log-probabilities."""
        pairs = self._build_bin_constraints(context, sample_data)
        log_terms = [dist.log_expression(ctx) for dist, ctx in pairs]
        return cast(TensorVar, pt.sum(pt.stack(log_terms), axis=0))  # type: ignore[no-untyped-call]


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

    def _build_bin_constraints(
        self, context: Context, sample_data: SampleData, data: StatErrorData
    ) -> list[tuple[Distribution, Context]]:
        """Construct the per-bin Gauss/Poisson constraint distributions.

        Only used in BB-full mode. In BB-lite mode, constraints are built at
        channel level. Shared by :meth:`make_constraint` (product of per-bin
        probabilities) and :meth:`log_constraint` (sum of per-bin
        log-probabilities) so the parametrization is defined exactly once.
        ``data`` is taken as an explicit argument (rather than reading
        ``self.data``) so callers narrow the ``StatErrorData | None`` type by
        raising before calling, instead of asserting it here.
        """
        name = f"constraint_{self.name}"
        augmented_context = dict(context)
        dists: list[Distribution] = []

        for i, (parameter, uncertainty) in enumerate(
            zip(self.parameters, data.uncertainties, strict=False)
        ):
            nominal_yield = sample_data.contents[i]
            # sigma_value is the relative uncertainty = uncertainty / nominal_yield
            sigma_value = uncertainty / nominal_yield if nominal_yield > 0 else 1.0

            if self.constraint == "Poisson":
                # Skip zero-yield bins: tau = (nu/sigma)^2 is undefined when nu <= 0
                if nominal_yield <= 0:
                    continue
                # Poisson: Poisson(tau | gamma * tau) where tau = (nominal/uncertainty)^2
                # Equivalently: tau = 1/sigma_value^2
                tau = 1.0 / sigma_value**2
                scaled_name = f"{parameter}_scaled"
                augmented_context[scaled_name] = context[parameter] * tau
                dist: Distribution = PoissonDist(
                    name=f"{name}_{parameter}",
                    x=float(tau),
                    mean=scaled_name,
                )
            else:  # "Gauss"
                # Gaussian: N(1.0 | gamma, sigma_value)
                sigma_param_name = f"{parameter}_sigma"
                augmented_context[sigma_param_name] = pt.constant(sigma_value)
                dist = GaussianDist(
                    name=f"{name}_{parameter}",
                    x=1.0,
                    mean=parameter,
                    sigma=sigma_param_name,
                )
            dists.append(dist)

        # Pair each distribution with a context augmented by its own constants.
        return [
            (dist, Context({**augmented_context, **dist.constants})) for dist in dists
        ]

    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations.

        Only used in BB-full mode. In BB-lite mode, constraints are built at channel level.
        """
        if self.data is None:
            msg = (
                "StatErrorModifier.data is required for BB-full mode (make_constraint)"
            )
            raise ValueError(msg)

        pairs = self._build_bin_constraints(context, sample_data, self.data)
        if not pairs:
            return pt.constant(1.0)

        factors = [dist.expression(ctx) for dist, ctx in pairs]
        return cast(TensorVar, pt.prod(pt.stack(factors), axis=0))  # type: ignore[no-untyped-call]

    def log_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term as the sum of per-bin Gauss/Poisson log-probabilities.

        Only used in BB-full mode. In BB-lite mode, constraints are built at channel level.
        """
        if self.data is None:
            msg = "StatErrorModifier.data is required for BB-full mode (log_constraint)"
            raise ValueError(msg)

        pairs = self._build_bin_constraints(context, sample_data, self.data)
        if not pairs:
            return pt.constant(0.0)

        log_terms = [dist.log_expression(ctx) for dist, ctx in pairs]
        return cast(TensorVar, pt.sum(pt.stack(log_terms), axis=0))  # type: ignore[no-untyped-call]


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
