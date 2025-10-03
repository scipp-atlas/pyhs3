from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Annotated, Literal, cast

import pytensor.tensor as pt
from pydantic import BaseModel, Field, RootModel, model_validator
from pytensor.compile.function import function

from pyhs3.context import Context
from pyhs3.distributions.basic import GaussianDist, LogNormalDist, PoissonDist
from pyhs3.distributions.core import Distribution
from pyhs3.distributions.histfactory import interpolations

# Import existing distributions for constraint terms
from pyhs3.exceptions import custom_error_msg
from pyhs3.typing.aliases import TensorVar

if TYPE_CHECKING:
    from pyhs3.distributions.histfactory.samples import SampleData


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
class Modifier(BaseModel, ABC):
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


class HasConstraint(ABC):
    """Base class for modifiers that can have constraint terms."""

    constraint: Literal["Gauss", "Poisson", "LogNormal"] | None

    @abstractmethod
    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term for this modifier."""


# Parameterized modifier base (single parameter)
class ParameterModifier(Modifier, ABC):
    """Base for modifiers that use a single parameter name."""

    parameter: str

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
    @abstractmethod
    def auxdata(self) -> list[float]:
        """Auxiliary data values associated with this modifier (list of floats)."""

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

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply normfactor modifier (simple scaling by parameter)."""

        mu = context[self.parameter]
        return cast("TensorVar", rates * mu)


class NormSysModifier(HasConstraint, ParameterModifier):
    """Normalization systematic modifier (with hi/lo interpolation)."""

    type: Literal["normsys"] = "normsys"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: NormSysData

    @property
    def auxdata(self) -> float:
        """Auxiliary data value for normsys (always 0.0)."""
        # For normsys with Gaussian constraint the aux data is typically 0.
        # Keep this simple and return 0.0 (the constraint builder will
        # interpret as needed).
        return 0.0

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply normsys modifier (systematic with hi/lo interpolation)."""
        alpha = context[self.parameter]

        hi_factor = self.data.hi
        lo_factor = self.data.lo

        # Apply interpolation method
        interpolation = self.data.interpolation
        nominal_factor = pt.constant(1.0)
        hi_factor_tensor = pt.constant(hi_factor)
        lo_factor_tensor = pt.constant(lo_factor)

        factor = interpolations.apply_interpolation(
            interpolation, alpha, nominal_factor, hi_factor_tensor, lo_factor_tensor
        )

        return cast("TensorVar", rates * factor)

    def make_constraint(self, context: Context, _: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations."""

        name = f"constraint_{self.name}"
        constraint_dist: Distribution

        if self.constraint == "Gauss":
            # Gaussian constraint: Normal(auxdata | mean=parameter, std=sigma)
            constraint_dist = GaussianDist(
                name=name, x=0.0, mean=self.parameter, sigma=1.0
            )
        elif self.constraint == "Poisson":
            constraint_dist = PoissonDist(name=name, x=1.0, mean=self.parameter)
        elif self.constraint == "LogNormal":
            # LogNormal constraint: log(param) ~ N(0, 1)
            constraint_dist = LogNormalDist(
                name=name, x=1.0, mu=self.parameter, sigma=1.0
            )

        # Use the distribution's constants to augment the context
        augmented_context = {**context, **constraint_dist.constants}
        return constraint_dist.expression(Context(augmented_context))


class HistoSysModifier(HasConstraint, ParameterModifier):
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

    def make_constraint(self, context: Context, _: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations."""

        name = f"constraint_{self.name}"
        constraint_dist: Distribution

        if self.constraint == "Gauss":
            # Gaussian constraint: Normal(auxdata | mean=parameter, std=sigma)
            constraint_dist = GaussianDist(
                name=name, x=0.0, mean=self.parameter, sigma=1.0
            )
        elif self.constraint == "Poisson":
            constraint_dist = PoissonDist(name=name, x=1.0, mean=self.parameter)
        elif self.constraint == "LogNormal":
            constraint_dist = LogNormalDist(
                name=name, x=1.0, mu=self.parameter, sigma=1.0
            )

        # Use the distribution's constants to augment the context
        augmented_context = {**context, **constraint_dist.constants}
        return constraint_dist.expression(Context(augmented_context))


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
        param_names = self.parameters
        factors = pt.stack([context[name] for name in param_names])
        return cast("TensorVar", rates * factors)


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
        # ShapeSys uses multiple parameters (one per bin)
        param_names = self.parameters
        factors = pt.stack([context[name] for name in param_names])
        return cast("TensorVar", rates * factors)

    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations."""

        name = f"constraint_{self.name}"

        nominal_yield = pt.vector("nominal_yield")
        uncertainty = pt.vector("uncertainty")
        # (sigma_b)^{-2} = (nominal / vals)^2
        rate_fn = function(
            [nominal_yield, uncertainty], (nominal_yield / uncertainty) ** 2
        )
        rates = rate_fn(sample_data.contents, self.data.vals)

        # Use augmented context pattern for parameter * rate scaling
        augmented_context = dict(context)
        dists = []

        for parameter, rate in zip(self.parameters, rates, strict=False):
            # Create scaled parameter in augmented context
            scaled_param_name = f"{parameter}_scaled"
            augmented_context[scaled_param_name] = context[parameter] * rate

            # Create Poisson distribution with scaled parameter
            dist = PoissonDist(
                name=f"{name}_{parameter}", x=rate, mean=scaled_param_name
            )
            dists.append(dist)

        # Evaluate all distributions with augmented context (including constants)
        factors = []
        for dist in dists:
            # Use the distribution's constants to augment the context
            dist_augmented_context = {**augmented_context, **dist.constants}
            augmented_ctx = Context(dist_augmented_context)
            factors.append(dist.expression(augmented_ctx))

        return cast(TensorVar, pt.prod(pt.stack(factors), axis=0))  # type: ignore[no-untyped-call]


class StatErrorModifier(HasConstraint, ParametersModifier):
    """Statistical uncertainty modifier (Barlow-Beeston method)."""

    type: Literal["staterror"] = "staterror"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    parameters: list[str]
    constraint: Literal["Gauss"] = "Gauss"
    data: StatErrorData

    @property
    def auxdata(self) -> list[float]:
        """Auxiliary data values for staterror (list of 1.0)."""
        # For staterror, each auxiliary measurement is typically 1.0 (or derived).
        return [1.0] * len(self.parameters)

    def apply(self, context: Context, rates: TensorVar) -> TensorVar:
        """Apply staterror modifier (Barlow-Beeston statistical uncertainties)."""

        param_names = self.parameters
        # Staterror is bin-by-bin statistical uncertainty
        # Each bin gets its own gamma parameter
        factors = pt.stack([context[name] for name in param_names])
        return cast("TensorVar", rates * factors)

    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations."""

        # Barlow-Beeston method: per-bin Gaussian constraints with relative uncertainties
        name = f"constraint_{self.name}"
        augmented_context = dict(context)
        dists = []

        for i, (parameter, uncertainty) in enumerate(
            zip(self.parameters, self.data.uncertainties, strict=False)
        ):
            # Calculate relative uncertainty: sigma = uncertainty / nominal_yield
            nominal_yield = sample_data.contents[i]
            sigma_value = uncertainty / nominal_yield if nominal_yield > 0 else 1.0

            # Create sigma parameter in augmented context
            sigma_param_name = f"{parameter}_sigma"
            augmented_context[sigma_param_name] = pt.constant(sigma_value)

            # Create Gaussian constraint: N(auxdata=1.0 | mean=parameter, sigma=relative_uncertainty)
            dist = GaussianDist(
                name=f"{name}_{parameter}",
                x=1.0,  # Auxiliary data (typically 1.0 for staterror)
                mean=parameter,
                sigma=sigma_param_name,
            )
            dists.append(dist)

        if not dists:
            return pt.constant(1.0)

        # Evaluate all distributions with augmented context (including constants) and multiply
        factors = []
        for dist in dists:
            # Use the distribution's constants to augment the context
            dist_augmented_context = {**augmented_context, **dist.constants}
            augmented_ctx = Context(dist_augmented_context)
            factors.append(dist.expression(augmented_ctx))

        return cast(TensorVar, pt.prod(pt.stack(factors), axis=0))  # type: ignore[no-untyped-call]


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

    root: Annotated[
        list[ModifierType],
        custom_error_msg(
            {
                "union_tag_invalid": "Unknown modifier type '{tag}' does not match any supported modifier types"
            }
        ),
    ] = Field(default_factory=list)

    def __iter__(self) -> Iterator[ModifierType]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, index: int) -> ModifierType:
        return self.root[index]
