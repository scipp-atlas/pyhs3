from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from pydantic import (
    BaseModel,
    ConfigDict,
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

# Interpolation codes each modifier type accepts, named so other modules (such
# as the pyhf converter) validate against the same set.
NormSysInterpolation = Literal["code1", "code4"]
HistoSysInterpolation = Literal["code0", "code2", "code4p"]


class ModifierData(BaseModel):
    """Base class for modifier data."""


class NormSysData(ModifierData):
    """Data for normsys modifier."""

    hi: float
    lo: float
    interpolation: NormSysInterpolation = Field(default="code4")


class HistoSysDataContents(BaseModel):
    """Contents data for histosys modifier."""

    contents: list[float]


class HistoSysData(ModifierData):
    """Data for histosys modifier."""

    hi: HistoSysDataContents
    lo: HistoSysDataContents
    interpolation: HistoSysInterpolation = Field(default="code4p")

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

    # A key the modifier does not define is an error. Ignoring it would change
    # the likelihood silently, e.g. a constraint given under a key pyhs3 does
    # not read would leave the default constraint in place.
    model_config = ConfigDict(extra="forbid")

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


def gamma_constraint(
    name: str,
    gamma: TensorVar,
    sigma_rel: npt.NDArray[np.float64],
    constraint: Literal["Gauss", "Poisson", "LogNormal"],
    context: Context,
) -> tuple[Distribution, Context]:
    """Constraint distribution of per-bin parameters whose nominal value is 1.

    Shared by ``shapesys``, ``staterror`` (Barlow-Beeston full) and the
    channel-level Barlow-Beeston-lite constraint, so each HS3 v0.2 constraint
    type is defined once. ``gamma`` and ``sigma_rel`` have shape (B,): the
    parameter and its relative uncertainty for each constrained bin. The
    returned distribution is elementwise, so it evaluates to (B,) per-bin terms:

    - ``Gauss``: Gauss(1 | gamma, sigma_rel)
    - ``Poisson``: Pois(tau | gamma * tau), continuous, with tau = 1 / sigma_rel**2
      (as RooFit's HistFactory builds it)
    - ``LogNormal``: LogNormal(1 | mu = ln gamma, sigma = sigma_rel), whose
      median is gamma

    Args:
        name: Name of the constraint distribution; helper tensors are named after it.
        gamma: (B,) per-bin parameters.
        sigma_rel: (B,) relative uncertainties.
        constraint: The HS3 v0.2 constraint type.
        context: Built tensors by name; extended with the helper tensors.

    Returns:
        The constraint distribution and the context it evaluates in.
    """
    augmented_context = dict(context)
    dist: Distribution
    if constraint == "Poisson":
        tau = 1.0 / sigma_rel**2
        x_name, mean_name = f"{name}_x", f"{name}_scaled"
        augmented_context[x_name] = pt.constant(tau)
        augmented_context[mean_name] = gamma * pt.constant(tau)
        dist = PoissonDist(name=name, x=x_name, mean=mean_name)
    elif constraint == "Gauss":
        mean_name, sigma_name = f"{name}_mean", f"{name}_sigma"
        augmented_context[mean_name] = gamma
        augmented_context[sigma_name] = pt.constant(sigma_rel)
        dist = GaussianDist(name=name, x=1.0, mean=mean_name, sigma=sigma_name)
    else:  # constraint == "LogNormal"
        mu_name, sigma_name = f"{name}_mu", f"{name}_sigma"
        augmented_context[mu_name] = pt.log(gamma)
        augmented_context[sigma_name] = pt.constant(sigma_rel)
        dist = LogNormalDist(name=name, x=1.0, mu=mu_name, sigma=sigma_name)
    return dist, Context({**augmented_context, **dist.constants})


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

    def unconstrained_parameters(
        self,
        sample_data: SampleData,  # noqa: ARG002  # pylint: disable=unused-argument
    ) -> list[str]:
        """Parameters that scale a positive yield but get no constraint term.

        A per-bin parameter whose bin has a positive yield and zero
        uncertainty is left out of the constraint; pyhf and ROOT fix it at its
        nominal value, so it must be ``const``. Modifiers with one constrained
        parameter have none.
        """
        return []


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
        # PyTensor infers float32 for a float that float32 represents exactly
        # (0.5, 1.5); the interpolation coefficients are built from these
        # constants alone, so they must be float64 to keep float64 precision.
        self._hi_factor_tensor = pt.constant(self.data.hi, dtype="float64")
        self._lo_factor_tensor = pt.constant(self.data.lo, dtype="float64")

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
    """Uncorrelated shape systematic with per-bin constraints.

    ``data.vals`` are absolute per-bin uncertainties (the pyhf convention); the
    relative uncertainty of bin ``i`` is ``vals[i] / nominal[i]``. See
    :func:`gamma_constraint` for the constraint term of each type.
    """

    type: Literal["shapesys"] = "shapesys"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Poisson"
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
    ) -> tuple[Distribution, Context] | None:
        """Construct a single vectorized constraint distribution over all bins.

        Stacks the B' constrained per-bin nuisance parameters into one
        length-B' tensor and builds ONE distribution over vector-valued inputs
        (see :func:`gamma_constraint`), instead of one scalar distribution per
        bin; :meth:`make_constraint` and :meth:`log_constraint` reduce the
        resulting (B',) vector with ``pt.prod``/``pt.sum``. Shared by both so
        the parametrization is defined exactly once.

        A bin with zero nominal yield or zero uncertainty has no relative
        uncertainty and gets no constraint term; pyhf and ROOT fix its
        parameter at 1 instead, which HS3 expresses as ``const`` in the
        parameter point. Returns ``None`` when no bin is constrained.
        """
        # Shape (B,): per-bin nominal yields and absolute uncertainties.
        nominal_yields = np.asarray(sample_data.contents, dtype=np.float64)
        uncertainties = np.asarray(self.data.vals, dtype=np.float64)
        keep = (nominal_yields > 0) & (uncertainties > 0)
        parameters = [p for p, k in zip(self.parameters, keep, strict=True) if k]
        if not parameters:
            return None
        # Shape (B',): relative uncertainty of the constrained bins.
        sigma_rel = uncertainties[keep] / nominal_yields[keep]
        # gamma: (B',) stacked per-bin nuisance-parameter tensors.
        gamma = pt.stack([context[parameter] for parameter in parameters])
        return gamma_constraint(
            f"constraint_{self.name}", gamma, sigma_rel, self.constraint, context
        )

    def unconstrained_parameters(self, sample_data: SampleData) -> list[str]:
        """Parameters of bins with a positive yield and zero uncertainty."""
        # Shape (B,): per-bin nominal yields and absolute uncertainties.
        nominal_yields = np.asarray(sample_data.contents, dtype=np.float64)
        uncertainties = np.asarray(self.data.vals, dtype=np.float64)
        unconstrained = (nominal_yields > 0) & (uncertainties <= 0)
        return [p for p, u in zip(self.parameters, unconstrained, strict=True) if u]

    def make_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term using PyTensor operations."""
        pair = self._build_bin_constraint(context, sample_data)
        if pair is None:
            return pt.constant(1.0)

        dist, augmented_context = pair
        return cast(TensorVar, pt.prod(dist.expression(augmented_context)))  # type: ignore[no-untyped-call]

    def log_constraint(self, context: Context, sample_data: SampleData) -> TensorVar:
        """Create constraint term as the sum of per-bin log-probabilities."""
        pair = self._build_bin_constraint(context, sample_data)
        if pair is None:
            return pt.constant(0.0)

        dist, augmented_context = pair
        return cast(TensorVar, pt.sum(dist.log_expression(augmented_context)))  # type: ignore[no-untyped-call]


class StatErrorModifier(HasConstraint, ParametersModifier):
    """Statistical uncertainty modifier (Barlow-Beeston method)."""

    type: Literal["staterror"] = "staterror"
    application: Literal["multiplicative"] = Field("multiplicative", exclude=True)
    parameters: list[str] = Field(default_factory=list)
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
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
        """Construct a single vectorized constraint distribution over all bins.

        Only used in BB-full mode. In BB-lite mode, constraints are built at
        channel level. Stacks the per-bin nuisance parameters into one
        length-B tensor and builds ONE distribution over vector-valued inputs
        (see :func:`gamma_constraint`), instead of one scalar distribution per
        bin (mirroring :meth:`ShapeSysModifier._build_bin_constraint`).

        Returns ``None`` when there is nothing to constrain: either
        ``self.parameters`` is empty, or every bin was excluded. A bin with a
        positive yield and zero uncertainty is excluded for every type; pyhf
        and ROOT fix its parameter at 1 instead (``const`` in HS3). For
        Poisson, every bin with ``nominal_yield <= 0`` is also excluded. Gauss
        and LogNormal bins with
        a nonpositive yield are NOT excluded -- they fall back to a relative
        uncertainty of 1.0, matching the pre-vectorization per-bin Gauss
        behavior exactly. Poisson needs tau = 1 / sigma**2, which such a bin
        does not have.

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

        # Shape (B,): concrete per-bin nominal yields and uncertainties.
        nominal_yields = np.asarray(sample_data.contents[:n], dtype=np.float64)
        uncertainties = np.asarray(data.uncertainties[:n], dtype=np.float64)
        positive = nominal_yields > 0
        # A zero uncertainty on a positive yield gives no constraint term: pyhf
        # and ROOT fix that bin's parameter at 1 instead (``const`` in HS3).
        unconstrained = positive & (uncertainties <= 0)

        if self.constraint == "Poisson":
            # Skip zero-yield bins: tau = (nu/sigma)^2 is undefined when nu <= 0.
            keep = positive & ~unconstrained
            # Shape (B',): relative uncertainty of the surviving bins.
            sigma_rel = uncertainties[keep] / nominal_yields[keep]
        else:  # "Gauss" or "LogNormal"
            # Relative uncertainty = uncertainty / nominal_yield, falling back
            # to 1.0 for nonpositive yields (bin kept, not skipped). Shape: (B',).
            keep = ~unconstrained
            sigma_rel = np.ones_like(nominal_yields)
            sigma_rel[positive] = uncertainties[positive] / nominal_yields[positive]
            sigma_rel = sigma_rel[keep]
        parameters = [p for p, k in zip(self.parameters[:n], keep, strict=True) if k]
        if not parameters:
            return None

        gamma = pt.stack([context[parameter] for parameter in parameters])
        return gamma_constraint(
            f"constraint_{self.name}", gamma, sigma_rel, self.constraint, context
        )

    def unconstrained_parameters(self, sample_data: SampleData) -> list[str]:
        """Parameters of bins with a positive yield and zero uncertainty (BB-full only).

        In BB-lite mode the channel finds them from its combined uncertainties.
        """
        if self.data is None:
            return []
        n = min(len(self.parameters), len(self.data.uncertainties))
        # Shape (B,): concrete per-bin nominal yields and uncertainties.
        nominal_yields = np.asarray(sample_data.contents[:n], dtype=np.float64)
        uncertainties = np.asarray(self.data.uncertainties[:n], dtype=np.float64)
        unconstrained = (nominal_yields > 0) & (uncertainties <= 0)
        return [p for p, u in zip(self.parameters[:n], unconstrained, strict=True) if u]

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
