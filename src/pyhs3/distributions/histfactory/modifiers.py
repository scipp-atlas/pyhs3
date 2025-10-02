from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel, model_validator

# Import existing distributions for constraint terms
from pyhs3.exceptions import custom_error_msg


class ModifierData(BaseModel):
    """Base class for modifier data."""


class NormSysData(ModifierData):
    """Data for normsys modifier."""

    hi: float
    lo: float
    interpolation: str = Field(default="lin")


class HistoSysDataContents(BaseModel):
    contents: list[float]


class HistoSysData(ModifierData):
    """Data for histosys modifier."""

    hi: HistoSysDataContents
    lo: HistoSysDataContents
    interpolation: str = Field(default="lin")

    @model_validator(mode="after")
    def validate_lengths(self) -> HistoSysData:
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


# Parameterized modifier base (single parameter)
class ParameterModifier(BaseModel, ABC):
    """Base for modifiers that use a single parameter name."""

    type: str
    parameter: str
    # constraint may be present (e.g. normsys/histosys/shapesys) or None (e.g. normfactor)
    constraint: Literal["Gauss", "Poisson", "LogNormal"] | None = Field(default=None)

    @property
    @abstractmethod
    def auxdata(self) -> float:
        """Auxiliary data value associated with this modifier (single float)."""


# Multi-parameter modifier base (per-bin parameters)
class ParametersModifier(BaseModel, ABC):
    """Base for modifiers that use multiple parameter names (one per bin)."""

    type: str
    parameters: list[str]
    constraint: Literal["Gauss", "Poisson", "LogNormal"] | None = Field(default=None)

    @property
    @abstractmethod
    def auxdata(self) -> list[float]:
        """Auxiliary data values associated with this modifier (list of floats)."""


class NormFactorModifier(ParameterModifier):
    """Normalization factor modifier (simple scaling by parameter value)."""

    type: Literal["normfactor"] = "normfactor"
    # NormFactor purposely has no constraint by default (keeps it None)
    constraint: None = Field(default=None)

    @property
    def auxdata(self) -> float:
        # normfactor has no auxiliary measurement associated
        # return a neutral value (not used by constraint builders)
        return 0.0


class NormSysModifier(ParameterModifier):
    """Normalization systematic modifier (with hi/lo interpolation)."""

    type: Literal["normsys"] = "normsys"
    parameter: str
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: NormSysData

    @property
    def auxdata(self) -> float:
        # For normsys with Gaussian constraint the aux data is typically 0.
        # Keep this simple and return 0.0 (the constraint builder will
        # interpret as needed).
        return 0.0


class HistoSysModifier(ParameterModifier):
    """Additive correlated shape systematic modifier."""

    type: Literal["histosys"] = "histosys"
    parameter: str
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: HistoSysData

    @property
    def auxdata(self) -> float:
        # histosys typical auxiliary measurement around 0
        return 0.0


class ShapeFactorModifier(ParametersModifier):
    """Uncorrelated multiplicative bin-by-bin scaling modifier."""

    type: Literal["shapefactor"] = "shapefactor"
    parameters: list[str]

    @property
    def auxdata(self) -> list[float]:
        # shapefactor doesn't produce aux measurements per se; return empty list
        return []


class ShapeSysModifier(ParameterModifier):
    """Uncorrelated shape systematic with Poisson constraints."""

    type: Literal["shapesys"] = "shapesys"
    parameter: str
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Poisson"
    data: ShapeSysData

    @property
    def auxdata(self) -> float:
        # shapesys typically uses auxiliary counts derived from the uncertainties.
        # The precise aux-data computation may require access to the sample nominal.
        # For now return a placeholder; your constraint builder can adapt for shapesys.
        # (You can replace this with a more accurate computation if you pass in nominal.)
        uncertainty = self.data.vals[0] if self.data.vals else 1.0
        nominal_yield = 5.0  # placeholder; ideally supplied by the sample context
        return (nominal_yield**2) / (uncertainty**2)


class StatErrorModifier(ParametersModifier):
    """Statistical uncertainty modifier (Barlow-Beeston method)."""

    type: Literal["staterror"] = "staterror"
    parameters: list[str]
    constraint: Literal["Gauss", "Poisson", "LogNormal"] = "Gauss"
    data: StatErrorData

    @property
    def auxdata(self) -> list[float]:
        # For staterror, each auxiliary measurement is typically 1.0 (or derived).
        return [1.0] * len(self.parameters)


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
