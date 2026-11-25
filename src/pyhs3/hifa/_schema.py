"""
Private pydantic models mirroring pyhf's ``schemas/1.0.0/defs.json``.

They check a pyhf workspace (HistFactory JSON, "HiFa") without a jsonschema
dependency.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from pyhs3.distributions.histfactory.modifiers import (
    HistoSysInterpolation,
    NormSysInterpolation,
)

NonEmptyFloats = Annotated[list[float], Field(min_length=1)]


class _Schema(BaseModel):
    """Base of the models mirroring pyhf's ``schemas/1.0.0/defs.json``.

    pyhf's schema sets ``additionalProperties: false`` on every object, so an
    unknown key is an error here too.
    """

    model_config = ConfigDict(extra="forbid")


class _NormFactor(_Schema):
    name: str
    type: Literal["normfactor"]
    data: None


class _NormSysData(_Schema):
    hi: float
    lo: float


class _NormSys(_Schema):
    name: str
    type: Literal["normsys"]
    data: _NormSysData


class _HistoSysData(_Schema):
    hi_data: NonEmptyFloats
    lo_data: NonEmptyFloats


class _HistoSys(_Schema):
    name: str
    type: Literal["histosys"]
    data: _HistoSysData


class _ShapeFactor(_Schema):
    name: str
    type: Literal["shapefactor"]
    data: None


class _ShapeSys(_Schema):
    name: str
    type: Literal["shapesys"]
    data: NonEmptyFloats


class _StatError(_Schema):
    name: str
    type: Literal["staterror"]
    data: NonEmptyFloats


class _Lumi(_Schema):
    name: Literal["lumi"]
    type: Literal["lumi"]
    data: None


_Modifier = Annotated[
    _NormFactor | _NormSys | _HistoSys | _ShapeFactor | _ShapeSys | _StatError | _Lumi,
    Field(discriminator="type"),
]


class _Sample(_Schema):
    name: str
    data: NonEmptyFloats
    modifiers: list[_Modifier]


class _Channel(_Schema):
    name: str
    samples: Annotated[list[_Sample], Field(min_length=1)]


class _Observation(_Schema):
    name: str
    data: NonEmptyFloats


class _Parameter(_Schema):
    name: str
    inits: NonEmptyFloats | None = None
    bounds: Annotated[list[tuple[float, float]], Field(min_length=1)] | None = None
    auxdata: NonEmptyFloats | None = None
    factors: NonEmptyFloats | None = None
    sigmas: NonEmptyFloats | None = None
    fixed: bool | None = None


class _Config(_Schema):
    poi: str
    parameters: list[_Parameter]


class _Measurement(_Schema):
    name: str
    config: _Config


class _HiFaWorkspace(_Schema):
    channels: Annotated[list[_Channel], Field(min_length=1)]
    measurements: Annotated[list[_Measurement], Field(min_length=1)]
    observations: Annotated[list[_Observation], Field(min_length=1)]
    version: Literal["1.0.0"]


class _NormSysSettings(_Schema):
    interpcode: NormSysInterpolation


class _HistoSysSettings(_Schema):
    interpcode: HistoSysInterpolation


class _ModifierSettings(_Schema):
    """pyhf's ``modifier_settings``, defaulting to pyhs3's (and ROOT's) codes.

    A modifier type left out keeps pyhs3's default.
    """

    normsys: _NormSysSettings = _NormSysSettings(interpcode="code4")
    histosys: _HistoSysSettings = _HistoSysSettings(interpcode="code4p")
