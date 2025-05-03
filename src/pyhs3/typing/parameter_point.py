from __future__ import annotations

from typing import TypedDict


class Parameter(TypedDict):
    name: str
    value: float


class ParameterPoint(TypedDict):
    name: str
    parameters: list[Parameter]
