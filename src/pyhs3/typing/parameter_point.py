from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Parameter:
    """
    Represents a single parameter point.

    Attributes:
        name (str): Name of the parameter.
        value (float): Value of the parameter.
    """

    name: str
    value: float


class ParameterPoint(TypedDict):
    name: str
    parameters: list[Parameter]
