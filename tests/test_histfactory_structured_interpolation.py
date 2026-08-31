"""ROOT-compatible structured interpolation at HistFactory channel level."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pydantic import ValidationError

from pyhs3.context import Context
from pyhs3.distributions import HistFactoryDistChannel
from pyhs3.distributions.histfactory.modifiers import (
    HistoSysModifier,
    NormSysModifier,
)

ADD_LINEAR = {"type": "add", "in": "poly1", "out": None}
MULT_EXP = {"type": "mult", "in": "exp", "out": None}
ADD_POLY6 = {"type": "add", "in": "poly6", "out": "poly1"}
MULT_POLY6 = {"type": "mult", "in": "poly6", "out": "exp"}


def _normsys(
    name: str,
    parameter: str,
    high: float,
    low: float,
    interpolation: dict[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "type": "normsys",
        "parameter": parameter,
        "interpolation": interpolation,
        "data": {"hi": high, "lo": low},
    }


def _histosys(
    name: str,
    parameter: str,
    high: float,
    low: float,
    interpolation: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "type": "histosys",
        "parameter": parameter,
        "data": {
            "hi": {"contents": [high]},
            "lo": {"contents": [low]},
        },
    }
    if interpolation is not None:
        result["interpolation"] = interpolation
    return result


def _channel(
    modifiers: Sequence[dict[str, object]],
    *,
    default: dict[str, object] | None = None,
) -> HistFactoryDistChannel:
    raw: dict[str, object] = {
        "name": "channel",
        "type": "histfactory_dist",
        "axes": [{"name": "x", "min": 0.0, "max": 1.0, "nbins": 1}],
        "samples": [
            {
                "name": "sample",
                "data": {"contents": [10.0], "errors": [0.0]},
                "modifiers": list(modifiers),
            }
        ],
    }
    if default is not None:
        raw["default_interpolation"] = default
    return HistFactoryDistChannel.model_validate(raw)


def _evaluate(channel: HistFactoryDistChannel, **parameters: float) -> float:
    context = Context(
        {
            name: pt.constant(np.asarray(value, dtype=np.float64))
            for name, value in parameters.items()
        }
    )
    expression = channel._process_sample(  # pylint: disable=protected-access
        context,
        channel.samples[0],
        1,
    )
    return float(
        np.asarray(pytensor.function([], expression, mode="FAST_COMPILE")()).item()
    )


def test_normsys_additive_entries_share_one_flexible_group() -> None:
    channel = _channel(
        [
            _normsys("first", "a", 1.2, 0.8, ADD_LINEAR),
            _normsys("second", "b", 1.1, 0.9, ADD_LINEAR),
        ]
    )

    # 10 * (1 + 0.5*(1.2-1) + 0.5*(1.1-1)) = 11.5. Treating the
    # entries as independent factors would incorrectly produce 11.55.
    assert _evaluate(channel, a=0.5, b=0.5) == pytest.approx(11.5)


def test_mixed_normsys_order_matches_flexible_interp_var() -> None:
    additive = _normsys("add", "a", 1.4, 0.6, ADD_LINEAR)
    multiplicative = _normsys("mult", "b", 2.0, 0.5, MULT_EXP)

    add_then_mult = _evaluate(_channel([additive, multiplicative]), a=0.5, b=0.5)
    mult_then_add = _evaluate(_channel([multiplicative, additive]), a=0.5, b=0.5)

    assert add_then_mult == pytest.approx(10.0 * 1.2 * np.sqrt(2.0))
    assert mult_then_add == pytest.approx(10.0 * (np.sqrt(2.0) + 0.2))
    assert add_then_mult != pytest.approx(mult_then_add)


def test_mixed_histosys_order_matches_piecewise_interpolation() -> None:
    additive = _histosys("add", "a", 14.0, 6.0, ADD_LINEAR)
    multiplicative = _histosys("mult", "b", 20.0, 5.0, MULT_EXP)

    add_then_mult = _evaluate(_channel([additive, multiplicative]), a=0.5, b=0.5)
    mult_then_add = _evaluate(_channel([multiplicative, additive]), a=0.5, b=0.5)

    assert add_then_mult == pytest.approx(12.0 * np.sqrt(2.0))
    assert mult_then_add == pytest.approx(10.0 * np.sqrt(2.0) + 2.0)
    assert add_then_mult != pytest.approx(mult_then_add)


def test_default_is_validated_only_after_modifier_override() -> None:
    channel = _channel(
        [
            _normsys("norm", "a", 1.2, 0.8, MULT_POLY6),
            _histosys("shape", "b", 12.0, 8.0),
        ],
        default=ADD_POLY6,
    )
    normsys, histosys = channel.samples[0].modifiers
    assert isinstance(normsys, NormSysModifier)
    assert isinstance(histosys, HistoSysModifier)
    assert normsys.resolved_interpolation.key == ("mult", "poly6", "exp")
    assert histosys.resolved_interpolation.key == ("add", "poly6", "poly1")

    dumped = channel.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert dumped["default_interpolation"] == ADD_POLY6
    assert dumped["samples"][0]["modifiers"][0]["interpolation"] == MULT_POLY6
    assert "interpolation" not in dumped["samples"][0]["modifiers"][1]


def test_piecewise_only_form_is_rejected_for_unoverridden_normsys() -> None:
    with pytest.raises(ValidationError, match="FlexibleInterpVar"):
        _channel(
            [_normsys("norm", "a", 1.2, 0.8, ADD_POLY6)],
        )


def test_group_positivity_is_applied_after_all_entries() -> None:
    norm = _channel([_normsys("norm", "a", -3.0, 0.8, ADD_LINEAR)])
    shape = _channel([_histosys("shape", "a", -5.0, 8.0, ADD_LINEAR)])

    assert _evaluate(norm, a=1.0) == pytest.approx(10.0 * np.finfo(np.float64).tiny)
    assert _evaluate(shape, a=2.0) == 0.0


def test_histfactory_normsys_poly6_exp_clamps_only_evaluation_anchors() -> None:
    channel = _channel([_normsys("norm", "a", 0.0, -1.0, MULT_POLY6)])

    # ROOT protects non-positive anchors for HistFactory's FlexibleInterpVar
    # construction, but keeps the serialized payload unchanged.
    assert _evaluate(channel, a=1.0) == pytest.approx(10.0 * np.finfo(np.float64).eps)
    dumped_data = channel.model_dump(mode="json", by_alias=True)["samples"][0][
        "modifiers"
    ][0]["data"]
    assert dumped_data == {"hi": 0.0, "lo": -1.0}


def test_legacy_modifier_code_is_canonicalized_on_dump() -> None:
    channel = HistFactoryDistChannel.model_validate(
        {
            "name": "channel",
            "type": "histfactory_dist",
            "axes": [{"name": "x", "min": 0.0, "max": 1.0, "nbins": 1}],
            "samples": [
                {
                    "name": "sample",
                    "data": {"contents": [10.0]},
                    "modifiers": [
                        {
                            "name": "norm",
                            "type": "normsys",
                            "parameter": "a",
                            "data": {
                                "hi": 1.2,
                                "lo": 0.8,
                                "interpolation": "code4",
                            },
                        }
                    ],
                }
            ],
        }
    )

    dumped = channel.model_dump(mode="json", by_alias=True, exclude_none=True)
    modifier = dumped["samples"][0]["modifiers"][0]
    assert modifier["interpolation"] == MULT_POLY6
    assert "interpolation" not in modifier["data"]
