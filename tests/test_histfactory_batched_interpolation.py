"""Tests for homogeneous HistFactory interpolation-group batching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.configdefaults import config
from pytensor.graph.traversal import io_toposort

import pyhs3.distributions.histfactory as histfactory_module
from pyhs3.context import Context
from pyhs3.distributions import HistFactoryDistChannel
from pyhs3.distributions.histfactory.interpolations import (
    _code4_coefficient_tensor,
    _code4_exponent_tensor,
    interpolate_code4,
    interpolate_poly6,
)
from pyhs3.distributions.histfactory.modifiers import (
    HistoSysModifier,
    NormSysModifier,
)
from pyhs3.transpile import jaxify
from pyhs3.typing.aliases import TensorVar

ADD_POLY6 = {"type": "add", "in": "poly6", "out": "poly1"}
MULT_POLY6 = {"type": "mult", "in": "poly6", "out": "exp"}
NOMINAL = [10.0, 7.0, 4.0]

type GroupKind = Literal["normsys", "histosys"]


def _normsys(
    name: str,
    parameter: str,
    high: float,
    low: float,
) -> dict[str, object]:
    return {
        "name": name,
        "type": "normsys",
        "parameter": parameter,
        "interpolation": MULT_POLY6,
        "data": {"hi": high, "lo": low},
    }


def _histosys(
    name: str,
    parameter: str,
    high: Sequence[float],
    low: Sequence[float],
) -> dict[str, object]:
    return {
        "name": name,
        "type": "histosys",
        "parameter": parameter,
        "interpolation": ADD_POLY6,
        "data": {
            "hi": {"contents": list(high)},
            "lo": {"contents": list(low)},
        },
    }


def _channel(modifiers: Sequence[dict[str, object]]) -> HistFactoryDistChannel:
    return HistFactoryDistChannel.model_validate(
        {
            "name": "channel",
            "type": "histfactory_dist",
            "axes": [{"name": "x", "min": 0.0, "max": 3.0, "nbins": 3}],
            "samples": [
                {
                    "name": "sample",
                    "data": {"contents": NOMINAL},
                    "modifiers": list(modifiers),
                }
            ],
        }
    )


def _process(channel: HistFactoryDistChannel, context: Context) -> TensorVar:
    return channel._process_sample(context, channel.samples[0], len(NOMINAL))


def _sequential_normsys_reference(
    channel: HistFactoryDistChannel,
    context: Context,
) -> TensorVar:
    """Evaluate an E group one entry at a time through the low-level kernel."""
    nominal = pt.constant(np.asarray(NOMINAL, dtype=config.floatX))
    one = pt.constant(1.0, dtype=nominal.dtype)
    factor = one

    for modifier in channel.samples[0].modifiers:
        assert isinstance(modifier, NormSysModifier)
        alpha, high, low = modifier.interpolation_inputs(context)
        factor = factor * interpolate_code4(alpha, one, high, low)

    tiny = pt.constant(np.finfo(np.dtype(factor.dtype)).tiny, dtype=factor.dtype)
    factor = pt.where(factor <= 0, tiny, factor)
    return nominal * factor


def _sequential_histosys_reference(
    channel: HistFactoryDistChannel,
    context: Context,
) -> TensorVar:
    """Evaluate a D group one entry at a time through the low-level kernel."""
    nominal = pt.constant(np.asarray(NOMINAL, dtype=config.floatX))
    current = nominal

    for modifier in channel.samples[0].modifiers:
        assert isinstance(modifier, HistoSysModifier)
        alpha, high, low = modifier.interpolation_inputs(context)
        current = current + interpolate_poly6(alpha, nominal, high, low) - nominal

    return pt.maximum(current, pt.constant(0.0, dtype=current.dtype))


def _repeated_normsys_channel() -> HistFactoryDistChannel:
    return _channel(
        [
            _normsys("first", "shared", 1.20, 0.82),
            _normsys("second", "beta", 0.91, 1.13),
            _normsys("third", "shared", 1.07, 0.96),
        ]
    )


def _repeated_histosys_channel() -> HistFactoryDistChannel:
    return _channel(
        [
            _histosys("first", "shared", [12.0, 9.0, 5.0], [8.0, 6.0, 3.5]),
            _histosys("second", "beta", [10.5, 8.4, 4.8], [9.7, 6.5, 3.8]),
            _histosys("third", "shared", [11.0, 7.5, 4.2], [9.5, 6.8, 3.7]),
        ]
    )


def test_batched_scalar_normsys_e_matches_sequential_low_level_reference() -> None:
    channel = _repeated_normsys_channel()
    shared = pt.dscalar("shared")
    beta = pt.dscalar("beta")
    context = Context({"shared": shared, "beta": beta})

    batched = _process(channel, context)
    sequential = _sequential_normsys_reference(channel, context)
    evaluate = pytensor.function(
        [shared, beta],
        [batched, sequential],
        mode="FAST_COMPILE",
    )

    batched_value, sequential_value = evaluate(0.35, -1.25)
    np.testing.assert_allclose(batched_value, sequential_value, rtol=2e-13, atol=0.0)


def test_batched_scalar_histosys_d_matches_sequential_low_level_reference() -> None:
    channel = _repeated_histosys_channel()
    shared = pt.dscalar("shared")
    beta = pt.dscalar("beta")
    context = Context({"shared": shared, "beta": beta})

    batched = _process(channel, context)
    sequential = _sequential_histosys_reference(channel, context)
    evaluate = pytensor.function(
        [shared, beta],
        [batched, sequential],
        mode="FAST_COMPILE",
    )

    batched_value, sequential_value = evaluate(-0.45, 1.35)
    np.testing.assert_allclose(batched_value, sequential_value, rtol=2e-13, atol=0.0)


def test_batched_scalar_normsys_e_matches_sequential_reference_with_jax() -> None:
    pytest.importorskip("jax")
    channel = _repeated_normsys_channel()
    shared = pt.dscalar("shared")
    beta = pt.dscalar("beta")
    context = Context({"shared": shared, "beta": beta})

    batched = _process(channel, context)
    sequential = _sequential_normsys_reference(channel, context)
    values = {"shared": 0.35, "beta": -1.25}
    expected = pytensor.function(
        [shared, beta],
        sequential,
        mode="FAST_COMPILE",
    )(*values.values())
    result = np.asarray(jaxify(batched, inputs=[shared, beta])(**values)[0])

    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=0.0)


def test_batched_normsys_e_reuses_one_code4_cache_entry_per_group() -> None:
    channel = _repeated_normsys_channel()
    _code4_coefficient_tensor.cache_clear()
    _code4_exponent_tensor.cache_clear()

    for graph_index in range(2):
        context = Context(
            {
                "shared": pt.dscalar(f"shared_{graph_index}"),
                "beta": pt.dscalar(f"beta_{graph_index}"),
            }
        )
        _process(channel, context)

        expected_hits = graph_index
        coefficient_info = _code4_coefficient_tensor.cache_info()
        exponent_info = _code4_exponent_tensor.cache_info()
        assert coefficient_info.misses == 1
        assert coefficient_info.hits == expected_hits
        assert exponent_info.misses == 1
        assert exponent_info.hits == expected_hits


def _homogeneous_channel(
    kind: GroupKind, modifier_count: int
) -> HistFactoryDistChannel:
    if kind == "normsys":
        modifiers = [
            _normsys(
                f"norm_{index}",
                f"alpha_{index}",
                1.05 + 0.001 * index,
                0.96 - 0.001 * index,
            )
            for index in range(modifier_count)
        ]
    else:
        modifiers = [
            _histosys(
                f"shape_{index}",
                f"alpha_{index}",
                [11.0 + 0.01 * index, 8.0, 5.0],
                [9.0 - 0.01 * index, 6.0, 3.5],
            )
            for index in range(modifier_count)
        ]
    return _channel(modifiers)


def _homogeneous_graph_size(kind: GroupKind, modifier_count: int) -> int:
    channel = _homogeneous_channel(kind, modifier_count)
    parameters = {
        f"alpha_{index}": pt.dscalar(f"{kind}_alpha_{modifier_count}_{index}")
        for index in range(modifier_count)
    }
    expression = _process(channel, Context(parameters))
    return len(io_toposort(list(parameters.values()), [expression]))


@pytest.mark.parametrize("kind", ["normsys", "histosys"])
def test_homogeneous_scalar_group_graph_growth_is_bounded(kind: GroupKind) -> None:
    two_modifiers = _homogeneous_graph_size(kind, 2)
    twenty_modifiers = _homogeneous_graph_size(kind, 20)

    # A batched kernel contributes a fixed interpolation graph. The D path may
    # add at most two template-broadcast nodes per extra modifier; the E path
    # is fully constant-size. A sequential graph would grow by dozens of nodes
    # per modifier and fail both bounds.
    assert twenty_modifiers <= two_modifiers + 2 * (20 - 2)
    assert twenty_modifiers < 2 * two_modifiers


@pytest.mark.parametrize("kind", ["normsys", "histosys"])
@pytest.mark.parametrize(
    ("alpha_type", "alpha_value"),
    [
        ("vector", np.asarray([-0.4, 0.2, 1.3])),
        ("matrix", np.asarray([[-0.4, 0.2, 1.3], [0.7, -1.2, 0.1]])),
    ],
)
def test_nonscalar_homogeneous_groups_use_sequential_fallback(
    monkeypatch: pytest.MonkeyPatch,
    kind: GroupKind,
    alpha_type: Literal["vector", "matrix"],
    alpha_value: np.ndarray,
) -> None:
    if kind == "normsys":
        channel = _channel(
            [
                _normsys("first", "alpha", 1.20, 0.82),
                _normsys("second", "alpha", 0.91, 1.13),
            ]
        )
        reference_builder = _sequential_normsys_reference
        batched_kernel = "interpolate_code4"
    else:
        channel = _channel(
            [
                _histosys("first", "alpha", [12.0, 9.0, 5.0], [8.0, 6.0, 3.5]),
                _histosys("second", "alpha", [10.5, 8.4, 4.8], [9.7, 6.5, 3.8]),
            ]
        )
        reference_builder = _sequential_histosys_reference
        batched_kernel = "interpolate_poly6"

    alpha = pt.dvector("alpha") if alpha_type == "vector" else pt.dmatrix("alpha")
    context = Context({"alpha": alpha})
    sequential = reference_builder(channel, context)

    def fail_if_batched(*_: object, **__: object) -> None:
        pytest.fail("non-scalar interpolation parameters must use the sequential path")

    # Patch only the channel's direct batching aliases. The low-level kernels
    # used by the ordinary modifier path remain available for the reference.
    monkeypatch.setattr(histfactory_module, batched_kernel, fail_if_batched)
    result = _process(channel, context)
    evaluate = pytensor.function(
        [alpha],
        [result, sequential],
        mode="FAST_COMPILE",
    )

    result_value, sequential_value = evaluate(alpha_value)
    np.testing.assert_allclose(result_value, sequential_value, rtol=2e-13, atol=0.0)
