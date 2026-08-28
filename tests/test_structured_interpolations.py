"""Structured ROOT HS3 interpolation descriptors and standalone functions."""

from __future__ import annotations

import itertools

import numpy as np
import pytensor.tensor as pt
import pytest
from pydantic import ValidationError
from pytensor import function
from pytensor.graph.traversal import io_toposort

from pyhs3.distributions.histfactory.interpolations import (
    InterpolationDescriptor,
    _code4_coefficient_tensor,
    _code4_exponent_tensor,
    apply_interpolation_descriptor,
    expand_interpolations,
)
from pyhs3.functions import (
    Functions,
    Interpolation0DFunction,
    InterpolationFunction,
)
from pyhs3.transpile import jaxify

PIECEWISE_FORMS = [
    {"type": "add", "in": "poly1", "out": None},
    {"type": "mult", "in": "exp", "out": None},
    {"type": "add", "in": "poly2", "out": "poly1"},
    {"type": "add", "in": "poly6", "out": "poly1"},
    {"type": "mult", "in": "poly6", "out": "exp"},
    {"type": "mult", "in": "poly6", "out": "poly1"},
]
FLEXIBLE_FORMS = [PIECEWISE_FORMS[index] for index in (0, 1, 2, 4)]


def test_descriptor_accepts_exactly_six_of_all_40_token_combinations():
    """Pin ROOT eac146db's complete 2 x 4 x 5 descriptor matrix."""
    raw_forms = [
        {"type": type_, "in": inside, "out": outside}
        for type_, inside, outside in itertools.product(
            ("add", "mult"),
            ("poly1", "poly2", "poly6", "exp"),
            (None, "poly1", "poly2", "poly6", "exp"),
        )
    ]
    accepted = []

    for form in raw_forms:
        try:
            descriptor = InterpolationDescriptor.model_validate(form)
        except ValidationError:
            continue
        accepted.append(descriptor.model_dump(exclude_none=True))

    assert len(raw_forms) == 40
    assert {(form["type"], form["in"], form["out"]) for form in accepted} == {
        (form["type"], form["in"], form["out"]) for form in PIECEWISE_FORMS
    }


@pytest.mark.parametrize("missing", ["type", "in", "out"])
def test_descriptor_requires_every_key(missing):
    form = dict(PIECEWISE_FORMS[0])
    del form[missing]
    with pytest.raises(ValidationError, match="Field required"):
        InterpolationDescriptor.model_validate(form)


def test_descriptor_rejects_extra_keys():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        InterpolationDescriptor.model_validate({**PIECEWISE_FORMS[0], "boundary": 1.0})


@pytest.mark.parametrize("form", PIECEWISE_FORMS[3::2])
def test_flexible_rejects_piecewise_only_forms(form):
    descriptor = InterpolationDescriptor.model_validate(form)
    with pytest.raises(ValueError, match="cannot be represented by FlexibleInterpVar"):
        expand_interpolations([descriptor], 1, "flexible")


# Generated with local ROOT commit eac146db1e842e1881c3167cbc3b656a9516a591,
# using nominal=10, low=6, high=15. The four points immediately around +/-1
# pin both the polynomial and extrapolation branches without requiring ROOT in CI.
ROOT_THETA = np.asarray(
    [
        -2.0,
        -1.000001,
        -1.0,
        -0.999999,
        -0.5,
        0.0,
        0.5,
        0.999999,
        1.0,
        1.000001,
        2.0,
    ]
)
ROOT_GOLDENS = {
    ("piecewise", 0): [
        2.0,
        5.9999960000000003,
        6.0,
        6.0000040000000006,
        8.0,
        10.0,
        12.5,
        14.999995,
        15.0,
        15.000005,
        20.0,
    ],
    ("piecewise", 1): [
        3.5999999999999996,
        5.9999969350470401,
        6.0,
        6.0000030649545257,
        7.745966692414834,
        10.0,
        12.24744871391589,
        14.999993918024613,
        15.0,
        15.000006081977855,
        22.5,
    ],
    ("piecewise", 2): [
        2.5,
        5.9999965,
        6.0,
        6.0000035000005001,
        7.875,
        10.0,
        12.375,
        14.999994500000501,
        15.0,
        15.0000055,
        20.5,
    ],
    ("piecewise", 3): [
        2.0,
        5.9999960000000003,
        6.0,
        6.0000040000000006,
        7.9482421875,
        10.0,
        12.4482421875,
        14.999995,
        15.0,
        15.000005,
        20.0,
    ],
    ("piecewise", 4): [
        3.5999999999999996,
        5.9999969350470401,
        6.0,
        6.0000030649545248,
        7.8061864253973212,
        10.0,
        12.294794137478638,
        14.999993918024613,
        15.0,
        15.000006081977855,
        22.5,
    ],
    ("piecewise", 5): [
        2.0,
        5.9999960000000003,
        6.0,
        6.0000039999999997,
        7.9482421875,
        10.0,
        12.4482421875,
        14.999995,
        15.0,
        15.000005,
        20.0,
    ],
    ("flexible", 0): [
        2.0,
        5.9999960000000003,
        6.0,
        6.0000040000000006,
        8.0,
        10.0,
        12.5,
        14.999995,
        15.0,
        15.000005,
        20.0,
    ],
    ("flexible", 1): [
        3.5999999999999996,
        5.9999969350470401,
        6.0,
        6.0000030649545257,
        7.745966692414834,
        10.0,
        12.24744871391589,
        14.999993918024613,
        15.0,
        15.000006081977855,
        22.5,
    ],
    ("flexible", 2): [
        2.5,
        5.9999965,
        6.0,
        6.0000035000005001,
        7.875,
        10.0,
        12.375,
        14.999994500000501,
        15.0,
        15.0000055,
        20.5,
    ],
    ("flexible", 3): [
        3.5999999999999996,
        5.9999969350470401,
        6.0,
        6.0000030649545248,
        7.8061864253973212,
        10.0,
        12.294794137478638,
        14.999993918024613,
        15.0,
        15.000006081977855,
        22.5,
    ],
}


@pytest.mark.parametrize(
    ("interpolation_class", "form_index"),
    [
        *(("piecewise", index) for index in range(len(PIECEWISE_FORMS))),
        *(("flexible", index) for index in range(len(FLEXIBLE_FORMS))),
    ],
)
def test_structured_forms_match_root_eac146db_in_pytensor_and_jax(
    interpolation_class, form_index
):
    pytest.importorskip("jax")
    theta = pt.dvector("theta")

    if interpolation_class == "piecewise":
        interpolation = InterpolationFunction(
            name="piecewise",
            nom="nominal",
            low=["low"],
            high=["high"],
            vars=["theta"],
            positiveDefinite=False,
            interpolations=[PIECEWISE_FORMS[form_index]],
        )
        expression = interpolation.expression(
            {
                "theta": theta,
                "nominal": pt.constant(10.0, dtype="float64"),
                "low": pt.constant(6.0, dtype="float64"),
                "high": pt.constant(15.0, dtype="float64"),
            }
        )
    else:
        interpolation = Interpolation0DFunction(
            name="flexible",
            nom=10.0,
            low=[6.0],
            high=[15.0],
            vars=["theta"],
            interpolations=[FLEXIBLE_FORMS[form_index]],
        )
        expression = interpolation.expression({"theta": theta})

    expected = ROOT_GOLDENS[(interpolation_class, form_index)]
    pytensor_result = function([theta], expression, mode="FAST_COMPILE")(ROOT_THETA)
    jax_result = np.asarray(jaxify(expression, inputs=[theta])(theta=ROOT_THETA)[0])

    np.testing.assert_allclose(pytensor_result, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(jax_result, expected, rtol=2e-13, atol=2e-13)


def test_structured_poly6_exp_uses_code4_caches_and_bin_vectorized_graph():
    descriptor = InterpolationDescriptor.model_validate(PIECEWISE_FORMS[4])
    _code4_coefficient_tensor.cache_clear()
    _code4_exponent_tensor.cache_clear()

    def build_graph(n_bins):
        theta = pt.dscalar("theta")
        nominal = pt.constant(np.full(n_bins, 10.0))
        high = pt.constant(np.full(n_bins, 15.0))
        low = pt.constant(np.full(n_bins, 6.0))
        return apply_interpolation_descriptor(
            descriptor,
            theta,
            nominal,
            high,
            low,
        )

    one_bin = build_graph(1)
    many_bins = build_graph(4096)

    assert _code4_coefficient_tensor.cache_info().hits == 1
    assert _code4_coefficient_tensor.cache_info().misses == 1
    assert _code4_exponent_tensor.cache_info().hits == 1
    assert _code4_exponent_tensor.cache_info().misses == 1
    assert len(io_toposort([], [one_bin])) == len(io_toposort([], [many_bins]))
    assert one_bin.type.shape == (1,)
    assert many_bins.type.shape == (4096,)


def test_piecewise_broadcasts_one_descriptor_and_canonicalizes_repeats():
    config = {
        "name": "pip",
        "type": "interpolation",
        "nom": "nom",
        "low": ["lo1", "lo2"],
        "high": ["hi1", "hi2"],
        "vars": ["alpha", "beta"],
        "positiveDefinite": False,
        "interpolations": [PIECEWISE_FORMS[0], PIECEWISE_FORMS[0]],
    }
    interpolation = InterpolationFunction.model_validate(config)

    assert len(interpolation.interpolations) == 1
    assert interpolation.model_dump(exclude_none=True)["interpolations"] == [
        PIECEWISE_FORMS[0]
    ]
    assert len(expand_interpolations(interpolation.interpolations, 2, "piecewise")) == 2


def test_empty_vars_require_empty_interpolations():
    with pytest.raises(ValidationError, match="got 1 for 0 parameters"):
        InterpolationFunction(
            name="empty",
            nom="nom",
            low=[],
            high=[],
            vars=[],
            positiveDefinite=False,
            interpolations=[PIECEWISE_FORMS[0]],
        )


def test_piecewise_sequential_order_matches_root():
    common = {
        "nom": "nom",
        "low": ["add_lo", "mult_lo"],
        "high": ["add_hi", "mult_hi"],
        "vars": ["alpha", "beta"],
        "positiveDefinite": False,
    }
    add_then_mult = InterpolationFunction(
        name="add_then_mult",
        interpolations=[PIECEWISE_FORMS[0], PIECEWISE_FORMS[1]],
        **common,
    )
    mult_then_add = InterpolationFunction(
        name="mult_then_add",
        low=list(reversed(common["low"])),
        high=list(reversed(common["high"])),
        vars=list(reversed(common["vars"])),
        nom="nom",
        positiveDefinite=False,
        interpolations=[PIECEWISE_FORMS[1], PIECEWISE_FORMS[0]],
    )
    context = {
        "nom": pt.constant(10.0),
        "add_lo": pt.constant(8.0),
        "add_hi": pt.constant(12.0),
        "mult_lo": pt.constant(5.0),
        "mult_hi": pt.constant(20.0),
        "alpha": pt.constant(1.0),
        "beta": pt.constant(1.0),
    }

    assert function([], add_then_mult.expression(context))() == pytest.approx(24.0)
    assert function([], mult_then_add.expression(context))() == pytest.approx(22.0)


def test_piecewise_positive_definite_clamps_to_zero_once_at_end():
    interpolation = InterpolationFunction(
        name="pip",
        nom="nom",
        low=["lo"],
        high=["hi"],
        vars=["alpha"],
        positiveDefinite=True,
        interpolations=[PIECEWISE_FORMS[0]],
    )
    context = {
        "nom": pt.constant(5.0),
        "lo": pt.constant(8.0),
        "hi": pt.constant(2.0),
        "alpha": pt.constant(2.0),
    }

    assert function([], interpolation.expression(context))() == 0.0


@pytest.mark.parametrize("form", FLEXIBLE_FORMS)
def test_interpolation0d_supports_every_flexible_form_at_nominal(form):
    interpolation = Interpolation0DFunction(
        name="fiv",
        nom=10.0,
        low=[8.0],
        high=[12.0],
        vars=["alpha"],
        interpolations=[form],
    )
    result = function([], interpolation.expression({"alpha": pt.constant(0.0)}))()

    assert result == pytest.approx(10.0)
    assert interpolation.parameters == {"alpha"}


def test_interpolation0d_nonpositive_result_uses_dtype_tiny():
    interpolation = Interpolation0DFunction(
        name="fiv",
        nom=1.0,
        low=[2.0],
        high=[-1.0],
        vars=["alpha"],
        interpolations=[FLEXIBLE_FORMS[0]],
    )
    result = function([], interpolation.expression({"alpha": pt.constant(2.0)}))()

    assert result == np.finfo(np.asarray(result).dtype).tiny


def test_functions_collection_parses_interpolation0d():
    functions = Functions(
        [
            {
                "type": "interpolation0d",
                "name": "fiv",
                "nom": 1.0,
                "low": [0.8],
                "high": [1.2],
                "vars": ["alpha"],
                "interpolations": [FLEXIBLE_FORMS[-1]],
            }
        ]
    )

    assert isinstance(functions["fiv"], Interpolation0DFunction)


@pytest.mark.parametrize("function_type", ["interpolation", "interpolation0d"])
def test_standalone_functions_reject_legacy_codes(function_type):
    config = {
        "type": function_type,
        "name": "legacy",
        "nom": "nom" if function_type == "interpolation" else 1.0,
        "low": ["lo"] if function_type == "interpolation" else [0.8],
        "high": ["hi"] if function_type == "interpolation" else [1.2],
        "vars": ["alpha"],
        "interpolationCodes": [0],
    }
    if function_type == "interpolation":
        config["positiveDefinite"] = False

    with pytest.raises(ValidationError, match="interpolationCodes"):
        Functions([config])
