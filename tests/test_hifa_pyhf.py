"""Integration tests: ``pyhs3.from_hifa`` against pyhf.

The specs are pyhf's validation models from ``tests/test_validation.py`` and
``validation/data/*.json`` in pyhf v0.7.6, restricted to ``normfactor``,
``normsys`` and ``histosys``: ``2bin_2channel`` uses a ``normsys`` per channel
where pyhf's original uses ``shapesys``. Each spec carries a measurement
config, so suggested inits, bounds and fixed parameters are compared too.

pyhf builds each model with the interpolation codes ``from_hifa`` defaults to,
so the two likelihoods agree for every parameter value, not just at alpha = 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pytensor import function

from pyhs3 import Workspace, from_hifa
from pyhs3.context import Context

pyhf = pytest.importorskip("pyhf")

MODIFIER_SETTINGS = {
    "normsys": {"interpcode": "code4"},
    "histosys": {"interpcode": "code4p"},
}
RTOL = 1e-10
SEEDS = [0, 1, 2]


def normfactor(name: str) -> dict[str, Any]:
    return {"name": name, "type": "normfactor", "data": None}


def normsys(name: str, lo: float, hi: float) -> dict[str, Any]:
    return {"name": name, "type": "normsys", "data": {"lo": lo, "hi": hi}}


def histosys(name: str, lo: list[float], hi: list[float]) -> dict[str, Any]:
    return {"name": name, "type": "histosys", "data": {"lo_data": lo, "hi_data": hi}}


def sample(name: str, data: list[float], *modifiers: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "data": data, "modifiers": list(modifiers)}


def workspace(
    channels: dict[str, list[dict[str, Any]]],
    observations: dict[str, list[float]],
    parameters: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "channels": [
            {"name": name, "samples": samples} for name, samples in channels.items()
        ],
        "observations": [
            {"name": name, "data": data} for name, data in observations.items()
        ],
        "measurements": [
            {"name": "meas", "config": {"poi": "mu", "parameters": parameters}}
        ],
        "version": "1.0.0",
    }


SPECS = {
    "1bin_normsys": workspace(
        {
            "singlechannel": [
                sample("signal", [30.0, 95.0], normfactor("mu")),
                sample("background", [100.0, 150.0], normsys("bkg_norm", 0.9, 1.1)),
            ]
        },
        {"singlechannel": [120.0, 180.0]},
        [],
    ),
    "2bin_histosys": workspace(
        {
            "singlechannel": [
                sample("signal", [30.0, 95.0], normfactor("mu")),
                sample(
                    "background",
                    [100.0, 150.0],
                    histosys("bkg_norm", [98.0, 100.0], [102.0, 190.0]),
                ),
            ]
        },
        {"singlechannel": [120.0, 180.0]},
        [{"name": "mu", "inits": [1.5], "bounds": [[0.0, 5.0]]}],
    ),
    "2bin_2channel": workspace(
        {
            "signal": [
                sample("signal", [10.0, 35.0], normfactor("mu")),
                sample(
                    "background",
                    [100.0, 150.0],
                    normsys("uncorr_bkguncrt_signal", 0.9, 1.1),
                ),
            ],
            "control": [
                sample(
                    "background",
                    [200.0, 350.0],
                    normsys("uncorr_bkguncrt_control", 0.95, 1.05),
                ),
            ],
        },
        {"signal": [110.0, 155.0], "control": [205.0, 345.0]},
        [{"name": "uncorr_bkguncrt_control", "inits": [0.3], "fixed": True}],
    ),
    "couplednorm": workspace(
        {
            "signal": [
                sample("signal", [10.0, 35.0], normfactor("mu")),
                sample("bkg1", [100.0, 100.0], normsys("coupled_normsys", 0.9, 1.1)),
                sample("bkg2", [50.0, 100.0], normsys("coupled_normsys", 0.5, 1.5)),
            ],
            "control": [
                sample(
                    "background", [100.0, 100.0], normsys("coupled_normsys", 0.9, 1.1)
                ),
            ],
        },
        {"signal": [105.0, 220.0], "control": [110.0, 105.0]},
        [
            {"name": "coupled_normsys", "inits": [0.2], "bounds": [[-3.0, 3.0]]},
            {"name": "not_in_model", "inits": [7.0]},
        ],
    ),
    "coupledhistosys": workspace(
        {
            "signal": [
                sample("signal", [30.0, 35.0], normfactor("mu")),
                sample(
                    "bkg1",
                    [100.0, 100.0],
                    histosys("coupled_histosys", [90.0, 90.0], [110.0, 110.0]),
                ),
                sample(
                    "bkg2",
                    [50.0, 120.0],
                    histosys("coupled_histosys", [45.0, 45.0], [55.0, 55.0]),
                ),
            ],
            "control": [
                sample(
                    "background",
                    [105.0, 100.0],
                    histosys("coupled_histosys", [95.0, 90.0], [110.0, 115.0]),
                ),
            ],
        },
        {"signal": [170.0, 220.0], "control": [110.0, 105.0]},
        [{"name": "mu", "inits": [0.5]}],
    ),
}


@pytest.fixture(params=list(SPECS), scope="module")
def models(request):
    """The pyhf model and data, and the converted pyhs3 workspace."""
    spec = SPECS[request.param]
    pyhf_ws = pyhf.Workspace(spec)
    pyhf_model = pyhf_ws.model(modifier_settings=MODIFIER_SETTINGS)
    return pyhf_model, pyhf_ws.data(pyhf_model), from_hifa(spec)


def by_name(pyhf_model, values) -> dict[str, Any]:
    """pyhf per-parameter values (in ``par_order``) keyed by parameter name.

    Every parameter here is scalar, so each name owns one entry.
    """
    return dict(zip(pyhf_model.config.par_names, values, strict=True))


def random_points(ws: Workspace, seed: int, n: int = 4) -> list[dict[str, float]]:
    """``n`` parameter points drawn inside the middle of each parameter's bounds.

    Fixed parameters stay at their initial value: pyhs3 bakes them into the
    likelihood as constants.
    """
    rng = np.random.default_rng(seed)
    domain = ws.domains["meas"]
    points = []
    for _ in range(n):
        point = {}
        for p in ws.parameter_points["meas"]:
            lo, hi = domain[p.name]
            point[p.name] = (
                p.value if p.const else lo + (hi - lo) * rng.uniform(0.3, 0.7)
            )
        points.append(point)
    return points


def compiled_logpdf(ws: Workspace):
    """pyhs3's log-likelihood (main Poisson terms plus Gauss constraints).

    The returned callable takes every parameter by name; the compiled NLL only
    takes the free ones, with the fixed ones baked in at their initial values.
    """
    nll = ws.model(ws.analyses["meas"], progress=False).compile_nll(gradient=False)

    def logpdf(point: dict[str, float]) -> float:
        free = {name: point[name] for name in nll.parameter_names}
        return -nll.value(nll.to_vector(free))

    return logpdf


def pyhf_logpdf(pyhf_model, data, point: dict[str, float]) -> float:
    pars = [point[name] for name in pyhf_model.config.par_names]
    return float(pyhf_model.logpdf(pars, data)[0])


def pyhs3_expected(ws: Workspace, channel: str, point: dict[str, float]):
    """Expected rates of one channel, shape (nbins,)."""
    dist = ws.distributions[f"model_{channel}"]
    names = sorted(dist.parameters - {f"{dist.name}_observed"})
    scalars = [pt.dscalar(name) for name in names]
    rates = dist._compute_expected_rates(
        Context(dict(zip(names, scalars, strict=True))), dist.axes.get_total_bins()
    )
    return function(scalars, rates)(*[point[name] for name in names])


def test_suggested_init(models):
    pyhf_model, _, ws = models
    expected = by_name(pyhf_model, pyhf_model.config.suggested_init())
    assert {p.name: p.value for p in ws.parameter_points["meas"]} == pytest.approx(
        expected, rel=RTOL
    )


def test_suggested_bounds(models):
    pyhf_model, _, ws = models
    expected = by_name(pyhf_model, pyhf_model.config.suggested_bounds())
    domain = ws.domains["meas"]
    assert set(domain.axis_names) == set(expected)
    for name, (lo, hi) in expected.items():
        assert domain[name] == pytest.approx((lo, hi), rel=RTOL)


def test_suggested_fixed(models):
    pyhf_model, _, ws = models
    expected = by_name(pyhf_model, pyhf_model.config.suggested_fixed())
    assert {p.name: p.const for p in ws.parameter_points["meas"]} == expected


def test_poi(models):
    pyhf_model, _, ws = models
    assert ws.analyses["meas"].parameters_of_interest == [pyhf_model.config.poi_name]


@pytest.mark.parametrize("seed", SEEDS)
def test_expected_rates(models, seed):
    pyhf_model, _, ws = models
    for point in random_points(ws, seed):
        pars = [point[name] for name in pyhf_model.config.par_names]
        # (total bins,) concatenated over pyhf's channels, sliced per channel.
        expected = pyhf_model.expected_actualdata(pars)
        for channel in pyhf_model.config.channels:
            assert_allclose(
                pyhs3_expected(ws, channel, point),
                expected[pyhf_model.config.channel_slices[channel]],
                rtol=RTOL,
            )


def test_logpdf_at_init(models):
    pyhf_model, data, ws = models
    init = {p.name: p.value for p in ws.parameter_points["meas"]}
    assert compiled_logpdf(ws)(init) == pytest.approx(
        pyhf_logpdf(pyhf_model, data, init), rel=RTOL
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_logpdf_random_points(models, seed):
    pyhf_model, data, ws = models
    logpdf = compiled_logpdf(ws)
    for point in random_points(ws, seed):
        assert logpdf(point) == pytest.approx(
            pyhf_logpdf(pyhf_model, data, point), rel=RTOL
        )
