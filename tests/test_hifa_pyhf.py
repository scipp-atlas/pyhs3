"""Integration tests: ``pyhs3.from_hifa`` against pyhf.

The specs are pyhf v0.7.6's validation models from ``tests/test_validation.py``
(``1bin_lumi`` included),
``validation/data/*.json`` and ``validation/shared_nuispar_across_types``, the
zero-uncertainty and zero-yield ``shapesys`` and the multi-sample ``staterror``
models of its ``tests/test_pdf.py``, specs written here for ``staterror``
cases those do not cover and for an observation without a channel, and the
uncorrelated simplemodel in
``tests/test_histfactory``. Most carry a measurement config, so suggested
inits, bounds and fixed parameters are compared too.

pyhf builds each model with the interpolation codes ``from_hifa`` defaults to,
so the two likelihoods agree for every parameter value, not just at alpha = 0.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
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


def shapefactor(name: str) -> dict[str, Any]:
    return {"name": name, "type": "shapefactor", "data": None}


def shapesys(name: str, data: list[float]) -> dict[str, Any]:
    return {"name": name, "type": "shapesys", "data": data}


def staterror(name: str, data: list[float]) -> dict[str, Any]:
    return {"name": name, "type": "staterror", "data": data}


LUMI = {"name": "lumi", "type": "lumi", "data": None}


def lumi_config(**overrides: Any) -> dict[str, Any]:
    """pyhf's ``1bin_lumi`` lumi config entry, with ``overrides`` applied."""
    entry = {
        "name": "lumi",
        "auxdata": [1.0],
        "sigmas": [0.1],
        "inits": [1.0],
        "bounds": [[0.0, 10.0]],
    }
    return {**entry, **overrides}


def sample(name: str, data: list[float], *modifiers: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "data": data, "modifiers": list(modifiers)}


def workspace(
    channels: dict[str, list[dict[str, Any]]],
    observations: dict[str, list[float]],
    parameters: list[dict[str, Any]],
    poi: str = "mu",
) -> dict[str, Any]:
    return {
        "channels": [
            {"name": name, "samples": samples} for name, samples in channels.items()
        ],
        "observations": [
            {"name": name, "data": data} for name, data in observations.items()
        ],
        "measurements": [
            {"name": "meas", "config": {"poi": poi, "parameters": parameters}}
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
                    shapesys("uncorr_bkguncrt_signal", [10.0, 10.0]),
                ),
            ],
            "control": [
                sample(
                    "background",
                    [200.0, 350.0],
                    shapesys("uncorr_bkguncrt_control", [5.0, 10.0]),
                ),
            ],
        },
        {"signal": [110.0, 155.0], "control": [205.0, 345.0]},
        [{"name": "uncorr_bkguncrt_control", "inits": [1.1, 0.9], "fixed": True}],
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
    "1bin_shapesys": workspace(
        {
            "singlechannel": [
                sample("signal", [10.0], normfactor("mu")),
                sample("background", [50.0], shapesys("uncorr_bkguncrt", [7.0])),
            ]
        },
        {"singlechannel": [55.0]},
        [{"name": "uncorr_bkguncrt", "bounds": [[0.5, 2.0]]}],
    ),
    "coupledshapefactor": workspace(
        {
            "signal": [
                sample("signal", [20.0, 20.0], normfactor("mu")),
                sample("bkg1", [100.0, 70.0], shapefactor("coupled_shapefactor")),
            ],
            "control": [
                sample(
                    "background", [100.0, 100.0], shapefactor("coupled_shapefactor")
                ),
            ],
        },
        {"signal": [220.0, 230.0], "control": [200.0, 300.0]},
        [{"name": "coupled_shapefactor", "inits": [1.2, 0.8]}],
    ),
    "shared_nuispar_across_types": workspace(
        {
            "channel1": [
                sample("signal", [3.0, 1.0], normfactor("SigXsecOverSM")),
                sample(
                    "background",
                    [12.0, 13.0],
                    normsys("sharednuis", 0.95, 1.05),
                    histosys("sharednuis", [10.0, 11.0], [14.0, 15.0]),
                ),
            ]
        },
        {"channel1": [15.0, 16.0]},
        [{"name": "SigXsecOverSM", "inits": [1.0], "bounds": [[0.0, 3.0]]}],
        poi="SigXsecOverSM",
    ),
    # test_shapesys_nuisparfilter_validation: a zero-uncertainty bin.
    "shapesys_nuisparfilter": workspace(
        {
            "channel1": [
                sample("signal", [20.0, 10.0], normfactor("SigXsecOverSM")),
                sample("background1", [100.0, 10.0], shapesys("syst", [10.0, 0.0])),
            ]
        },
        {"channel1": [100.0, 10.0]},
        [],
        poi="SigXsecOverSM",
    ),
    # test_pdf_integration_shapesys_zeros: a zero-yield and a zero-uncertainty bin.
    "shapesys_zeros": workspace(
        {
            "channel1": [
                sample("signal", [20.0, 10.0, 5.0, 3.0, 2.0, 1.0], normfactor("mu")),
                sample(
                    "background1",
                    [100.0, 90.0, 0.0, 70.0, 0.1, 50.0],
                    shapesys("syst", [10.0, 9.0, 1.0, 0.0, 0.1, 5.0]),
                ),
            ]
        },
        {"channel1": [120.0, 100.0, 5.0, 72.0, 2.0, 52.0]},
        [],
    ),
    # test_pdf_integration_staterror: one staterror group on two of three
    # background samples.
    "staterror_multisample": workspace(
        {
            "firstchannel": [
                sample("mu", [10.0, 10.0], normfactor("mu")),
                sample(
                    "bkg1", [50.0, 70.0], staterror("stat_firstchannel", [12.0, 12.0])
                ),
                sample(
                    "bkg2", [30.0, 20.0], staterror("stat_firstchannel", [5.0, 5.0])
                ),
                sample("bkg3", [20.0, 15.0]),
            ]
        },
        {"firstchannel": [115.0, 110.0]},
        [{"name": "stat_firstchannel", "inits": [1.05, 0.95]}],
    ),
    # One staterror name across channels, numbered over CR then SR (AA does not
    # use it), with a zero-uncertainty bin in CR and a zero-yield bin in SR.
    "staterror_cross_channel": workspace(
        {
            "SR": [
                sample("signal", [5.0, 10.0], normfactor("mu")),
                sample("bkg", [50.0, 0.0], staterror("stat_shared", [2.0, 1.0])),
            ],
            "CR": [
                sample(
                    "bkg",
                    [100.0, 80.0, 60.0],
                    staterror("stat_shared", [5.0, 0.0, 4.0]),
                ),
                sample("other", [10.0, 10.0, 10.0], normsys("other_norm", 0.9, 1.1)),
            ],
            "AA": [sample("bkg", [40.0], normsys("other_norm", 0.8, 1.2))],
        },
        {"SR": [57.0, 9.0], "CR": [112.0, 87.0, 73.0], "AA": [41.0]},
        [],
    ),
    # A lite group on two samples, plus a second staterror group on one sample
    # (a Gauss shapesys in HS3) with a zero-uncertainty bin.
    "staterror_extra_group": workspace(
        {
            "SR": [
                sample("signal", [5.0, 10.0], normfactor("mu")),
                sample(
                    "bkg1",
                    [50.0, 60.0],
                    staterror("staterror_SR", [3.0, 4.0]),
                    staterror("stat_bkg1_extra", [2.0, 0.0]),
                ),
                sample("bkg2", [30.0, 25.0], staterror("staterror_SR", [2.0, 1.0])),
            ]
        },
        {"SR": [90.0, 92.0]},
        [{"name": "stat_bkg1_extra", "inits": [0.9, 1.0]}],
    ),
    "1bin_lumi": workspace(
        {
            "channel1": [
                sample("signal", [20.0], normfactor("mu")),
                sample("background1", [100.0], LUMI),
                sample("background2", [0.0], LUMI),
            ]
        },
        {"channel1": [55.0]},
        [lumi_config()],
    ),
    # lumi on signal and background in two channels, beside a normsys, with
    # auxdata off 1.
    "lumi_two_channels": workspace(
        {
            "SR": [
                sample("signal", [10.0, 35.0], normfactor("mu"), LUMI),
                sample(
                    "background",
                    [100.0, 150.0],
                    LUMI,
                    normsys("bkg_norm", 0.9, 1.1),
                ),
            ],
            "CR": [sample("background", [200.0, 350.0], LUMI)],
        },
        {"SR": [110.0, 190.0], "CR": [205.0, 345.0]},
        [lumi_config(auxdata=[1.02], sigmas=[0.017], bounds=[[0.5, 1.5]])],
    ),
    # A fixed lumi away from its auxdata: pyhf still evaluates its constraint.
    "lumi_fixed": workspace(
        {
            "SR": [
                sample("signal", [10.0, 35.0], normfactor("mu")),
                sample("background", [100.0, 150.0], LUMI),
            ]
        },
        {"SR": [110.0, 190.0]},
        [lumi_config(inits=[1.05], fixed=True)],
    ),
    # A patch that removes channel CR leaves its observation, which pyhf
    # ignores; its bin count need not match any channel.
    "orphan_observation": workspace(
        {
            "SR": [
                sample("signal", [10.0, 35.0], normfactor("mu")),
                sample("background", [100.0, 150.0], normsys("bkg_norm", 0.9, 1.1)),
            ]
        },
        {"SR": [110.0, 190.0], "CR": [205.0, 345.0, 12.0]},
        [],
    ),
    "simplemodel_uncorrelated": json.loads(
        (
            Path(__file__).parent
            / "test_histfactory"
            / "simplemodel_uncorrelated-background_hifa.json"
        ).read_text()
    ),
}

# pyhf keeps a constraint term for each bin it fixes because the bin has no
# uncertainty or no yield, with dummy width 1 (staterror) or tau = 1
# (shapesys); pyhs3 gives such a bin no term. At gamma = 1, where such a bin
# stays fixed, the terms are log Normal(1 | 1, 1) = -log(2 pi) / 2 and
# log Poisson(1 | 1) = -1. Each entry is pyhf's log-likelihood minus pyhs3's,
# subtracted explicitly so the log-likelihoods themselves are compared.
_LOG_NORMAL_DUMMY = -0.5 * math.log(2 * math.pi)
_LOG_POISSON_DUMMY = -1.0
DUMMY_CONSTRAINTS = {
    "shapesys_nuisparfilter": _LOG_POISSON_DUMMY,
    "shapesys_zeros": 2 * _LOG_POISSON_DUMMY,
    "staterror_cross_channel": 2 * _LOG_NORMAL_DUMMY,
    "staterror_extra_group": _LOG_NORMAL_DUMMY,
}


@pytest.fixture(params=list(SPECS), scope="module")
def models(request):
    """pyhf's model and data, the converted workspace, and pyhf's dummy-term offset."""
    spec = SPECS[request.param]
    pyhf_ws = pyhf.Workspace(spec)
    pyhf_model = pyhf_ws.model(modifier_settings=MODIFIER_SETTINGS)
    offset = DUMMY_CONSTRAINTS.get(request.param, 0.0)
    return pyhf_model, pyhf_ws.data(pyhf_model), from_hifa(spec), offset


def hs3_name(pyhf_name: str) -> str:
    """pyhs3's name for a pyhf parameter name: ``name[i]`` becomes ``name_bin_i``."""
    return re.sub(r"\[(\d+)\]$", r"_bin_\1", pyhf_name)


def by_name(pyhf_model, values) -> dict[str, Any]:
    """pyhf per-parameter values (flattened ``par_order``) keyed by pyhs3 name."""
    names = [hs3_name(name) for name in pyhf_model.config.par_names]
    return dict(zip(names, values, strict=True))


def measurement(ws: Workspace) -> str:
    """Name of the workspace's one measurement."""
    (analysis,) = ws.analyses
    return analysis.name


def random_points(ws: Workspace, seed: int, n: int = 4) -> list[dict[str, float]]:
    """``n`` parameter points drawn inside the middle of each parameter's bounds.

    Fixed parameters stay at their initial value: pyhs3 bakes them into the
    likelihood as constants.
    """
    rng = np.random.default_rng(seed)
    domain = ws.domains[measurement(ws)]
    points = []
    for _ in range(n):
        point = {}
        for p in ws.parameter_points[measurement(ws)]:
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
    nll = ws.model(ws.analyses[measurement(ws)], progress=False).compile_nll(
        gradient=False
    )

    def logpdf(point: dict[str, float]) -> float:
        free = {name: point[name] for name in nll.parameter_names}
        return -nll.value(nll.to_vector(free))

    return logpdf


def pyhf_pars(pyhf_model, point: dict[str, float]) -> list[float]:
    """pyhf's flat parameter vector for a point keyed by pyhs3 name."""
    return [point[hs3_name(name)] for name in pyhf_model.config.par_names]


def pyhf_logpdf(pyhf_model, data, point: dict[str, float]) -> float:
    return float(pyhf_model.logpdf(pyhf_pars(pyhf_model, point), data)[0])


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
    pyhf_model, _, ws, _ = models
    expected = by_name(pyhf_model, pyhf_model.config.suggested_init())
    point = ws.parameter_points[measurement(ws)]
    assert {p.name: p.value for p in point} == pytest.approx(expected, rel=RTOL)


def test_suggested_bounds(models):
    pyhf_model, _, ws, _ = models
    expected = by_name(pyhf_model, pyhf_model.config.suggested_bounds())
    domain = ws.domains[measurement(ws)]
    assert set(domain.axis_names) == set(expected)
    for name, (lo, hi) in expected.items():
        assert domain[name] == pytest.approx((lo, hi), rel=RTOL)


def test_suggested_fixed(models):
    pyhf_model, _, ws, _ = models
    expected = by_name(pyhf_model, pyhf_model.config.suggested_fixed())
    point = ws.parameter_points[measurement(ws)]
    assert {p.name: p.const for p in point} == expected


def test_poi(models):
    pyhf_model, _, ws, _ = models
    analysis = ws.analyses[measurement(ws)]
    assert analysis.parameters_of_interest == [pyhf_model.config.poi_name]


@pytest.mark.parametrize("seed", SEEDS)
def test_expected_rates(models, seed):
    pyhf_model, _, ws, _ = models
    for point in random_points(ws, seed):
        # (total bins,) concatenated over pyhf's channels, sliced per channel.
        expected = pyhf_model.expected_actualdata(pyhf_pars(pyhf_model, point))
        for channel in pyhf_model.config.channels:
            assert_allclose(
                pyhs3_expected(ws, channel, point),
                expected[pyhf_model.config.channel_slices[channel]],
                rtol=RTOL,
            )


def assert_expected_data_matches(pyhf_model, model, point: dict[str, float]) -> None:
    """``Model.expected_data`` against pyhf's expected main-measurement data."""
    # (total bins,) concatenated over pyhf's channels, sliced per channel.
    expected = pyhf_model.expected_data(
        pyhf_pars(pyhf_model, point), include_auxdata=False
    )
    actual = model.expected_data(**point)
    channels = pyhf_model.config.channels
    assert set(actual) == {f"model_{channel}" for channel in channels}
    for channel in channels:
        assert_allclose(
            actual[f"model_{channel}"],
            expected[pyhf_model.config.channel_slices[channel]],
            rtol=1e-12,
        )


def test_model_expected_data_at_init(models):
    pyhf_model, _, ws, _ = models
    model = ws.model(ws.analyses[measurement(ws)], progress=False)
    init = {p.name: p.value for p in ws.parameter_points[measurement(ws)]}
    assert_expected_data_matches(pyhf_model, model, init)


@pytest.mark.parametrize("seed", SEEDS)
def test_model_expected_data_random_points(models, seed):
    pyhf_model, _, ws, _ = models
    model = ws.model(ws.analyses[measurement(ws)], progress=False)
    for point in random_points(ws, seed):
        assert_expected_data_matches(pyhf_model, model, point)


def test_logpdf_at_init(models):
    pyhf_model, data, ws, offset = models
    init = {p.name: p.value for p in ws.parameter_points[measurement(ws)]}
    assert compiled_logpdf(ws)(init) == pytest.approx(
        pyhf_logpdf(pyhf_model, data, init) - offset, rel=RTOL
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_logpdf_random_points(models, seed):
    pyhf_model, data, ws, offset = models
    logpdf = compiled_logpdf(ws)
    for point in random_points(ws, seed):
        assert logpdf(point) == pytest.approx(
            pyhf_logpdf(pyhf_model, data, point) - offset, rel=RTOL
        )


def test_staterror_global_bin_index():
    """pyhf's ``stat_shared[i]`` runs over CR's bins, then SR's; AA takes none."""
    spec = SPECS["staterror_cross_channel"]
    model = pyhf.Workspace(spec).model()
    shared = model.config.par_names[model.config.par_slice("stat_shared")]
    assert shared == [f"stat_shared[{i}]" for i in range(5)]
    ws = from_hifa(spec)
    cr = ws.distributions["model_CR"].samples["bkg"].modifiers[0]
    sr = ws.distributions["model_SR"].samples["bkg"].modifiers[0]
    assert cr.parameters == [hs3_name(name) for name in shared[:3]]
    assert sr.parameters == [hs3_name(name) for name in shared[3:]]


@pytest.fixture
def pyhf_minuit():
    """pyhf's numpy backend with the iminuit optimizer, restored afterwards."""
    pytest.importorskip("iminuit")
    saved = pyhf.get_backend()
    pyhf.set_backend("numpy", pyhf.optimize.minuit_optimizer(tolerance=1e-6))
    yield
    pyhf.set_backend(*saved)


@pytest.mark.usefixtures("pyhf_minuit")
def test_inverse_hessian_matches_pyhf_covariance():
    """The inverse Hessian at pyhf's best fit is the covariance Minuit's HESSE gives."""
    # Observed counts above the background put the best fit of mu inside its
    # bounds, where HESSE's covariance is not distorted by the limit.
    spec = {**SPECS["simplemodel_uncorrelated"]}
    spec["observations"] = [{"name": "singlechannel", "data": [64.0, 61.0]}]
    pyhf_ws = pyhf.Workspace(spec)
    pyhf_model = pyhf_ws.model(modifier_settings=MODIFIER_SETTINGS)
    # Minuit's limits distort HESSE's errors near them, so mu gets wide ones.
    bounds = pyhf_model.config.suggested_bounds()
    bounds[pyhf_model.config.poi_index] = (-10.0, 10.0)
    # (P, 2): best-fit value and uncertainty per parameter; (P, P) correlations.
    fit, correlations = pyhf.infer.mle.fit(
        pyhf_ws.data(pyhf_model),
        pyhf_model,
        par_bounds=bounds,
        return_uncertainties=True,
        return_correlations=True,
    )
    pyhf_order = [hs3_name(name) for name in pyhf_model.config.par_names]
    ws = from_hifa(spec)
    nll = ws.model(ws.analyses[measurement(ws)], progress=False).compile_nll()
    best_fit = dict(zip(pyhf_order, fit[:, 0], strict=True))

    covariance = np.linalg.inv(nll.hessian(nll.to_vector(best_fit)))

    # Reorder pyhf's results to nll.parameter_names.
    order = [pyhf_order.index(name) for name in nll.parameter_names]
    errors = np.sqrt(np.diag(covariance))
    assert_allclose(errors, fit[order, 1], rtol=1e-3)
    assert_allclose(
        covariance / np.outer(errors, errors),
        correlations[np.ix_(order, order)],
        atol=1e-3,
    )
