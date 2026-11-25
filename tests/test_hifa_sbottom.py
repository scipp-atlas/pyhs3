"""End-to-end check of from_hifa against pyhf on a published ATLAS likelihood.

The ATLAS sbottom search (SUSY-2018-31, INSPIRE 1748602) published its
likelihoods as pyhf JSON: a background-only workspace per signal region and
one JSON patch per signal point. scikit-hep-testdata carries them as
``pyhf-ins1748602-probability-models.tar.gz``. The signal patches remove the
validation-region channel and leave its observation, which pyhf ignores.

Compiling the likelihood takes minutes, so the test is marked slow.
"""

from __future__ import annotations

import json
import tarfile

import jsonpatch
import numpy as np
import pytest
from skhep_testdata import data_path
from test_hifa_pyhf import hs3_name

from pyhs3 import from_hifa

pyhf = pytest.importorskip("pyhf")

ARCHIVE = "pyhf-ins1748602-probability-models.tar.gz"
REGION = "RegionA"
SIGNAL_POINT = "sbottom_1300_205_60"


@pytest.fixture(scope="module")
def spec():
    """RegionA's background-only workspace with one signal point patched in."""
    with tarfile.open(data_path(ARCHIVE)) as archive:

        def read(name):
            member = archive.extractfile(name)
            assert member is not None
            return json.load(member)

        background_only = read(f"{REGION}/BkgOnly.json")
        patch = read(f"{REGION}/patch.{SIGNAL_POINT}.json")
    return jsonpatch.apply_patch(background_only, patch)


@pytest.fixture(scope="module")
def pyhf_model(spec):
    """pyhf's model and observed data (main plus auxiliary)."""
    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    return model, workspace.data(model)


@pytest.fixture(scope="module")
def pyhs3_logpdf(spec):
    """pyhs3's log-likelihood as a function of every parameter, keyed by name."""
    ws = from_hifa(spec)
    (analysis,) = ws.analyses
    nll = ws.model(analysis, progress=False).compile_nll(gradient=False)

    def logpdf(point):
        free = {name: point[name] for name in nll.parameter_names}
        return -nll.value(nll.to_vector(free))

    return logpdf


def by_name(model, pars):
    """pyhf's flat parameter vector keyed by pyhs3 name."""
    names = [hs3_name(name) for name in model.config.par_names]
    return dict(zip(names, pars, strict=True))


@pytest.mark.slow
class TestSbottomIns1748602:
    """RegionA with the (1300, 205, 60) GeV signal point."""

    def test_logpdf_at_suggested_init(self, pyhf_model, pyhs3_logpdf):
        model, data = pyhf_model
        pars = model.config.suggested_init()
        expected = float(model.logpdf(pars, data)[0])
        assert pyhs3_logpdf(by_name(model, pars)) == pytest.approx(expected, rel=1e-10)

    def test_logpdf_at_pyhf_best_fit(self, pyhf_model, pyhs3_logpdf):
        model, data = pyhf_model
        pars = np.asarray(pyhf.infer.mle.fit(data, model))
        expected = float(model.logpdf(pars, data)[0])
        assert pyhs3_logpdf(by_name(model, pars)) == pytest.approx(expected, rel=1e-10)
