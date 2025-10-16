from __future__ import annotations

import json

import pytest

import pyhf
import pyhs3


@pytest.mark.xfail(reason="To be implemented")
@pytest.mark.parametrize(
    ("pars"),
    [[i, j, k] for i in [0.0, 1.0] for j in [0.0, 1.0] for k in [0.0, 1.0]],
)
def test_simplemodel_pyhf(pars, datadir):
    """
    To convert the pyhf simplemodel in HiFa JSON to HS3 JSON:

        $ pyhf xml2json <hifa.json> --output-dir hs3
        $ cd hs3
        $ hist2workspace FitConfig.xml
        $ root -b config/FitConfig_combined_measurement_model.root
        root [1] combined
        (RooWorkspace *) 0x1362aac00
        root [2] auto mytool = RooJSONFactoryWSTool(*combined);
        root [3] mytool.exportJSON("<hs3.json>")
        (bool) true
    """
    ws_pyhf = pyhf.Workspace(
        json.loads(
            datadir.joinpath(
                "simplemodel_uncorrelated-background_hifa.json"
            ).read_text()
        )
    )
    model_pyhf = ws_pyhf.model()
    data_pyhf = ws_pyhf.data(model_pyhf)

    ws_pyhs3 = pyhs3.Workspace(
        **json.loads(
            datadir.joinpath("simplemodel_uncorrelated-background_hs3.json").read_text()
        )
    )
    model_pyhs3 = ws_pyhs3.model()
    data_pyhs3 = ws_pyhs3.data()

    assert model_pyhs3.pdf(pars, data_pyhs3) == pytest.approx(
        model_pyhf.pdf(pars, data_pyhf)
    )
    assert model_pyhs3.logpdf(pars, data_pyhs3) == pytest.approx(
        model_pyhf.logpdf(pars, data_pyhf)
    )
