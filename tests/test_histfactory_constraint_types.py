"""Constraint terms of the per-bin (gamma) HistFactory modifiers, for each HS3 v0.2 type.

HS3 v0.2 lets a modifier's ``constraint`` be ``Gauss``, ``Poisson`` or
``LogNormal``. For the per-bin modifiers (``shapesys``, ``staterror``) with
relative uncertainty sigma_i in bin i, the terms are

- Gauss:     Gauss(1 | gamma_i, sigma_i)
- Poisson:   Pois(tau_i | gamma_i * tau_i), continuous, tau_i = 1 / sigma_i^2
- LogNormal: LogNormal(1 | mu = ln gamma_i, sigma = sigma_i)

``shapesys`` ``vals`` follow the pyhf convention (absolute uncertainties), so
sigma_i = vals_i / nominal_i; ``staterror`` uses sigma_i = errors_i / nominal_i.

The inputs mirror a ROOT 6.40.04 HistFactory model (nominal 50 and 60, relative
uncertainties 0.2 and 0.05, gamma 1.1 and 0.97). The Poisson values are
RooPoisson's, printed by ROOT for that model; the Gauss values agree with
RooGaussian's to the 1e-6 its finite-range normalization leaves.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest
from scipy.stats import lognorm, norm

from pyhs3.context import Context
from pyhs3.distributions import HistFactoryDistChannel
from pyhs3.distributions.histfactory.data import SampleData
from pyhs3.distributions.histfactory.modifiers import (
    ShapeSysData,
    ShapeSysModifier,
    StatErrorData,
    StatErrorModifier,
)

NOMINAL = np.array([50.0, 60.0])
SIGMA_REL = np.array([0.2, 0.05])
GAMMA = np.array([1.1, 0.97])
PARAMETERS = ["gamma_0", "gamma_1"]

#: RooPoisson(nom_gamma = tau | gamma * tau) from ROOT 6.40.04, per bin.
ROOT_POISSON = np.array([0.0707250746593, 0.0165965218293])

EXPECTED = {
    "Gauss": norm.pdf(1.0, loc=GAMMA, scale=SIGMA_REL),
    "Poisson": ROOT_POISSON,
    "LogNormal": lognorm.pdf(1.0, s=SIGMA_REL, scale=GAMMA),
}

CONSTRAINT_TYPES = ["Gauss", "Poisson", "LogNormal"]


@pytest.fixture
def context():
    return Context({p: pt.constant(g) for p, g in zip(PARAMETERS, GAMMA, strict=True)})


def _assert_constraint(make, make_log, constraint_type):
    """The product of per-bin terms, and its log, equal the expected per-bin terms."""
    expected = EXPECTED[constraint_type]
    np.testing.assert_allclose(make.eval(), np.prod(expected), rtol=1e-9)
    np.testing.assert_allclose(make_log.eval(), np.sum(np.log(expected)), rtol=1e-9)


class TestShapeSysConstraintTypes:
    @pytest.mark.parametrize("constraint_type", CONSTRAINT_TYPES)
    def test_constraint_term(self, constraint_type, context):
        modifier = ShapeSysModifier(
            name="shapesys",
            parameters=PARAMETERS,
            # pyhf convention: absolute uncertainties.
            data=ShapeSysData(vals=list(SIGMA_REL * NOMINAL)),
            constraint=constraint_type,
        )
        sample = SampleData(contents=list(NOMINAL), errors=[0.0, 0.0])
        _assert_constraint(
            modifier.make_constraint(context, sample),
            modifier.log_constraint(context, sample),
            constraint_type,
        )

    def test_rejects_types_outside_hs3_v02(self):
        with pytest.raises(ValueError, match="constraint"):
            ShapeSysModifier(
                name="shapesys",
                parameters=PARAMETERS,
                data=ShapeSysData(vals=[0.0, 0.0]),
                constraint="Const",
            )


class TestStatErrorConstraintTypes:
    @pytest.mark.parametrize("constraint_type", CONSTRAINT_TYPES)
    def test_barlow_beeston_full(self, constraint_type, context):
        modifier = StatErrorModifier(
            name="staterror",
            parameters=PARAMETERS,
            data=StatErrorData(uncertainties=list(SIGMA_REL * NOMINAL)),
            constraint=constraint_type,
        )
        sample = SampleData(contents=list(NOMINAL), errors=list(SIGMA_REL * NOMINAL))
        _assert_constraint(
            modifier.make_constraint(context, sample),
            modifier.log_constraint(context, sample),
            constraint_type,
        )

    @pytest.mark.parametrize("constraint_type", CONSTRAINT_TYPES)
    def test_barlow_beeston_lite(self, constraint_type, context):
        channel = HistFactoryDistChannel(
            name="channel",
            axes=[{"name": "x", "min": 0.0, "max": 2.0, "nbins": 2}],
            samples=[
                {
                    "name": "background",
                    "data": {
                        "contents": list(NOMINAL),
                        "errors": list(SIGMA_REL * NOMINAL),
                    },
                    "modifiers": [
                        {
                            "name": "staterror",
                            "type": "staterror",
                            "parameters": PARAMETERS,
                            "constraint": constraint_type,
                        }
                    ],
                }
            ],
            barlow_beeston_method="lite",
        )
        _assert_constraint(
            channel._make_barlow_beeston_lite_constraint(context),
            channel._make_barlow_beeston_lite_log_constraint(context),
            constraint_type,
        )
