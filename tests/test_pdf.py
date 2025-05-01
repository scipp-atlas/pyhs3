from __future__ import annotations

import json
import math

import numpy as np
import pytensor.tensor as pt
from pytensor import function

from pyhs3 import Workspace
from pyhs3.core import boundedscalar


def test_rf501_simultaneouspdf(datadir):
    workspace = Workspace(
        json.loads(datadir.joinpath("rf501_simultaneouspdf.json").read_text())
    )

    model = workspace.model(
        parameter_point=workspace.parameter_collection["default_values"],
        domain=workspace.domain_collection["default_domain"],
    )

    physicspdfval = model.pdf(
        "model",
        x=model.parameterset["x"].value,
        f=model.parameterset["f"].value,
        mean=model.parameterset["mean"].value,
        sigma=model.parameterset["sigma"].value,
        mean2=model.parameterset["mean2"].value,
        sigma2=model.parameterset["sigma2"].value,
    )
    physicspdfvalctl = model.pdf(
        "model_ctl",
        x=model.parameterset["x"].value,
        f_ctl=model.parameterset["f_ctl"].value,
        mean_ctl=model.parameterset["mean_ctl"].value,
        mean2_ctl=model.parameterset["mean2_ctl"].value,
        sigma=model.parameterset["sigma"].value,
    )

    assert np.allclose(physicspdfval, 1.3298076)
    assert np.allclose(physicspdfvalctl, 2.56486621e-22)


def test_rf501_manual(datadir):
    workspace = Workspace(
        json.loads(datadir.joinpath("rf501_simultaneouspdf.json").read_text())
    )

    model = workspace.model(
        parameter_point=workspace.parameter_collection["default_values"],
        domain=workspace.domain_collection["default_domain"],
    )

    scalarranges = workspace.domain_collection["default_domain"]

    f = boundedscalar("f", scalarranges["f"])
    f_ctl = boundedscalar("f_ctl", scalarranges["f_ctl"])
    mean = boundedscalar("mean", scalarranges["mean"])
    mean2 = boundedscalar("mean2", scalarranges["mean2"])
    sigma = boundedscalar("sigma", scalarranges["sigma"])
    sigma2 = boundedscalar("sigma2", scalarranges["sigma2"])
    mean_ctl = boundedscalar("mean_ctl", scalarranges["mean_ctl"])
    mean2_ctl = boundedscalar("mean2_ctl", scalarranges["mean2_ctl"])
    x = boundedscalar("x", scalarranges["x"])

    def gaussian_pdf(x, mu, sigma):
        norm_const = 1.0 / (pt.sqrt(2 * math.pi) * sigma)
        exponent = pt.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return norm_const * exponent

    def mixture_pdf(coeff, pdf1, pdf2):
        return coeff * pdf1 + (1.0 - coeff) * pdf2

    gx = gaussian_pdf(x, mean, sigma)
    px = gaussian_pdf(x, mean2, sigma2)
    f_model = mixture_pdf(f, gx, px)

    gx_ctl = gaussian_pdf(x, mean_ctl, sigma)
    px_ctl = gaussian_pdf(x, mean2_ctl, sigma)
    f_model_ctl = mixture_pdf(f_ctl, gx_ctl, px_ctl)

    sample = pt.scalar("sample", dtype="int32")

    simPdf = pt.switch(pt.eq(sample, 0), f_model, f_model_ctl)

    pdf_physics = function(
        [x, f, mean, sigma, mean2, sigma2], f_model, name="pdf_physics"
    )

    pdf_control = function(
        [x, f_ctl, mean_ctl, mean2_ctl, sigma], f_model_ctl, name="pdf_control"
    )

    pdf_combined = function(
        [sample, x, f, mean, sigma, mean2, sigma2, f_ctl, mean_ctl, mean2_ctl],
        simPdf,
        name="pdf_combined",
    )

    val_physics = pdf_physics(
        model.parameterset["x"].value,
        model.parameterset["f"].value,
        model.parameterset["mean"].value,
        model.parameterset["sigma"].value,
        model.parameterset["mean2"].value,
        model.parameterset["sigma2"].value,
    )

    val_control = pdf_control(
        model.parameterset["x"].value,
        model.parameterset["f_ctl"].value,
        model.parameterset["mean_ctl"].value,
        model.parameterset["mean2_ctl"].value,
        model.parameterset["sigma"].value,
    )

    val_combined_physics = pdf_combined(
        0,  # sample
        model.parameterset["x"].value,
        model.parameterset["f"].value,
        model.parameterset["mean"].value,
        model.parameterset["sigma"].value,
        model.parameterset["mean2"].value,
        model.parameterset["sigma2"].value,
        model.parameterset["f_ctl"].value,
        model.parameterset["mean_ctl"].value,
        model.parameterset["mean2_ctl"].value,
    )

    val_combined_control = pdf_combined(
        1,  # sample
        model.parameterset["x"].value,
        model.parameterset["f"].value,
        model.parameterset["mean"].value,
        model.parameterset["sigma"].value,
        model.parameterset["mean2"].value,
        model.parameterset["sigma2"].value,
        model.parameterset["f_ctl"].value,
        model.parameterset["mean_ctl"].value,
        model.parameterset["mean2_ctl"].value,
    )

    assert np.allclose(val_physics, 1.3298076)
    assert np.allclose(val_control, 2.56486621e-22)
    assert np.allclose(val_combined_physics, 1.3298076)
    assert np.allclose(val_combined_control, 2.56486621e-22)
