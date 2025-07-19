from __future__ import annotations

import json
import math

import numpy as np
import pytensor.tensor as pt
from pytensor import function

from pyhs3 import Workspace
from pyhs3.core import create_bounded_tensor


def test_nondefault_points_domains_access(datadir):
    workspace = Workspace(
        json.loads(datadir.joinpath("nondefault_points_domains.json").read_text())
    )

    assert workspace.parameter_collection[0].name == "default_values"
    assert workspace.parameter_collection[1].name == "nondefault_values"
    assert workspace.parameter_collection["default_values"].name == "default_values"
    assert (
        workspace.parameter_collection["nondefault_values"].name == "nondefault_values"
    )

    assert workspace.domain_collection[0].name == "default_domain"
    assert workspace.domain_collection[1].name == "nondefault_domain"
    assert workspace.domain_collection["default_domain"].name == "default_domain"
    assert workspace.domain_collection["nondefault_domain"].name == "nondefault_domain"


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

    f = create_bounded_tensor("f", scalarranges["f"])
    f_ctl = create_bounded_tensor("f_ctl", scalarranges["f_ctl"])
    mean = create_bounded_tensor("mean", scalarranges["mean"])
    mean2 = create_bounded_tensor("mean2", scalarranges["mean2"])
    sigma = create_bounded_tensor("sigma", scalarranges["sigma"])
    sigma2 = create_bounded_tensor("sigma2", scalarranges["sigma2"])
    mean_ctl = create_bounded_tensor("mean_ctl", scalarranges["mean_ctl"])
    mean2_ctl = create_bounded_tensor("mean2_ctl", scalarranges["mean2_ctl"])
    x = create_bounded_tensor("x", scalarranges["x"])

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


def test_combine_long_exercise_logpdf_evaluation(datadir):
    """Test logPDF evaluation for pdf_binsignal_region from combine long exercise."""
    workspace = Workspace(
        json.loads(
            datadir.joinpath("combine_long_exercise_part1_nosys.json").read_text()
        )
    )

    model = workspace.model(
        parameter_point=workspace.parameter_collection["default_values"],
        domain=workspace.domain_collection["default_domain"],
    )

    # Get default parameter values
    default_params = {par.name: par.value for par in model.parameterset}

    # Test logPDF evaluation at default parameter values
    default_logpdf_val = model.logpdf("pdf_binsignal_region", **default_params)

    # Verify logPDF is finite
    assert np.isfinite(default_logpdf_val)

    # Test logPDF evaluation with different r values (parameter of interest)
    params_r05 = default_params.copy()
    params_r05["r"] = 0.5  # Signal strength = 0.5
    logpdf_val_r05 = model.logpdf("pdf_binsignal_region", **params_r05)

    params_r20 = default_params.copy()
    params_r20["r"] = 2.0  # Signal strength = 2.0
    logpdf_val_r20 = model.logpdf("pdf_binsignal_region", **params_r20)

    # Verify logPDFs are finite
    assert np.isfinite(logpdf_val_r05)
    assert np.isfinite(logpdf_val_r20)

    # LogPDFs should be different for different r values
    assert not np.allclose(logpdf_val_r05, default_logpdf_val)
    assert not np.allclose(logpdf_val_r20, default_logpdf_val)
    assert not np.allclose(logpdf_val_r05, logpdf_val_r20)

    # Test that the distribution follows expected Poisson behavior
    # With observed count = 10 and varying expected counts through r parameter
    # Expected count = sum of backgrounds + r * signal = (4.43803 + 3.18309 + 3.7804 + 1.63396) + r * 0.711064
    # Background total ≈ 13.0355, signal contribution = 0.711064

    # At r=1, expected ≈ 13.747
    # At r=0.5, expected ≈ 13.391
    # At r=2, expected ≈ 14.458

    # Since observed = 10 and all expected values > 10, smaller r values should give higher logPDF
    assert logpdf_val_r05 > default_logpdf_val  # r=0.5 closer to optimum than r=1.0
    assert default_logpdf_val > logpdf_val_r20  # r=1.0 closer to optimum than r=2.0
