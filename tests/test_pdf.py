from __future__ import annotations

import math

import pytensor.tensor as pt
from pytensor import function

from pyhs3.core import boundedscalar, mymodel, myworkspace

"""
## \file
## \ingroup tutorial_roofit_main
## \notebook
## Organization and simultaneous fits: using simultaneous pdfs to describe simultaneous
## fits to multiple datasets
##
## \macro_image
## \macro_code
## \macro_output
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)
from __future__ import annotations

import ROOT

# Create model for physics sample
# -------------------------------------------------------------

# Create observables
x = ROOT.RooRealVar("x", "x", -8, 8)

# Construct signal pdf
mean = ROOT.RooRealVar("mean", "mean", 0, -8, 8)
sigma = ROOT.RooRealVar("sigma", "sigma", 0.3, 0.1, 10)
gx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)

mean2 = ROOT.RooRealVar("mean2", "mean2", 0, -3, 3)
sigma2 = ROOT.RooRealVar("sigma2", "sigma2", 0.3, 0.1, 10)
px = ROOT.RooGaussian("px", "px", x, mean2, sigma2)

# Construct composite pdf
f = ROOT.RooRealVar("f", "f", 0.2, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "model", [gx, px], [f])

# Create model for control sample
# --------------------------------------------------------------

# Construct signal pdf.
# NOTE that sigma is shared with the signal sample model
mean_ctl = ROOT.RooRealVar("mean_ctl", "mean_ctl", -3, -8, 8)
gx_ctl = ROOT.RooGaussian("gx_ctl", "gx_ctl", x, mean_ctl, sigma)

mean2_ctl = ROOT.RooRealVar("mean2_ctl", "mean2_ctl", -3, -3, 3)
px_ctl = ROOT.RooGaussian("px_ctl", "px_ctl", x, mean2_ctl, sigma)

# Construct the composite model
f_ctl = ROOT.RooRealVar("f_ctl", "f_ctl", 0.5, 0.0, 1.0)
model_ctl = ROOT.RooAddPdf("model_ctl", "model_ctl", [gx_ctl, px_ctl], [f_ctl])

# Generate events for both samples
# ---------------------------------------------------------------

# Generate 1000 events in x and y from model
data = model.generate({x}, 1000)
data_ctl = model_ctl.generate({x}, 2000)

# Create index category and join samples
# ---------------------------------------------------------------------------

# Define category to distinguish physics and control samples events
sample = ROOT.RooCategory("sample", "sample")
sample.defineType("physics")
sample.defineType("control")

# Construct combined dataset in (x,sample)
combData = ROOT.RooDataSet(
    "combData",
    "combined data",
    {x},
    Index=sample,
    Import={"physics": data, "control": data_ctl},
)

# Construct a simultaneous pdf in (x, sample)
# -----------------------------------------------------------------------------------

# Construct a simultaneous pdf using category sample as index: associate model
# with the physics state and model_ctl with the control state
simPdf = ROOT.RooSimultaneous(
    "simPdf", "simultaneous pdf", {"physics": model, "control": model_ctl}, sample
)
w = ROOT.RooWorkspace("w", "w")
w.Import({simPdf})

w.var("x").setVal(0.0)
w.var("f").setVal(0.2)
w.var("mean").setVal(0.0)
w.var("sigma").setVal(0.3)
w.var("mean2").setVal(0.0)
w.var("sigma2").setVal(0.3)
w.var("f_ctl").setVal(0.5)
w.var("mean_ctl").setVal(-3.0)
w.var("mean2_ctl").setVal(-3.0)

obs_set = ROOT.RooArgSet(w.var("x"))  # Define `observable` based on your dataset

for label in ["physics", "control"]:
    sample.setLabel(label)
    # Get the probability density function value
    pdf_value = simPdf.getVal(obs_set)

    print(f"Simultaneous PDF ({label}):", pdf_value)
"""


scalarranges = myworkspace.domain_collection["default_domain"]

f = boundedscalar("f", scalarranges["f"])
f_ctl = boundedscalar("f_ctl", scalarranges["f_ctl"])
mean = boundedscalar("mean", scalarranges["mean"])
mean2 = boundedscalar("mean2", scalarranges["mean2"])
sigma = boundedscalar("sigma", scalarranges["sigma"])
sigma2 = boundedscalar("sigma2", scalarranges["sigma2"])
mean_ctl = boundedscalar("mean_ctl", scalarranges["mean_ctl"])
mean2_ctl = boundedscalar("mean2_ctl", scalarranges["mean2_ctl"])
# breakpoint()
x = boundedscalar("x", scalarranges["x"])


def gaussian_pdf(x, mu, sigma):
    norm_const = 1.0 / (pt.sqrt(2 * math.pi) * sigma)
    exponent = pt.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * exponent


def mixture_pdf(coeff, pdf1, pdf2):
    return coeff * pdf1 + (1.0 - coeff) * pdf2


print("distributions: ", mymodel.distributions)
gx = gaussian_pdf(x, mean, sigma)
px = gaussian_pdf(x, mean2, sigma2)
model = mixture_pdf(f, gx, px)

gx_ctl = gaussian_pdf(x, mean_ctl, sigma)
px_ctl = gaussian_pdf(x, mean2_ctl, sigma)
model_ctl = mixture_pdf(f_ctl, gx_ctl, px_ctl)

sample = pt.scalar("sample", dtype="int32")

simPdf = pt.switch(pt.eq(sample, 0), model, model_ctl)

pdf_physics = function([x, f, mean, sigma, mean2, sigma2], model, name="pdf_physics")

pdf_control = function(
    [x, f_ctl, mean_ctl, mean2_ctl, sigma], model_ctl, name="pdf_control"
)

pdf_combined = function(
    [sample, x, f, mean, sigma, mean2, sigma2, f_ctl, mean_ctl, mean2_ctl],
    simPdf,
    name="pdf_combined",
)

# default_params = {p["name"]: p["value"] for p in mymodel.startingpoints}
# default_params = mymodel.parameterset
# mymodel.parameters['x']
# breakpoint()
val_physics = pdf_physics(
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
)

val_control = pdf_control(
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value,
    mymodel.parameterset["sigma"].value,
)

val_combined_physics = pdf_combined(
    0,  # sample
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value,
)

val_combined_control = pdf_combined(
    1,  # sample
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value,
)

print("Physics PDF:", val_physics)
print("Control PDF:", val_control)
print("Simultaneous PDF:", val_combined_physics)
print("Simultaneous PDF(control):", val_combined_control)

print(mymodel.parameterset)
