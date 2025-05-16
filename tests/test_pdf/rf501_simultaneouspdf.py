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
