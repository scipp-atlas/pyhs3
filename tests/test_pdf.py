from __future__ import annotations

import math

import pytensor.tensor as pt
from pytensor import function

from pyhs3.allexjsontest import boundedscalar, mymodel, myworkspace

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
