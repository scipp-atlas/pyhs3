import json
import math
import pytensor
from pytensor import function as function
import pytensor.tensor as pt

json_content = r"""
{
  "distributions": [
    {
      "coefficients": [
        "f"
      ],
      "extended": false,
      "name": "model",
      "summands": [
        "gx",
        "px"
      ],
      "type": "mixture_dist"
    },
    {
      "mean": "mean",
      "name": "gx",
      "sigma": "sigma",
      "type": "gaussian_dist",
      "x": "x"
    },
    {
      "mean": "mean2",
      "name": "px",
      "sigma": "sigma2",
      "type": "gaussian_dist",
      "x": "x"
    },
    {
      "coefficients": [
        "f_ctl"
      ],
      "extended": false,
      "name": "model_ctl",
      "summands": [
        "gx_ctl",
        "px_ctl"
      ],
      "type": "mixture_dist"
    },
    {
      "mean": "mean_ctl",
      "name": "gx_ctl",
      "sigma": "sigma",
      "type": "gaussian_dist",
      "x": "x"
    },
    {
      "mean": "mean2_ctl",
      "name": "px_ctl",
      "sigma": "sigma",
      "type": "gaussian_dist",
      "x": "x"
    }
  ],
  "domains": [
    {
      "axes": [
        {
          "max": 1.0,
          "min": 0.0,
          "name": "f"
        },
        {
          "max": 1.0,
          "min": 0.0,
          "name": "f_ctl"
        },
        {
          "max": 8.0,
          "min": -8.0,
          "name": "mean"
        },
        {
          "max": 3.0,
          "min": -3.0,
          "name": "mean2"
        },
        {
          "max": 3.0,
          "min": -3.0,
          "name": "mean2_ctl"
        },
        {
          "max": 8.0,
          "min": -8.0,
          "name": "mean_ctl"
        },
        {
          "max": 10.0,
          "min": 0.1,
          "name": "sigma"
        },
        {
          "max": 10.0,
          "min": 0.1,
          "name": "sigma2"
        },
        {
          "max": 8.0,
          "min": -8.0,
          "name": "x"
        }
      ],
      "name": "default_domain",
      "type": "product_domain"
    }
  ],
  "metadata": {
    "hs3_version": "0.2",
    "packages": [
      {
        "name": "ROOT",
        "version": "6.32.06"
      }
    ]
  },
  "misc": {
    "ROOT_internal": {
      "combined_distributions": {
        "simPdf": {
          "distributions": [
            "model_ctl",
            "model"
          ],
          "index_cat": "sample",
          "indices": [
            1,
            0
          ],
          "labels": [
            "control",
            "physics"
          ]
        }
      }
    }
  },
  "parameter_points": [
    {
      "name": "default_values",
      "parameters": [
        {
          "name": "f",
          "value": 0.2
        },
        {
          "name": "x",
          "value": 0.0
        },
        {
          "name": "mean",
          "value": 0.0
        },
        {
          "name": "sigma",
          "value": 0.3
        },
        {
          "name": "mean2",
          "value": 0.0
        },
        {
          "name": "sigma2",
          "value": 0.3
        },
        {
          "name": "f_ctl",
          "value": 0.5
        },
        {
          "name": "mean_ctl",
          "value": -3.0
        },
        {
          "name": "mean2_ctl",
          "value": -3.0
        }
      ]
    }
  ]
}
"""

data = json.loads(json_content)

f = pt.scalar("f")
f_ctl = pt.scalar("f_ctl")
mean = pt.scalar("mean")
mean2 = pt.scalar("mean2")
sigma = pt.scalar("sigma")
sigma2 = pt.scalar("sigma2")
mean_ctl = pt.scalar("mean_ctl")
mean2_ctl = pt.scalar("mean2_ctl")
x = pt.scalar("x")

def gaussian_pdf(x, mu, sigma):
    norm_const = 1.0 / (pt.sqrt(2 * math.pi) * sigma)
    exponent = pt.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * exponent

def mixture_pdf(coeff, pdf1, pdf2):

    return coeff * pdf1 + (1.0 - coeff) * pdf2

gx = gaussian_pdf(x, mean, sigma)
px = gaussian_pdf(x, mean2, sigma2)
model = mixture_pdf(f, gx, px)

gx_ctl = gaussian_pdf(x, mean_ctl, sigma)
px_ctl = gaussian_pdf(x, mean2_ctl, sigma)
model_ctl = mixture_pdf(f_ctl, gx_ctl, px_ctl)

sample = pt.scalar("sample", dtype="int32")

simPdf = pt.switch(
    pt.eq(sample, 0),
    model,
    model_ctl
)

pdf_physics = function(
    [x, f, mean, sigma, mean2, sigma2],
    model,
    name="pdf_physics"
)

pdf_control = function(
    [x, f_ctl, mean_ctl, mean2_ctl, sigma],
    model_ctl,
    name="pdf_control"
)

pdf_combined = function(
    [sample, x,
     f, mean, sigma, mean2, sigma2,
     f_ctl, mean_ctl, mean2_ctl],
    simPdf,
    name="pdf_combined"
)

default_values_block = data["parameter_points"][0]["parameters"]
default_params = {p["name"]: p["value"] for p in default_values_block}

val_physics = pdf_physics(
    default_params["x"],
    default_params["f"],
    default_params["mean"],
    default_params["sigma"],
    default_params["mean2"],
    default_params["sigma2"]
)

val_control = pdf_control(
    default_params["x"],
    default_params["f_ctl"],
    default_params["mean_ctl"],
    default_params["mean2_ctl"],
    default_params["sigma"]
)

val_combined_physics = pdf_combined(
    0,
    default_params["x"],
    default_params["f"],
    default_params["mean"],
    default_params["sigma"],
    default_params["mean2"],
    default_params["sigma2"],
    default_params["f_ctl"],
    default_params["mean_ctl"],
    default_params["mean2_ctl"]
)

val_combined_control = pdf_combined(
    1,
    default_params["x"],
    default_params["f"],
    default_params["mean"],
    default_params["sigma"],
    default_params["mean2"],
    default_params["sigma2"],
    default_params["f_ctl"],
    default_params["mean_ctl"],
    default_params["mean2_ctl"]
)

print("Physics PDF:", val_physics)
print("Control PDF:", val_control)
print("Simultaneous PDF:", val_combined_physics)
print("Simultaneous PDF:", val_combined_control)

