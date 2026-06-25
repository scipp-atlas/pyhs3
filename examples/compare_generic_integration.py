#!/usr/bin/env python3
# ruff: noqa: T201
"""Compare how pyhs3 and RooFit normalize generic_dist / RooGenericPdf.

Both engines must divide a generic PDF by its integral over the observable
range to normalize it.  pyhs3 uses a fixed 64-point Gauss-Legendre quadrature
(pyhs3.normalization.gauss_legendre_integral); RooFit's RooGenericPdf has no
analytic integral, so RooAbsPdf.createIntegral falls back to its adaptive
numeric integrator (RooIntegrator1D).

This script evaluates the *same* normalization integral, on the same expression
families and parameter values used by the toy workspaces, three ways:

  * truth  : the exact closed-form integral over [lo, hi]
  * pyhs3  : pyhs3's 64-point Gauss-Legendre value (run under pixi/pyhs3)
  * roofit : RooGenericPdf.createIntegral value     (run under the ROOT env)

The engine whose value is closer to ``truth`` is the more accurate one.

Because ROOT and pyhs3 live in different Python environments, run the two
engines separately (each writes a self-contained JSON), then compare:

    # in the pyhs3 / pixi environment
    pixi run python examples/compare_generic_integration.py \
        --engine pyhs3 --output pyhs3_integ.json

    # in the ROOT / quickFit environment
    python3 examples/compare_generic_integration.py \
        --engine roofit --output roofit_integ.json

    # anywhere (pure stdlib)
    python3 examples/compare_generic_integration.py \
        --compare pyhs3_integ.json roofit_integ.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ─── Test cases: families mirror the workspace generic PDFs ──────────────────
# Each case carries both representations of the SAME function so the two
# engines integrate identical math:
#   pyhs3_expr : named-variable string for pyhs3's parser
#   root_expr  : @-indexed string for RooGenericPdf, args ordered as `args`
#   args       : ("x", <param names...>) — x is always the observable, first
#   params     : parameter values
#   truth      : exact closed-form integral over [lo, hi]
LO, HI = 10.0, 20.0


def _truth_exp(c: float, lo: float, hi: float) -> float:
    # ∫ exp(c x) dx = (e^{c hi} - e^{c lo}) / c
    return (math.exp(c * hi) - math.exp(c * lo)) / c


def _truth_poly1(c: float, lo: float, hi: float) -> float:
    # ∫ (1 + c x) dx = (hi - lo) + c (hi^2 - lo^2)/2
    return (hi - lo) + c * (hi * hi - lo * lo) / 2.0


def _truth_gauss(mean: float, sigma: float, lo: float, hi: float) -> float:
    # ∫ exp(-1/2 ((x-m)/s)^2) dx = s sqrt(pi/2) [erf((hi-m)/(s√2)) - erf((lo-m)/(s√2))]
    r2 = math.sqrt(2.0)
    return (
        sigma
        * math.sqrt(math.pi / 2.0)
        * (math.erf((hi - mean) / (sigma * r2)) - math.erf((lo - mean) / (sigma * r2)))
    )


def build_cases(lo: float = LO, hi: float = HI) -> list[dict]:
    cases: list[dict] = []

    # Exponential background (generic exp form), tau values from the channels
    for tau in (-0.30, -0.25, -0.35):
        cases.append(
            {
                "name": f"exp(c*x) c={tau:+.2f}",
                "family": "exp",
                "pyhs3_expr": "exp(c*x)",
                "root_expr": "exp(@1*@0)",
                "args": ["x", "c"],
                "params": {"c": tau},
                "truth": _truth_exp(tau, lo, hi),
            }
        )

    # Linear background (--bkg-form poly), slope around the fit init/range
    for slope in (-0.04, -0.02, 0.0, 0.02):
        cases.append(
            {
                "name": f"1+c*x c={slope:+.2f}",
                "family": "poly1",
                "pyhs3_expr": "1 + c*x",
                "root_expr": "1 + @1*@0",
                "args": ["x", "c"],
                "params": {"c": slope},
                "truth": _truth_poly1(slope, lo, hi),
            }
        )

    # Gaussian signal (--generic-sig), mean=15, channel sigmas
    for sigma in (0.90, 1.00, 1.10):
        cases.append(
            {
                "name": f"gauss m=15 s={sigma:.2f}",
                "family": "gauss",
                "pyhs3_expr": "exp(-0.5*((x-mean)/sigma)**2)",
                "root_expr": "exp(-0.5*((@0-@1)/@2)**2)",
                "args": ["x", "mean", "sigma"],
                "params": {"mean": 15.0, "sigma": sigma},
                "truth": _truth_gauss(15.0, sigma, lo, hi),
            }
        )

    return cases


# ─── pyhs3 engine: the real 64-point Gauss-Legendre code path ────────────────
def run_pyhs3(cases: list[dict], lo: float, hi: float) -> None:
    import pytensor  # noqa: PLC0415
    import pytensor.tensor as pt  # noqa: PLC0415

    from pyhs3.generic_parse import parse_expression, sympy_to_pytensor  # noqa: PLC0415
    from pyhs3.normalization import gauss_legendre_integral  # noqa: PLC0415

    for case in cases:
        sym = parse_expression(case["pyhs3_expr"])
        x = pt.dvector("x")
        param_names = case["args"][1:]
        pt_params = [pt.dscalar(n) for n in param_names]
        variables = [x, *pt_params]

        raw = sympy_to_pytensor(sym, variables)
        integ = gauss_legendre_integral(raw, x, lo, hi)
        fn = pytensor.function(pt_params, integ, on_unused_input="ignore")
        value = float(fn(*[case["params"][n] for n in param_names]))
        case["value"] = value


# ─── RooFit engine: RooGenericPdf.createIntegral ─────────────────────────────
def run_roofit(cases: list[dict], lo: float, hi: float) -> None:
    import ROOT  # noqa: PLC0415

    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    for case in cases:
        xv = ROOT.RooRealVar("x", "x", lo, hi)  # range defines the integration domain
        param_names = case["args"][1:]
        rvars = {
            n: ROOT.RooRealVar(n, n, float(case["params"][n])) for n in param_names
        }
        arglist = ROOT.RooArgList(xv)
        for n in param_names:
            arglist.add(rvars[n])
        pdf = ROOT.RooGenericPdf("gpdf", case["root_expr"], arglist)
        # createIntegral over x integrates the raw formula => normalization denom.
        integ = pdf.createIntegral(ROOT.RooArgSet(xv))
        case["value"] = float(integ.getVal())


# ─── reporting ───────────────────────────────────────────────────────────────
def _rel_err(value: float, truth: float) -> float:
    return abs(value - truth) / abs(truth) if truth != 0.0 else float("nan")


def print_single(engine: str, cases: list[dict]) -> None:
    print(f"\nEngine: {engine}    (integral over [{LO}, {HI}])\n")
    header = f"{'case':<22}  {'truth':>18}  {'value':>18}  {'rel err':>12}"
    print(header)
    print("-" * len(header))
    for c in cases:
        print(
            f"{c['name']:<22}  {c['truth']:>18.12g}  {c['value']:>18.12g}  "
            f"{_rel_err(c['value'], c['truth']):>12.3e}"
        )


def compare(file_a: Path, file_b: Path) -> None:
    da = json.loads(file_a.read_text())
    db = json.loads(file_b.read_text())
    eng_a, eng_b = da["engine"], db["engine"]
    ca = {c["name"]: c for c in da["cases"]}
    cb = {c["name"]: c for c in db["cases"]}

    print(f"\nComparing  {eng_a}  vs  {eng_b}   (integral over closed form truth)\n")
    header = (
        f"{'case':<22}  {'truth':>16}  "
        f"{eng_a + ' relerr':>16}  {eng_b + ' relerr':>16}  {'winner':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, a in ca.items():
        if name not in cb:
            continue
        b = cb[name]
        truth = a["truth"]
        ra = _rel_err(a["value"], truth)
        rb = _rel_err(b["value"], truth)
        winner = eng_a if ra < rb else eng_b if rb < ra else "tie"
        print(f"{name:<22}  {truth:>16.10g}  {ra:>16.3e}  {rb:>16.3e}  {winner:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--engine", choices=["pyhs3", "roofit"], default="pyhs3")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--lower", type=float, default=LO)
    parser.add_argument("--upper", type=float, default=HI)
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("FILE_A", "FILE_B"),
        help="Compare two engine JSON outputs against the closed-form truth.",
    )
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    cases = build_cases(args.lower, args.upper)
    if args.engine == "pyhs3":
        run_pyhs3(cases, args.lower, args.upper)
    else:
        run_roofit(cases, args.lower, args.upper)

    print_single(args.engine, cases)

    out = args.output or Path(f"{args.engine}_integ.json")
    out.write_text(
        json.dumps(
            {
                "engine": args.engine,
                "lower": args.lower,
                "upper": args.upper,
                "cases": cases,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
