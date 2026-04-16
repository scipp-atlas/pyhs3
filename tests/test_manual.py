from __future__ import annotations

from inspect import Parameter
import json
import pickle
from pathlib import Path
# from symbol import parameters

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pyhs3 import parameter_points
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3 as hs3
from pyhs3.parameter_points import ParameterSet

test_data = """
    {
    "nll": [
        2115.2146170568185,
        2116.5050528141624,
        2116.32024442325,
        2116.147536871291,
        2115.9898748809364,
        2115.8484135401295,
        2115.7234904744287,
        2115.614760462136,
        2115.5217651455596,
        2115.4428493502382,
        2115.3774140758096,
        2115.3244289405097,
        2115.282917807648,
        2115.2519751699165,
        2115.2307707411514,
        2115.2185475973124,
        2115.214617056965,
        2115.218352131681,
        2115.229180543526,
        2115.2465778591986,
        2115.270060985844,
        2115.299182173344,
        2115.333523579179,
        2115.372692531467,
        2115.4163176659454,
        2115.4640463133146,
        2115.5155437672224,
        2115.5704953916675,
        2115.6286127532935,
        2115.689167112463,
        2115.7526215376183,
        2115.8184195879385
    ],
    "qmu": [
        0.0,
        2.580871514687715,
        2.211254732863381,
        1.865839628944741,
        1.550515648235887,
        1.2675929666220327,
        1.0177468352203505,
        0.8002868106350434,
        0.6142961774821742,
        0.45646458683950186,
        0.32559403798222775,
        0.21962376738247258,
        0.13660150165924279,
        0.07471622619596019,
        0.03230736866589723,
        0.007861080987822788,
        2.928572939708829e-10,
        0.00747014972512261,
        0.029126973415259272,
        0.0639216047602531,
        0.1108878580507735,
        0.1691302330509643,
        0.23781304472140619,
        0.3161509492965706,
        0.4034012182537481,
        0.4988585129922285,
        0.6018534208078563,
        0.7117566696979338,
        0.8279913929500253,
        0.9491001112892263,
        1.0760089615996549,
        1.2076050622399634
    ],
    "status": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "mu_HH": [
        0.9999909338901939,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5
    ],
    "mu": [
        0.9999909338901939,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5
    ]
    } """


def ws_json():
    """Load issue41_diHiggs_workspace.json file and return parsed JSON content.

    This workspace is from Alex Wang for the diHiggs gamgam bb analysis,
    related to GitHub issue #41.
    """
    fpath = Path.home() / ".local" / "skhepdata" / "WS-bbyy-non-resonant-non-param-isofix-unbinnedFix.json"
    return json.loads(fpath.read_text(encoding="utf-8"))


def nz_weighted_entries(entries, weights):
    weighted_array = []
    non_zero_terms = []

    for i in range(len(entries)):
        weighted_array.append(entries[i][0] * weights[i])

    for entry in weighted_array:
        if abs(entry) > 1e-6:
            non_zero_terms.append(entry)

    return sorted(non_zero_terms)


def build_histogram(points, bins, range, figname=None):
    if figname:
        fig = plt.figure(num=figname)
    hist, bin_edges = np.histogram(points, bins=bins, range=range)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_centers, hist, width=width, align="center")


def plot_histogram_from_bins(heights, num_bins, data_range, figname=None):
    if figname:
        fig = plt.figure(num=figname)
    x_min, x_max = data_range
    if len(heights) != num_bins:
        raise ValueError("Length of heights must match num_bins.")

    bin_edges = np.linspace(x_min, x_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (x_max - x_min) / num_bins

    plt.bar(bin_centers, heights, width=bin_width, align="center")

def plot_dist(
    model, 
    parameters, 
    dist_name, 
    data_set, 
    factor=1, 
    plot_name=None,
    label=None,
    linewidth=2.5,
    color="red"
    ):
    xs = [val[0] for val in data_set.entries]
    ys = [
        model.pdf_unsafe(dist_name, **{**parameters, data_set.axes[0].name: [x]}) * factor
        for x in xs
    ]

    sorted_pairs = sorted(zip(xs, ys), key=lambda p: p[0])
    xs, ys = zip(*sorted_pairs) if sorted_pairs else ([], [])

    plt.figure(plot_name)
    plt.title(plot_name)
    if label != None:
        plt.plot(xs, ys, color=color, linewidth=linewidth, label=label)
    else:
        plt.plot(xs, ys, color=color, linewidth=linewidth)
    plt.ylim(0,18)

def get_all_psets(workspace):
    psets = []
    for i in range(len(workspace.parameter_points)):
        if i == 0:
            continue
        psets.append(
            ParameterSet(
                name=f"pset {i}",
                parameters=[
                    *workspace.parameter_points[0].parameters
                    *workspace.parameter_points[i].parameters
                ]
            )
        )
    psets.append(
        ParameterSet(
            name=f'main_set',
            parameters=[
                *workspace.parameter_points[0].parameters
                *workspace.parameter_points["unconditionalGlabs_muhat"].parameters
                *workspace.parameter_points["unconditionalNuis_muhat"].parameters
                *workspace.parameters_points["POI_muhat"].parameters
            ]
        )
    )

    return psets

def main():
    ws = hs3.Workspace(**ws_json())
    test_mus = json.loads(test_data)["mu_HH"]
    cached_file = "ws.pkl"

    # if Path(cached_file).exists():
    #     print("loading model...")
    #     with Path(cached_file).open("rb") as f:
    #         model = pickle.load(f)

    # else:
    merged_pset = ParameterSet(
        name="merged",
        parameters=[
            *ws.parameter_points[0].parameters,
            *ws.parameter_points["unconditionalGlobs_muhat"].parameters,
            *ws.parameter_points["unconditionalNuis_muhat"].parameters,
            *ws.parameter_points["POI_muhat"].parameters
        ],
    )
    print("building model")
    model = ws.model(parameter_set=merged_pset)

    with Path(cached_file).open("wb") as f:
        pickle.dump(model, f)

    asys = ws.analyses["CombinedPdf_combData"]
    like = asys.likelihood

    nlls = []
    parameters = {par.name: par.value for par in model.parameterset}

    unbinned = [d for d in ws.data.root if getattr(d, "type", None) == "unbinned"]

    unbinned_filtered = [data for data in unbinned if "binned" not in data.name]

    nll_given_mu = []
    nll_given_mu_without_constraints = []

    for dataset in unbinned_filtered:
        key = dataset.axes[0].name
        value = parameters[key]
        parameters[key] = [value] if np.ndim(value) == 0 else value 

    run2hm1 = ws.distributions["_model_Run2HM_1"]
    sb_run2hm1 = ws.distributions[run2hm1.factors[0]]

    # reproduce likelihood plot for run2hm1

    with PdfPages("sb_run2hm1_likelihood.pdf") as pdf:
        combdatarun2hm1 = unbinned_filtered[0]
        plot_dist(model, parameters, sb_run2hm1.name, combdatarun2hm1, plot_name=f"distribution: {sb_run2hm1.name}", factor=56*2.5, label="Total Background")

        # overlay histogram of combdata_run2hm1 bin heights
        h = combdatarun2hm1.to_hist(nbins=22)
        bin_edges = h.axes[0].edges
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        heights = h.values()
        mask = heights > 0.001
        plt.scatter(bin_centers[mask], heights[mask], color="black", label="combdata_run2hm1")
        plt.legend()

        pdf.savefig()
        plt.close()

    # calculate nll values

    i = 0
    for mu in test_mus:
        i += 1
        s = f"({i}/{len(test_mus)})"
        print(f"computing nll given mu = {mu }{s:>{20}}")
        parameters["mu_HH"] = mu
        nlls = []
        unconstrained_nlls = []
        for j, (topdist, data_set) in enumerate(
            zip(like.distributions, unbinned_filtered)
        ):
            # print(f"building dist {dist_name} {i}/{len(like.distributions)}")
            unconstrained_dist = ws.distributions[topdist.factors[0]]
            dist = topdist

            nll = 0
            unll = 0

            print(f"datset = {data_set.name}")
            for val in nz_weighted_entries(data_set.entries, data_set.weights):
                temp = {**parameters, data_set.axes[0].name: [val]}
                unconstrained_contribution = (
                    -2
                    * model.logpdf_unsafe(unconstrained_dist.name, **temp)
                    / len(nz_weighted_entries(data_set.entries, data_set.weights))
                )
                contribution = (
                    -2
                    * model.logpdf_unsafe(dist.name, **temp)
                    / len(nz_weighted_entries(data_set.entries, data_set.weights))
                )
                nll += contribution
                unll += unconstrained_contribution
                # print(f"value {val} results in nll sum == {nll}, contribution == {contribution}, dataset = {data_set.name}, distname = {dist_name}")

            # pdf__ggFHH_kl1p0_mc23a_Run3LM_4 

            nlls.append(nll)
            unconstrained_nlls.append(unll)
        # 4/3: make plot of nll minimization overlayed with alex wangs values, both with min set to zero for scale

        nll_given_mu.append(np.sum(nlls))
        nll_given_mu_without_constraints.append(np.sum(unconstrained_nlls))

    plt.figure()
    plt.scatter(test_mus, nll_given_mu_without_constraints)
    plt.xlabel("mu_HH")
    plt.ylabel("nll")
    plt.savefig("nll_curve_without_constraints.pdf")
    plt.figure()
    plt.scatter(test_mus, nll_given_mu)
    plt.xlabel("mu_HH")
    plt.ylabel("nll")
    plt.savefig("nll_curve.pdf")

    # make plot of comparison graph:
    provided_nll = json.loads(test_data)["nll"]
    provided_nll_shifted = [v - min(provided_nll) for v in provided_nll]
    computed_nll_shifted = [v - min(nll_given_mu_without_constraints) for v in nll_given_mu_without_constraints]
    plt.figure()
    plt.scatter(test_mus, provided_nll_shifted, label="provided nll", marker="o")
    plt.scatter(test_mus, computed_nll_shifted, label="computed nll (unconstrained)", marker="x")
    plt.xlabel("mu_HH")
    plt.ylabel("nll")
    plt.title("NLL Comparison")
    plt.legend()
    plt.savefig("nll_comparison.pdf")

    # plot seperate regions of nll curves
    # plot total nll curve over all parameter sets

    breakpoint()

if __name__ == "__main__":
    main()
