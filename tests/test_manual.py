from __future__ import annotations

import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
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
    fpath = Path(skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json"))
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
    plt.bar(bin_centers, hist, width=width, align='center')
    return 

def plot_histogram_from_bins(heights, num_bins, data_range, figname=None):
    if figname:
        fig = plt.figure(num=figname)
    x_min, x_max = data_range
    if len(heights) != num_bins:
        raise ValueError("Length of heights must match num_bins.")

    bin_edges = np.linspace(x_min, x_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (x_max - x_min) / num_bins

    plt.bar(bin_centers, heights, width=bin_width, align='center')
    return

# def plot_crystallball_doublesided(x, distribution_name, ws):
#     dist = ws.distributions[distribution_name]

#     t = (x - m0) / sigma_L
#     if t < -alpha_L:
#         A = (n_L / abs(alpha_L))**n_L * np.exp(-0.5 * alpha_L**2)
#         B = n_L/abs(alpha_L) - abs(alpha_L)
#         return A * (B - t)**(-n_L)
#     # Right side
#     t = (x - m0) / sigma_R
#     if t > alpha_R:
#         A = (n_R / abs(alpha_R))**n_R * np.exp(-0.5 * alpha_R**2)
#         B = n_R/abs(alpha_R) - abs(alpha_R)
#         return A * (B + t)**(-n_R)
#     # Core
#     return np.exp(-0.5 * ((x - m0) / sigma_L)**2)

def base_name(name):
    return (
        name.replace("AsimovData_", "")
        .replace("combData", "")
        .replace("binned_", "")
        .strip("_")
    )

def plot_dist(model, parameters, dist_name, data_set, plot_name=None):
    xs = [val[0] for val in data_set.entries]
    ys = [model.pdf_unsafe(dist_name, **{**parameters, data_set.axes[0].name : x}) for x in xs]

    plt.figure(plot_name)
    plt.title(plot_name)
    plt.scatter(xs, ys)
    return


def main():
    ws = hs3.Workspace(**ws_json())

    # test_mus = json.loads(test_data)["mu_HH"]
    test_mus = np.linspace(-10,-5,20)

    cached_file = "ws.pkl"

    if Path(cached_file).exists():
        print("loading model...")
        with Path(cached_file).open("rb") as f:
            model = pickle.load(f)

    else:
        merged_pset = ParameterSet(name="merged", parameters=[*ws.parameter_points[0].parameters, *ws.parameter_points['unconditionalGlobs_muhat'].parameters])
        print("building model")
        model = ws.model(parameter_set=merged_pset)

    with Path(cached_file).open("wb") as f:
        pickle.dump(model, f)

    asys = ws.analyses["CombinedPdf_combData"]
    like = ws.likelihoods[asys.likelihood]

    nlls = []
    parameters = {par.name: par.value for par in model.parameterset}

    with Path.open("nll_output_test.json", "w") as f:
        nlls.append(sum(nlls[i] for i in range(len(nlls))))
        json.dump(nlls, f, indent=2)
        print("NLLs output saved to nll_output_test.json")

    data = ws.data["combDatabinned_Run2HM_2"]
    nz = nz_weighted_entries(data.entries, data.weights)
    occurrences = np.zeros(220)
    ms = np.linspace(100, 200, 220)
    for i in range(1, 220):
        for m_obs in nz:
            if (m_obs > ms[i - 1]) and (m_obs < ms[i + 1]):
                occurrences[i] += 1

    print(nz)
    print(occurrences)

    binned = [d for d in ws.data.root if getattr(d, "type", None) == "binned"]
    unbinned = [d for d in ws.data.root if getattr(d, "type", None) == "unbinned"]

    unbinned_filtered = [data for data in unbinned if "binned" not in data.name]

    nll_given_mu = []
    mus = [-1000,-100,-10,-1,0,1,10,100,1000]

    # _modelSB_Run2HM_3
    # dist type: "crystalball_doublesided_dist"
    # _modelSB_Run2HM_1
    # _modelSB_Run2HM_2
    # _modelSB_Run2HM_3
    # _modelSB_Run2LM_1
    # _modelSB_Run2LM_2
    # _modelSB_Run2LM_3
    # _modelSB_Run2LM_4
    # _modelSB_Run3HM_1
    # _modelSB_Run3HM_2
    # _modelSB_Run3HM_3
    # _modelSB_Run3LM_1
    # _modelSB_Run3LM_2
    # _modelSB_Run3LM_3
    # _modelSB_Run3LM_4

    # for each plot plot pdfs and coefficients on same plot

    # 1/9:
    # investigate: yield__ggFH_mc20ade_Run2HM_1 and the pdf version which the yield is multiplied by
    ggFH_dist = ws.distributions['_modelSB_Run2HM_1']
    with PdfPages("ggFH_plot.pdf") as pdf:
        plot_dist(model, parameters, ggFH_dist.name, unbinned_filtered[0])
        pdf.savefig()
        plt.close()

    ggFH_yield = ws.functions['yield__VBFH_mc20ade_Run2HM_1']
    # ggFH_yield = ws.functions['yield__background_Run2HM_1']
    for factor in ggFH_yield.factors:
        if factor in parameters:
            print(f'factor: {factor}, value: {parameters[factor]}')
        else:
            print(f'function: {factor}, with factors:\n\t{ws.functions[factor].factors}')

    # 1/16:
    # modelSB is a term within model, continue checking contraint terms within model to find why the overall normalization is so small
    run2hm1 = ws.distributions['_model_Run2HM_1']
    # for con in run2hm1.factors[1:]:
    #     if con in ws.distributions:
    #         con = ws.distributions[con]
    #         if con.type == 'gaussian_dist':
    #             parameters[con.mean] = 1.0
    #             print(f'constraint = {con.name}, gauss mean = {parameters[con.mean]}')

    # breakpoint()
    

    # with PdfPages("log_distribution_plots.pdf") as pdf:
    #     for dist_name, data_set in zip(like.distributions, unbinned_filtered):
    #         plot_dist(model, parameters, dist_name, data_set, plot_name=f"distribution: {dist_name}")
    #         pdf.savefig()
    #         plt.close()
    i = 0
    for mu in test_mus:
        i += 1
        s = f"({i}/{len(test_mus)})"
        print(f"computing nll given mu = {mu}{s:>{20}}")
        parameters["mu_HH"] = mu
        nlls = []
        for i, (dist_name, data_set) in enumerate(zip(like.distributions, unbinned_filtered)):
            # print(f"building dist {dist_name} {i}/{len(like.distributions)}")
            dist = ws.distributions[ws.distributions[dist_name].factors[0]]

            nll = 0
            print(f"datset = {data_set.name}")
            for val in nz_weighted_entries(data_set.entries, data_set.weights):
                temp = {**parameters, data_set.axes[0].name: val}
                contribution = -2 * model.logpdf_unsafe(dist.name, **temp) / len(nz_weighted_entries(data_set.entries, data_set.weights))
                nll += contribution
                # print(f"value {val} results in nll sum == {nll}, contribution == {contribution}, dataset = {data_set.name}, distname = {dist_name}")

            # pdf__ggFHH_kl1p0_mc23a_Run3LM_4
                        

            nlls.append(nll)
        
        nll_given_mu.append(np.sum(nlls))

    breakpoint()
    plt.figure()
    plt.plot(test_mus, nll_given_mu)
    plt.xlabel("mu_HH")
    plt.ylabel("nll")
    plt.savefig("pure_negative_range.pdf")
    breakpoint()

    for b in binned:
        axis = b.axes[0]
        plot_histogram_from_bins(
            b.contents, 
            axis.nbins, 
            (axis.min, axis.max), 
            figname=b.name
            )
    with PdfPages("binned_histograms.pdf") as pdf:
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = fig.get_label()
            if name:
                fig.suptitle(name, fontsize=12, y=0.98)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)  
    
    for u in unbinned:
        # if "binned" in u.name:
        build_histogram(
            nz_weighted_entries(u.entries, u.weights),
            bins=220,
            range=(100, 200),
            figname=u.name
        )
    with PdfPages("unbinned_histograms.pdf") as pdf:
        for num in plt.get_fignums():
            fig = plt.figure(num)
            name = fig.get_label()
            if name:
                fig.suptitle(name, fontsize=12, y=0.98)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            
    print(f"{'Binned Data':<45} | {'Unbinned Data'}")
    print("-" * 80)
    while len(binned) > len(unbinned):
        unbinned.append("")
    for b, u in zip(binned, unbinned):
        print(f"{b.name:<45} | {u}")
    
    out_pdf = "all_histograms.pdf"




if __name__ == "__main__":
    main()
