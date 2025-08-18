from __future__ import annotations

import json
import pickle
from pathlib import Path

from skhep_testdata import data_path as skhep_testdata_path

import pyhs3 as hs3

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


def main():
    ws = hs3.Workspace(**ws_json())

    cached_file = "ws.pkl"

    if Path.exists(cached_file):
        print("loading model...")
        with Path.open(cached_file, "rb") as f:
            model = pickle.load(f)

    else:
        print("building model")
        model = ws.model()

        with Path.open(cached_file, "wb") as f:
            pickle.dump(model, f)

    asys = ws.analyses["CombinedPdf_combData"]
    like = ws.likelihoods[asys.likelihood]

    nlls = []
    parameters = {par.name: par.value for par in model.parameterset}

    for i, dist_name in enumerate(like.distributions):
        print(f"building dist {dist_name} {i}/{len(like.distributions)}")
        dist = ws.distributions[dist_name]

        nlls.append(-2 * model.logpdf(dist.name, **parameters))

    with Path.open("nll_output_test.json", "w") as f:
        json.dump(nlls, f, indent=2)


if __name__ == "__main__":
    main()
