"""Unit tests for the HS3 to pyhf (HiFa JSON) converter, ``pyhs3.to_hifa``.

Each test feeds an HS3 workspace in and checks the pyhf dict that comes out,
or the error, without pyhf itself. Most inputs start from what
``pyhs3.from_hifa`` produces for a small pyhf workspace and edit the HS3 dict
from there. ``tests/test_to_hifa_pyhf.py`` checks the output against pyhf.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from pyhs3 import Workspace, from_hifa, hifa_modifier_settings, to_hifa

MU = {"name": "mu", "type": "normfactor", "data": None}
NORMSYS = {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}}
HISTOSYS = {
    "name": "bkg_shape",
    "type": "histosys",
    "data": {"hi_data": [55.0, 62.0], "lo_data": [45.0, 58.0]},
}
SHAPEFACTOR = {"name": "sf", "type": "shapefactor", "data": None}
SHAPESYS = {"name": "bkg_unc", "type": "shapesys", "data": [5.0, 6.0]}
STATERROR = {"name": "staterror_SR", "type": "staterror", "data": [2.0, 3.0]}
# The same staterror name on a three-bin channel ``AA``.
STATERROR_AA = {**STATERROR, "data": [4.0, 5.0, 6.0]}
LUMI = {"name": "lumi", "type": "lumi", "data": None}
LUMI_CONFIG = {
    "name": "lumi",
    "auxdata": [1.0],
    "sigmas": [0.1],
    "inits": [1.0],
    "bounds": [[0.5, 1.5]],
}


def make_spec(
    modifiers: list[dict[str, Any]] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    signal_modifiers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A one-channel, two-bin pyhf workspace with the given background modifiers."""
    return {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "signal",
                        "data": [5.0, 10.0],
                        "modifiers": [MU, *(signal_modifiers or [])],
                    },
                    {
                        "name": "background",
                        "data": [50.0, 60.0],
                        "modifiers": modifiers or [],
                    },
                ],
            }
        ],
        "observations": [{"name": "SR", "data": [53.0, 65.0]}],
        "measurements": [
            {"name": "meas", "config": {"poi": "mu", "parameters": parameters or []}}
        ],
        "version": "1.0.0",
    }


def add_channel(
    spec: dict[str, Any], name: str, nbins: int, *modifiers: dict[str, Any]
) -> None:
    """Append a channel with one background sample of ``nbins`` bins of 100."""
    spec["channels"].append(
        {
            "name": name,
            "samples": [
                {
                    "name": "background",
                    "data": [100.0] * nbins,
                    "modifiers": list(modifiers),
                }
            ],
        }
    )
    spec["observations"].append({"name": name, "data": [100.0] * nbins})


def hs3_of(spec: dict[str, Any]) -> dict[str, Any]:
    """The HS3 dict ``from_hifa`` produces for ``spec``."""
    return from_hifa(spec).model_dump(mode="json", exclude_none=True)


def convert(hs3: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return to_hifa(Workspace(**copy.deepcopy(hs3)), **kwargs)


def dist(hs3: dict[str, Any], name: str = "model_SR") -> dict[str, Any]:
    """The distribution ``name`` of an HS3 dict, for editing in place."""
    return next(d for d in hs3["distributions"] if d["name"] == name)


def sample(hs3: dict[str, Any], name: str, channel: str = "model_SR") -> dict[str, Any]:
    return next(s for s in dist(hs3, channel)["samples"] if s["name"] == name)


def point(hs3: dict[str, Any], name: str) -> dict[str, Any]:
    """The entry for parameter ``name`` in the first parameter point."""
    return next(
        p for p in hs3["parameter_points"][0]["parameters"] if p["name"] == name
    )


def axis(hs3: dict[str, Any], name: str) -> dict[str, Any]:
    """The axis of parameter ``name`` in the first domain."""
    return next(a for a in hs3["domains"][0]["axes"] if a["name"] == name)


def config(result: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return result["measurements"][index]["config"]


class TestBasicMapping:
    def test_normfactor_and_normsys(self):
        assert convert(hs3_of(make_spec([NORMSYS]))) == {
            "channels": [
                {
                    "name": "SR",
                    "samples": [
                        {
                            "name": "signal",
                            "data": [5.0, 10.0],
                            "modifiers": [MU],
                        },
                        {
                            "name": "background",
                            "data": [50.0, 60.0],
                            "modifiers": [NORMSYS],
                        },
                    ],
                }
            ],
            "observations": [{"name": "SR", "data": [53.0, 65.0]}],
            "measurements": [
                {"name": "meas", "config": {"poi": "mu", "parameters": []}}
            ],
            "version": "1.0.0",
        }

    def test_histosys(self):
        result = convert(hs3_of(make_spec([HISTOSYS])))
        assert result["channels"][0]["samples"][1]["modifiers"] == [HISTOSYS]

    def test_modifier_name_is_the_parameter(self):
        hs3 = hs3_of(make_spec([NORMSYS]))
        modifier = sample(hs3, "background")["modifiers"][0]
        modifier["name"] = "renamed"
        modifier["parameter"] = "alpha_bkg"
        point(hs3, "bkg_norm")["name"] = "alpha_bkg"
        axis(hs3, "bkg_norm")["name"] = "alpha_bkg"
        result = convert(hs3)
        assert result["channels"][0]["samples"][1]["modifiers"] == [
            {**NORMSYS, "name": "alpha_bkg"}
        ]

    def test_channel_name_without_model_prefix_is_kept(self):
        hs3 = hs3_of(make_spec())
        dist(hs3)["name"] = "SR_dist"
        hs3["likelihoods"][0]["distributions"] = ["SR_dist"]
        assert convert(hs3)["channels"][0]["name"] == "SR_dist"
        assert convert(hs3)["observations"][0]["name"] == "SR_dist"

    def test_channels_and_observations_follow_the_likelihood(self):
        spec = make_spec()
        add_channel(spec, "CR", 3)
        hs3 = hs3_of(spec)
        hs3["likelihoods"][0]["distributions"].reverse()
        hs3["likelihoods"][0]["data"].reverse()
        result = convert(hs3)
        assert [c["name"] for c in result["channels"]] == ["CR", "SR"]
        assert result["observations"] == [
            {"name": "CR", "data": [100.0] * 3},
            {"name": "SR", "data": [53.0, 65.0]},
        ]

    def test_two_channels_with_one_name_raise(self):
        spec = make_spec()
        add_channel(spec, "CR", 2)
        hs3 = hs3_of(spec)
        dist(hs3, "model_CR")["name"] = "SR"
        hs3["likelihoods"][0]["distributions"][1] = "SR"
        with pytest.raises(ValueError, match=r"channel name 'SR'"):
            convert(hs3)

    def test_multidimensional_axes_are_flattened(self):
        hs3 = hs3_of(make_spec())
        hs3["distributions"][0]["axes"] = [
            {"name": "x", "min": 0.0, "max": 1.0, "nbins": 1},
            {"name": "y", "min": 0.0, "max": 2.0, "nbins": 2},
        ]
        hs3["data"][0]["axes"] = copy.deepcopy(hs3["distributions"][0]["axes"])
        result = convert(hs3)
        assert result["channels"][0]["samples"][1]["data"] == [50.0, 60.0]
        assert result["observations"][0]["data"] == [53.0, 65.0]


class TestPerBinModifiers:
    def test_shapefactor(self):
        result = convert(hs3_of(make_spec([SHAPEFACTOR])))
        assert result["channels"][0]["samples"][1]["modifiers"] == [SHAPEFACTOR]

    def test_poisson_shapesys(self):
        result = convert(hs3_of(make_spec([SHAPESYS])))
        assert result["channels"][0]["samples"][1]["modifiers"] == [SHAPESYS]

    def test_gauss_shapesys_is_a_single_sample_staterror(self):
        hs3 = hs3_of(make_spec([SHAPESYS]))
        sample(hs3, "background")["modifiers"][0]["constraint"] = "Gauss"
        result = convert(hs3)
        assert result["channels"][0]["samples"][1]["modifiers"] == [
            {"name": "bkg_unc", "type": "staterror", "data": [5.0, 6.0]}
        ]

    def test_lognormal_shapesys_raises(self):
        hs3 = hs3_of(make_spec([SHAPESYS]))
        sample(hs3, "background")["modifiers"][0]["constraint"] = "LogNormal"
        with pytest.raises(ValueError, match=r"LogNormal shapesys 'bkg_unc'"):
            convert(hs3)

    def test_lite_staterror_takes_the_sample_errors(self):
        spec = make_spec([STATERROR], signal_modifiers=[STATERROR])
        result = convert(hs3_of(spec))
        samples = result["channels"][0]["samples"]
        assert samples[0]["modifiers"] == [MU, STATERROR]
        assert samples[1]["modifiers"] == [STATERROR]

    def test_extra_staterror_group_round_trips(self):
        extra = {"name": "stat_extra", "type": "staterror", "data": [1.0, 0.5]}
        spec = make_spec([STATERROR, extra], signal_modifiers=[STATERROR])
        result = convert(hs3_of(spec))
        assert result["channels"][0]["samples"][1]["modifiers"] == [STATERROR, extra]

    def test_staterror_across_channels_uses_the_global_index(self):
        spec = make_spec([STATERROR])
        add_channel(spec, "AA", 3, STATERROR_AA)
        result = convert(hs3_of(spec))
        modifiers = [c["samples"][-1]["modifiers"] for c in result["channels"]]
        assert modifiers == [[STATERROR], [STATERROR_AA]]

    def test_staterror_off_the_global_index_raises(self):
        spec = make_spec([STATERROR])
        add_channel(spec, "AA", 3, STATERROR_AA)
        hs3 = hs3_of(spec)
        # pyhf numbers staterror_SR over AA (bins 0-2), then SR (bins 3-4).
        modifier = sample(hs3, "background")["modifiers"][0]
        modifier["parameters"] = ["staterror_SR_bin_5", "staterror_SR_bin_6"]
        with pytest.raises(
            ValueError, match=r"starts at bin 5, but pyhf numbers its bins from 3"
        ):
            convert(hs3)

    def test_shapefactor_not_from_bin_zero_raises(self):
        hs3 = hs3_of(make_spec([SHAPEFACTOR]))
        sample(hs3, "background")["modifiers"][0]["parameters"] = [
            "sf_bin_1",
            "sf_bin_2",
        ]
        with pytest.raises(ValueError, match=r"pyhf numbers its bins from 0"):
            convert(hs3)

    @pytest.mark.parametrize(
        "parameters",
        [["sf_bin_0", "other_bin_1"], ["sf_0", "sf_1"], ["sf_bin_0", "sf_bin_2"]],
    )
    def test_parameters_not_one_bin_group_raise(self, parameters):
        hs3 = hs3_of(make_spec([SHAPEFACTOR]))
        sample(hs3, "background")["modifiers"][0]["parameters"] = parameters
        with pytest.raises(ValueError, match=r"not one group <name>_bin_<i>"):
            convert(hs3)

    def test_partly_overlapping_parameter_lists_raise(self):
        spec = make_spec([SHAPEFACTOR])
        add_channel(spec, "CR", 3)
        hs3 = hs3_of(spec)
        # pyhs3 accepts what pyhf does not: sf_bin_0..1 on SR, sf_bin_0..2 on CR.
        cr = sample(hs3, "background", "model_CR")
        cr["modifiers"] = [
            {
                "name": "sf",
                "type": "shapefactor",
                "parameters": ["sf_bin_0", "sf_bin_1", "sf_bin_2"],
            }
        ]
        with pytest.raises(ValueError, match=r"partly overlap"):
            convert(hs3)

    def test_one_name_on_other_bins_in_the_same_channel_raises(self):
        hs3 = hs3_of(make_spec([SHAPEFACTOR], signal_modifiers=[SHAPEFACTOR]))
        sample(hs3, "signal")["modifiers"][1]["parameters"] = ["sf_bin_2", "sf_bin_3"]
        with pytest.raises(ValueError, match=r"the same bins"):
            convert(hs3)

    def test_parameter_in_a_bin_group_and_a_scalar_raises(self):
        hs3 = hs3_of(make_spec([SHAPEFACTOR]))
        sample(hs3, "signal")["modifiers"][0]["parameter"] = "sf_bin_0"
        with pytest.raises(ValueError, match=r"'sf_bin_0'"):
            convert(hs3, poi="mu")

    def test_shapesys_group_on_two_samples_raises(self):
        hs3 = hs3_of(make_spec([SHAPESYS]))
        sample(hs3, "signal")["modifiers"].append(
            copy.deepcopy(sample(hs3, "background")["modifiers"][0])
        )
        with pytest.raises(ValueError, match=r"shapesys 'bkg_unc' .* more than one"):
            convert(hs3)

    def test_name_shared_by_types_with_different_defaults_raises(self):
        hs3 = hs3_of(make_spec([NORMSYS]))
        sample(hs3, "signal")["modifiers"].append(
            {"name": "n", "type": "normfactor", "parameter": "bkg_norm"}
        )
        with pytest.raises(ValueError, match=r"'bkg_norm' .* normfactor and normsys"):
            convert(hs3)

    def test_lite_staterrors_with_different_parameters_raise(self):
        spec = make_spec([STATERROR], signal_modifiers=[STATERROR])
        hs3 = hs3_of(spec)
        modifier = sample(hs3, "signal")["modifiers"][1]
        modifier["parameters"] = ["other_bin_0", "other_bin_1"]
        with pytest.raises(ValueError, match=r"Barlow-Beeston lite"):
            convert(hs3)

    def test_non_gauss_staterror_raises(self):
        hs3 = hs3_of(make_spec([STATERROR]))
        sample(hs3, "background")["modifiers"][0]["constraint"] = "Poisson"
        with pytest.raises(ValueError, match=r"Poisson staterror"):
            convert(hs3)

    def test_barlow_beeston_full_raises(self):
        hs3 = hs3_of(make_spec([STATERROR]))
        dist(hs3)["barlow_beeston_method"] = "full"
        modifier = sample(hs3, "background")["modifiers"][0]
        modifier["data"] = {"uncertainties": [2.0, 3.0]}
        with pytest.raises(ValueError, match=r"Barlow-Beeston full"):
            convert(hs3)

    def test_gauss_shapesys_beside_a_lite_group_of_its_name_raises(self):
        hs3 = hs3_of(make_spec([STATERROR], signal_modifiers=[SHAPESYS]))
        signal = sample(hs3, "signal")["modifiers"][1]
        signal["constraint"] = "Gauss"
        signal["parameters"] = ["staterror_SR_bin_0", "staterror_SR_bin_1"]
        with pytest.raises(ValueError, match=r"staterror 'staterror_SR' shares"):
            convert(hs3)


class TestConstraints:
    @pytest.mark.parametrize("constraint", ["Poisson", "LogNormal"])
    @pytest.mark.parametrize("modifier", [NORMSYS, HISTOSYS])
    def test_non_gauss_normsys_and_histosys_raise(self, modifier, constraint):
        hs3 = hs3_of(make_spec([modifier]))
        sample(hs3, "background")["modifiers"][0]["constraint"] = constraint
        with pytest.raises(
            ValueError, match=rf"{constraint} {modifier['type']} '{modifier['name']}'"
        ):
            convert(hs3)


class TestLumi:
    def spec(self) -> dict[str, Any]:
        return make_spec([LUMI], parameters=[LUMI_CONFIG], signal_modifiers=[LUMI])

    def test_lumi(self):
        result = convert(hs3_of(self.spec()))
        samples = result["channels"][0]["samples"]
        assert samples[0]["modifiers"] == [MU, LUMI]
        assert samples[1]["modifiers"] == [LUMI]
        assert config(result)["parameters"] == [LUMI_CONFIG]

    def test_fixed_lumi(self):
        spec = self.spec()
        spec["measurements"][0]["config"]["parameters"] = [
            {**LUMI_CONFIG, "inits": [1.05], "fixed": True}
        ]
        assert config(convert(hs3_of(spec)))["parameters"] == [
            {**LUMI_CONFIG, "inits": [1.05], "fixed": True}
        ]

    def test_lumi_init_defaults_to_its_auxdata(self):
        hs3 = hs3_of(self.spec())
        hs3["parameter_points"][0]["parameters"].remove(point(hs3, "lumi"))
        assert config(convert(hs3))["parameters"] == [LUMI_CONFIG]

    def test_lumi_without_bounds_raises(self):
        hs3 = hs3_of(self.spec())
        hs3["domains"][0]["axes"].remove(axis(hs3, "lumi"))
        with pytest.raises(ValueError, match=r"needs finite bounds for 'lumi'"):
            convert(hs3)

    def test_lumi_constraint_on_another_name_raises(self):
        hs3 = hs3_of(self.spec())
        for s in dist(hs3)["samples"]:
            for m in s["modifiers"]:
                if m.get("parameter") == "lumi":
                    m["parameter"] = "Lumi"
        hs3["distributions"][-1]["mean"] = "Lumi"
        point(hs3, "lumi")["name"] = "Lumi"
        axis(hs3, "lumi")["name"] = "Lumi"
        with pytest.raises(ValueError, match=r"named 'lumi'.*'Lumi'"):
            convert(hs3)

    def test_lumi_parameter_on_a_normsys_raises(self):
        hs3 = hs3_of(self.spec())
        sample(hs3, "background")["modifiers"].append(
            {
                "name": "x",
                "type": "normsys",
                "parameter": "lumi",
                "data": {"hi": 1.1, "lo": 0.9},
            }
        )
        with pytest.raises(ValueError, match=r"'lumi' .* lumi and normsys"):
            convert(hs3)

    def test_aux_distribution_not_a_lumi_constraint_raises(self):
        hs3 = hs3_of(self.spec())
        hs3["distributions"][-1]["x"] = "lumi_nominal"
        hs3["parameter_points"][0]["parameters"].append(
            {"name": "lumi_nominal", "value": 1.0, "const": True}
        )
        with pytest.raises(
            ValueError, match=r"aux distribution 'lumi_constraint_meas'"
        ):
            convert(hs3)

    def test_two_aux_distributions_raise(self):
        hs3 = hs3_of(self.spec())
        hs3["distributions"].append({**hs3["distributions"][-1], "name": "other"})
        hs3["likelihoods"][0]["aux_distributions"].append("other")
        with pytest.raises(
            ValueError, match=r"aux distributions \['lumi_constraint_meas', 'other'\]"
        ):
            convert(hs3)

    def test_likelihoods_differing_in_the_lumi_constraint_raise(self):
        hs3 = hs3_of(self.spec())
        hs3["likelihoods"].append(
            {**hs3["likelihoods"][0], "name": "no_lumi", "aux_distributions": []}
        )
        with pytest.raises(ValueError, match=r"differ in whether they constrain lumi"):
            convert(hs3)

    def test_lumi_constraint_without_a_lumi_normfactor_raises(self):
        hs3 = hs3_of(self.spec())
        for s in dist(hs3)["samples"]:
            s["modifiers"] = [m for m in s["modifiers"] if m["parameter"] != "lumi"]
        with pytest.raises(ValueError, match=r"no normfactor uses it"):
            convert(hs3)


class TestUnsupportedContent:
    def test_non_histfactory_distribution_raises(self):
        hs3 = hs3_of(make_spec())
        hs3["distributions"].append(
            {
                "name": "g",
                "type": "gaussian_dist",
                "x": "obs",
                "mean": "mu",
                "sigma": 1.0,
            }
        )
        hs3["data"].append(
            {
                "name": "d",
                "type": "unbinned",
                "axes": [{"name": "obs", "min": -5.0, "max": 5.0}],
                "entries": [[0.1]],
            }
        )
        hs3["likelihoods"][0]["distributions"].append("g")
        hs3["likelihoods"][0]["data"].append("d")
        with pytest.raises(ValueError, match=r"'g' is a gaussian_dist"):
            convert(hs3)

    def test_unbinned_data_raises(self):
        hs3 = hs3_of(make_spec())
        hs3["data"][0] = {
            "name": "obsData_SR",
            "type": "unbinned",
            "axes": [{"name": "obs_x_SR", "min": 0.0, "max": 2.0}],
            "entries": [[0.5]],
        }
        with pytest.raises(ValueError, match=r"'obsData_SR' is unbinned"):
            convert(hs3)

    def test_data_with_another_bin_count_raises(self):
        hs3 = hs3_of(make_spec())
        hs3["data"][0]["axes"][0].update(max=3.0, nbins=3)
        hs3["data"][0]["contents"] = [53.0, 65.0, 1.0]
        with pytest.raises(ValueError, match=r"'obsData_SR' has 3 bins"):
            convert(hs3)

    def test_parameter_that_is_a_function_raises(self):
        hs3 = hs3_of(make_spec())
        hs3["functions"] = [
            {"name": "mu", "type": "product", "factors": ["mu_a", "mu_b"]}
        ]
        with pytest.raises(ValueError, match=r"'mu' is a function"):
            convert(hs3, poi="mu_a")


class TestSelection:
    def two_likelihoods(self) -> dict[str, Any]:
        hs3 = hs3_of(make_spec())
        hs3["data"].append(
            {**hs3["data"][0], "name": "asimov_SR", "contents": [55.0, 70.0]}
        )
        hs3["likelihoods"].append(
            {**hs3["likelihoods"][0], "name": "asimov", "data": ["asimov_SR"]}
        )
        hs3["analyses"].append(
            {**hs3["analyses"][0], "name": "asimov_fit", "likelihood": "asimov"}
        )
        return hs3

    def test_ambiguous_likelihoods_raise(self):
        with pytest.raises(ValueError, match=r"choose one with likelihood="):
            convert(self.two_likelihoods())

    @pytest.mark.parametrize("likelihood", ["asimov", 1])
    def test_likelihood_by_name_or_index(self, likelihood):
        result = convert(self.two_likelihoods(), likelihood=likelihood)
        assert result["observations"] == [{"name": "SR", "data": [55.0, 70.0]}]
        assert [m["name"] for m in result["measurements"]] == ["asimov_fit"]

    @pytest.mark.parametrize("likelihood", ["nope", 2])
    def test_unknown_likelihood_raises(self, likelihood):
        with pytest.raises(ValueError, match=r"no likelihood"):
            convert(self.two_likelihoods(), likelihood=likelihood)

    def test_likelihoods_with_the_same_data_give_one_measurement_each(self):
        spec = make_spec()
        spec["measurements"].append(
            {
                "name": "other",
                "config": {
                    "poi": "mu",
                    "parameters": [{"name": "mu", "bounds": [[0.0, 5.0]]}],
                },
            }
        )
        result = convert(hs3_of(spec))
        assert result["measurements"] == [
            {"name": "meas", "config": {"poi": "mu", "parameters": []}},
            {
                "name": "other",
                "config": {
                    "poi": "mu",
                    "parameters": [{"name": "mu", "bounds": [[0.0, 5.0]]}],
                },
            },
        ]

    def test_workspace_without_likelihood_raises(self):
        hs3 = hs3_of(make_spec())
        hs3["likelihoods"] = []
        hs3["analyses"] = []
        with pytest.raises(ValueError, match=r"no likelihood to convert"):
            convert(hs3)

    def test_likelihood_without_analysis_raises(self):
        hs3 = hs3_of(make_spec())
        hs3["analyses"] = []
        with pytest.raises(ValueError, match=r"no analysis"):
            convert(hs3)

    def test_several_pois_raise(self):
        hs3 = hs3_of(make_spec([NORMSYS]))
        hs3["analyses"][0]["parameters_of_interest"] = ["mu", "bkg_norm"]
        with pytest.raises(ValueError, match=r"choose one with poi="):
            convert(hs3)

    def test_poi_chooses_among_several(self):
        hs3 = hs3_of(make_spec([NORMSYS]))
        hs3["analyses"][0]["parameters_of_interest"] = ["mu", "bkg_norm"]
        assert config(convert(hs3, poi="bkg_norm"))["poi"] == "bkg_norm"

    def test_poi_not_listed_raises(self):
        with pytest.raises(ValueError, match=r"'nope' is not"):
            convert(hs3_of(make_spec()), poi="nope")

    def test_poi_given_when_none_listed(self):
        hs3 = hs3_of(make_spec())
        del hs3["analyses"][0]["parameters_of_interest"]
        assert config(convert(hs3, poi="mu"))["poi"] == "mu"
        with pytest.raises(ValueError, match=r"no parameters of interest"):
            convert(hs3)


class TestMeasurementConfig:
    def test_values_differing_from_pyhf_defaults_are_written(self):
        parameters = [
            {"name": "bkg_norm", "inits": [0.5], "fixed": True},
            {"name": "mu", "bounds": [[-1.0, 5.0]]},
            {"name": "sf", "inits": [1.2, 1.0], "bounds": [[0.0, 10.0], [0.0, 2.0]]},
        ]
        spec = make_spec([NORMSYS, SHAPEFACTOR], parameters=parameters)
        assert config(convert(hs3_of(spec)))["parameters"] == parameters

    def test_per_bin_fixed(self):
        parameters = [{"name": "sf", "fixed": True}]
        spec = make_spec([SHAPEFACTOR], parameters=parameters)
        assert config(convert(hs3_of(spec)))["parameters"] == parameters

    def test_per_bin_partly_fixed_raises(self):
        hs3 = hs3_of(make_spec([SHAPEFACTOR]))
        point(hs3, "sf_bin_0")["const"] = True
        with pytest.raises(ValueError, match=r"'sf', but only some"):
            convert(hs3)

    def test_bins_pyhf_fixes_itself_are_not_fixed(self):
        zero = {"name": "bkg_unc", "type": "shapesys", "data": [5.0, 0.0]}
        hs3 = hs3_of(make_spec([zero]))
        assert point(hs3, "bkg_unc_bin_1")["const"] is True
        assert config(convert(hs3))["parameters"] == []

    def test_missing_values_take_pyhf_defaults(self):
        hs3 = hs3_of(make_spec([NORMSYS]))
        hs3["parameter_points"][0]["parameters"] = []
        hs3["domains"][0]["axes"] = [axis(hs3, "mu")]
        assert config(convert(hs3))["parameters"] == []

    def test_analysis_without_init_takes_pyhf_defaults(self):
        hs3 = hs3_of(make_spec([NORMSYS], parameters=[{"name": "mu", "inits": [2.0]}]))
        del hs3["analyses"][0]["init"]
        assert config(convert(hs3))["parameters"] == []

    def test_bounds_intersect_over_the_analysis_domains(self):
        hs3 = hs3_of(make_spec())
        hs3["domains"].append(
            {
                "name": "narrow",
                "type": "product_domain",
                "axes": [{"name": "mu", "min": -1.0, "max": 3.0}],
            }
        )
        hs3["analyses"][0]["domains"].append("narrow")
        assert config(convert(hs3))["parameters"] == [
            {"name": "mu", "bounds": [[0.0, 3.0]]}
        ]

    def test_open_bound_raises(self):
        hs3 = hs3_of(make_spec())
        del axis(hs3, "mu")["max"]
        with pytest.raises(ValueError, match=r"finite bounds for 'mu'"):
            convert(hs3)

    def test_open_bound_of_a_parameter_outside_the_model_is_ignored(self):
        hs3 = hs3_of(make_spec())
        hs3["domains"][0]["axes"].append({"name": "unused", "min": 0.0})
        assert config(convert(hs3))["parameters"] == []


class TestModifierSettings:
    def test_defaults(self):
        ws = Workspace(**hs3_of(make_spec()))
        assert hifa_modifier_settings(ws) == {
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        }

    def test_codes_from_the_workspace(self):
        settings = {
            "normsys": {"interpcode": "code1"},
            "histosys": {"interpcode": "code0"},
        }
        ws = from_hifa(make_spec([NORMSYS, HISTOSYS]), modifier_settings=settings)
        assert hifa_modifier_settings(ws) == settings

    def test_mixed_codes_raise(self):
        spec = make_spec([NORMSYS])
        add_channel(spec, "CR", 2, NORMSYS)
        hs3 = hs3_of(spec)
        sample(hs3, "background", "model_CR")["modifiers"][0]["data"][
            "interpolation"
        ] = "code1"
        with pytest.raises(ValueError, match=r"normsys .* \['code1', 'code4'\]"):
            hifa_modifier_settings(Workspace(**copy.deepcopy(hs3)))
        with pytest.raises(ValueError, match=r"normsys .* \['code1', 'code4'\]"):
            convert(hs3)

    def test_likelihood_restricts_the_channels(self):
        spec = make_spec([NORMSYS])
        add_channel(spec, "CR", 2, NORMSYS)
        hs3 = hs3_of(spec)
        sample(hs3, "background", "model_CR")["modifiers"][0]["data"][
            "interpolation"
        ] = "code1"
        hs3["likelihoods"].append(
            {"name": "SR_only", "distributions": ["model_SR"], "data": ["obsData_SR"]}
        )
        ws = Workspace(**hs3)
        assert hifa_modifier_settings(ws, likelihood="SR_only")["normsys"] == {
            "interpcode": "code4"
        }

    def test_histosys_code2_warns(self, caplog):
        settings = {"histosys": {"interpcode": "code2"}}
        ws = from_hifa(make_spec([HISTOSYS]), modifier_settings=settings)
        with caplog.at_level(logging.WARNING, logger="pyhs3.hifa"):
            assert hifa_modifier_settings(ws)["histosys"] == {"interpcode": "code2"}
        assert "code2" in caplog.text


class TestDroppedContent:
    def dropped(self, hs3: dict[str, Any], caplog: pytest.LogCaptureFixture) -> str:
        with caplog.at_level(logging.DEBUG, logger="pyhs3.hifa"):
            convert(hs3)
        return caplog.text

    def test_binning(self, caplog):
        assert "binning of channel 'SR'" in self.dropped(hs3_of(make_spec()), caplog)

    def test_metadata_and_misc(self, caplog):
        hs3 = hs3_of(make_spec())
        hs3["misc"] = {"note": "x"}
        text = self.dropped(hs3, caplog)
        assert "metadata" in text
        assert "misc" in text

    def test_errors_without_staterror(self, caplog):
        hs3 = hs3_of(make_spec())
        sample(hs3, "background")["data"]["errors"] = [1.0, 2.0]
        assert "errors of sample 'background'" in self.dropped(hs3, caplog)

    def test_other_parameter_points(self, caplog):
        hs3 = hs3_of(make_spec())
        hs3["parameter_points"].append({"name": "extra", "parameters": []})
        assert "parameter points ['extra']" in self.dropped(hs3, caplog)
