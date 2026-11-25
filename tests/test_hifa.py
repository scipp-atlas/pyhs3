"""Unit tests for the pyhf (HiFa JSON) to HS3 converter, ``pyhs3.from_hifa``.

Each test feeds a pyhf workspace dict in and checks the HS3 structure that
comes out, without pyhf itself. ``tests/test_hifa_pyhf.py`` compares the
converted models numerically against pyhf.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest
from pydantic import ValidationError

import pyhs3
from pyhs3 import from_hifa


def make_spec(
    modifiers: list[dict[str, Any]] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    poi: str = "mu",
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
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
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
            {"name": "meas", "config": {"poi": poi, "parameters": parameters or []}}
        ],
        "version": "1.0.0",
    }


NORMSYS = {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}}
HISTOSYS = {
    "name": "bkg_shape",
    "type": "histosys",
    "data": {"hi_data": [55.0, 62.0], "lo_data": [45.0, 58.0]},
}


SHAPEFACTOR = {"name": "sf", "type": "shapefactor", "data": None}
SHAPESYS = {"name": "bkg_unc", "type": "shapesys", "data": [5.0, 6.0]}
STATERROR = {"name": "staterror_SR", "type": "staterror", "data": [2.0, 3.0]}
LUMI = {"name": "lumi", "type": "lumi", "data": None}
LUMI_CONFIG = {
    "name": "lumi",
    "auxdata": [1.0],
    "sigmas": [0.1],
    "inits": [1.0],
    "bounds": [[0.5, 1.5]],
}


def add_signal_modifiers(spec: dict[str, Any], *modifiers: dict[str, Any]) -> None:
    """Append modifiers to the signal sample of channel ``SR``."""
    spec["channels"][0]["samples"][0]["modifiers"].extend(modifiers)


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


def modifiers_of(
    ws: pyhs3.Workspace, sample: str, channel: str = "SR"
) -> list[dict[str, Any]]:
    """Serialized modifiers of one sample of one channel."""
    dist = ws.distributions[f"model_{channel}"]
    return [m.model_dump() for m in dist.samples[sample].modifiers]


def points_of(ws: pyhs3.Workspace, name: str = "meas") -> dict[str, dict[str, Any]]:
    """Parameter point ``name`` as {parameter: {value, const}}."""
    return {
        p.name: {"value": p.value, "const": p.const}
        for p in ws.parameter_points[name].parameters
    }


def bounds_of(
    ws: pyhs3.Workspace, name: str = "meas"
) -> dict[str, tuple[float, float]]:
    """Domain ``name`` as {parameter: (min, max)}."""
    return {axis.name: (axis.min, axis.max) for axis in ws.domains[name].axes}


class TestStructure:
    """Names and layout of the generated HS3 objects."""

    def test_returns_workspace(self):
        assert isinstance(from_hifa(make_spec()), pyhs3.Workspace)

    def test_metadata_version(self):
        assert from_hifa(make_spec()).metadata.hs3_version == "0.2"

    def test_channel_becomes_histfactory_dist(self):
        ws = from_hifa(make_spec())
        dist = ws.distributions["model_SR"]
        assert dist.type == "histfactory_dist"
        assert [s.name for s in dist.samples] == ["signal", "background"]
        assert dist.samples["background"].data.contents == [50.0, 60.0]
        axis = dist.axes[0]
        assert (axis.name, axis.min, axis.max, axis.nbins) == ("obs_x_SR", 0, 2, 2)

    def test_observation_becomes_binned_data(self):
        ws = from_hifa(make_spec())
        datum = ws.data["obsData_SR"]
        assert datum.type == "binned"
        assert datum.contents == [53.0, 65.0]
        axis = datum.axes[0]
        assert (axis.name, axis.min, axis.max, axis.nbins) == ("obs_x_SR", 0, 2, 2)

    def test_measurement_names_likelihood_analysis_domain_point(self):
        ws = from_hifa(make_spec())
        likelihood = ws.likelihoods["meas"]
        assert likelihood.distribution_names() == ["model_SR"]
        assert [d.name for d in likelihood.data] == ["obsData_SR"]
        analysis = ws.analyses["meas"]
        assert analysis.likelihood.name == "meas"
        assert analysis.parameters_of_interest == ["mu"]
        assert [d.name for d in analysis.domains] == ["meas"]
        assert analysis.init == "meas"
        assert "meas" in ws.domains
        assert "meas" in ws.parameter_points

    def test_one_set_per_measurement(self):
        spec = make_spec()
        spec["measurements"].append(
            {"name": "other", "config": {"poi": "mu", "parameters": []}}
        )
        ws = from_hifa(spec)
        for name in ("meas", "other"):
            assert name in ws.likelihoods
            assert name in ws.analyses
            assert name in ws.domains
            assert name in ws.parameter_points

    def test_two_channels(self):
        spec = make_spec()
        spec["channels"].append(
            {
                "name": "CR",
                "samples": [
                    {
                        "name": "background",
                        "data": [100.0, 110.0, 120.0],
                        "modifiers": [NORMSYS],
                    }
                ],
            }
        )
        spec["observations"].append({"name": "CR", "data": [101.0, 108.0, 125.0]})
        ws = from_hifa(spec)
        assert ws.likelihoods["meas"].distribution_names() == ["model_SR", "model_CR"]
        assert ws.distributions["model_CR"].axes[0].nbins == 3
        assert ws.data["obsData_CR"].axes[0].max == 3

    def test_input_not_mutated(self):
        spec = make_spec([NORMSYS, HISTOSYS])
        original = copy.deepcopy(spec)
        from_hifa(spec)
        assert spec == original

    def test_model_builds(self):
        ws = from_hifa(make_spec([NORMSYS, HISTOSYS]))
        model = ws.model(ws.analyses["meas"], progress=False)
        assert set(model.free_params) == {"mu", "bkg_norm", "bkg_shape"}


class TestNormFactor:
    """``normfactor``: unconstrained, init 1, bounds [0, 10]."""

    def test_modifier(self):
        ws = from_hifa(make_spec())
        assert modifiers_of(ws, "signal") == [
            {"name": "mu", "type": "normfactor", "parameter": "mu", "constraint": None}
        ]

    def test_defaults(self):
        ws = from_hifa(make_spec())
        assert points_of(ws)["mu"] == {"value": 1.0, "const": False}
        assert bounds_of(ws)["mu"] == (0.0, 10.0)


class TestNormSys:
    """``normsys``: Gauss-constrained, init 0, bounds [-5, 5], code4 by default."""

    def test_modifier(self):
        ws = from_hifa(make_spec([NORMSYS]))
        assert modifiers_of(ws, "background") == [
            {
                "name": "bkg_norm",
                "type": "normsys",
                "parameter": "bkg_norm",
                "constraint": "Gauss",
                "data": {"hi": 1.1, "lo": 0.9, "interpolation": "code4"},
            }
        ]

    def test_defaults(self):
        ws = from_hifa(make_spec([NORMSYS]))
        assert points_of(ws)["bkg_norm"] == {"value": 0.0, "const": False}
        assert bounds_of(ws)["bkg_norm"] == (-5.0, 5.0)

    def test_interpcode_override(self):
        ws = from_hifa(
            make_spec([NORMSYS]), modifier_settings={"normsys": {"interpcode": "code1"}}
        )
        assert modifiers_of(ws, "background")[0]["data"]["interpolation"] == "code1"


class TestHistoSys:
    """``histosys``: Gauss-constrained, init 0, bounds [-5, 5], code4p by default."""

    def test_modifier(self):
        ws = from_hifa(make_spec([HISTOSYS]))
        assert modifiers_of(ws, "background") == [
            {
                "name": "bkg_shape",
                "type": "histosys",
                "parameter": "bkg_shape",
                "constraint": "Gauss",
                "data": {
                    "hi": {"contents": [55.0, 62.0]},
                    "lo": {"contents": [45.0, 58.0]},
                    "interpolation": "code4p",
                },
            }
        ]

    def test_defaults(self):
        ws = from_hifa(make_spec([HISTOSYS]))
        assert points_of(ws)["bkg_shape"] == {"value": 0.0, "const": False}
        assert bounds_of(ws)["bkg_shape"] == (-5.0, 5.0)

    @pytest.mark.parametrize("code", ["code0", "code2", "code4p"])
    def test_interpcode_override(self, code):
        ws = from_hifa(
            make_spec([HISTOSYS]), modifier_settings={"histosys": {"interpcode": code}}
        )
        assert modifiers_of(ws, "background")[0]["data"]["interpolation"] == code

    def test_override_one_type_keeps_other_default(self):
        ws = from_hifa(
            make_spec([NORMSYS, HISTOSYS]),
            modifier_settings={"histosys": {"interpcode": "code0"}},
        )
        normsys, histosys = modifiers_of(ws, "background")
        assert normsys["data"]["interpolation"] == "code4"
        assert histosys["data"]["interpolation"] == "code0"

    def test_shared_with_normsys(self):
        """pyhf correlates a normsys and a histosys of the same name."""
        shared = {**HISTOSYS, "name": "bkg_norm"}
        ws = from_hifa(make_spec([NORMSYS, shared]))
        assert [m["parameter"] for m in modifiers_of(ws, "background")] == [
            "bkg_norm",
            "bkg_norm",
        ]
        assert points_of(ws)["bkg_norm"] == {"value": 0.0, "const": False}


class TestShapeFactor:
    """``shapefactor``: per-bin, unconstrained, init 1, bounds [0, 10]."""

    def test_modifier(self):
        ws = from_hifa(make_spec([SHAPEFACTOR]))
        assert modifiers_of(ws, "background") == [
            {
                "name": "sf",
                "type": "shapefactor",
                "parameters": ["sf_bin_0", "sf_bin_1"],
            }
        ]

    def test_defaults(self):
        ws = from_hifa(make_spec([SHAPEFACTOR]))
        points = points_of(ws)
        bounds = bounds_of(ws)
        for name in ("sf_bin_0", "sf_bin_1"):
            assert points[name] == {"value": 1.0, "const": False}
            assert bounds[name] == (0.0, 10.0)
        assert "sf" not in points

    def test_shared_across_samples_and_channels(self):
        """pyhf shares a shapefactor's bin ``i`` wherever the name appears."""
        spec = make_spec([SHAPEFACTOR])
        add_signal_modifiers(spec, SHAPEFACTOR)
        add_channel(spec, "CR", 2, SHAPEFACTOR)
        ws = from_hifa(spec)
        expected = ["sf_bin_0", "sf_bin_1"]
        assert modifiers_of(ws, "signal")[1]["parameters"] == expected
        assert modifiers_of(ws, "background")[0]["parameters"] == expected
        assert modifiers_of(ws, "background", "CR")[0]["parameters"] == expected

    def test_bin_count_mismatch_across_channels_raises(self):
        spec = make_spec([SHAPEFACTOR])
        add_channel(spec, "CR", 3, SHAPEFACTOR)
        with pytest.raises(ValueError, match=r"'sf'.*3.*2|'sf'.*2.*3"):
            from_hifa(spec)

    def test_name_reused_by_normfactor_raises(self):
        """pyhf rejects a name used by a scalar and a per-bin parameter."""
        with pytest.raises(ValueError, match=r"'mu'.*normfactor.*shapefactor"):
            from_hifa(make_spec([{**SHAPEFACTOR, "name": "mu"}]))


class TestShapeSys:
    """``shapesys``: per-bin, Poisson-constrained, init 1, bounds [1e-10, 10]."""

    def test_modifier(self):
        ws = from_hifa(make_spec([SHAPESYS]))
        assert modifiers_of(ws, "background") == [
            {
                "name": "bkg_unc",
                "type": "shapesys",
                "parameters": ["bkg_unc_bin_0", "bkg_unc_bin_1"],
                "constraint": "Poisson",
                "data": {"vals": [5.0, 6.0]},
            }
        ]

    def test_defaults(self):
        ws = from_hifa(make_spec([SHAPESYS]))
        points = points_of(ws)
        bounds = bounds_of(ws)
        for name in ("bkg_unc_bin_0", "bkg_unc_bin_1"):
            assert points[name] == {"value": 1.0, "const": False}
            assert bounds[name] == (1e-10, 10.0)

    def test_zero_uncertainty_bin_const(self):
        ws = from_hifa(make_spec([{**SHAPESYS, "data": [5.0, 0.0]}]))
        points = points_of(ws)
        assert points["bkg_unc_bin_0"] == {"value": 1.0, "const": False}
        assert points["bkg_unc_bin_1"] == {"value": 1.0, "const": True}

    def test_zero_yield_bin_const(self):
        spec = make_spec([SHAPESYS])
        spec["channels"][0]["samples"][1]["data"] = [0.0, 60.0]
        points = points_of(from_hifa(spec))
        assert points["bkg_unc_bin_0"] == {"value": 1.0, "const": True}
        assert points["bkg_unc_bin_1"] == {"value": 1.0, "const": False}

    def test_shared_across_samples_raises(self):
        spec = make_spec([SHAPESYS])
        add_signal_modifiers(spec, SHAPESYS)
        with pytest.raises(ValueError, match=r"shapesys.*'bkg_unc'.*more than one"):
            from_hifa(spec)

    def test_shared_across_channels_raises(self):
        spec = make_spec([SHAPESYS])
        add_channel(spec, "CR", 2, SHAPESYS)
        with pytest.raises(ValueError, match=r"shapesys.*'bkg_unc'.*more than one"):
            from_hifa(spec)

    def test_name_reused_by_staterror_raises(self):
        """pyhf rejects one name for a Poisson and a Gauss constraint."""
        spec = make_spec([SHAPESYS])
        add_signal_modifiers(spec, {**STATERROR, "name": "bkg_unc"})
        with pytest.raises(ValueError, match=r"'bkg_unc'.*shapesys.*staterror"):
            from_hifa(spec)


class TestStatError:
    """``staterror``: Barlow-Beeston lite, Gauss, init 1, bounds [1e-10, 10]."""

    def test_modifier(self):
        ws = from_hifa(make_spec([STATERROR]))
        dist = ws.distributions["model_SR"]
        assert dist.barlow_beeston_method == "lite"
        assert modifiers_of(ws, "background") == [
            {
                "name": "staterror_SR",
                "type": "staterror",
                "parameters": ["staterror_SR_bin_0", "staterror_SR_bin_1"],
                "constraint": "Gauss",
                "data": None,
            }
        ]
        assert dist.samples["background"].data.errors == [2.0, 3.0]

    def test_sample_without_staterror_has_no_errors(self):
        ws = from_hifa(make_spec([STATERROR]))
        signal = ws.distributions["model_SR"].samples["signal"]
        assert signal.data.model_dump() == {"contents": [5.0, 10.0]}

    def test_defaults(self):
        ws = from_hifa(make_spec([STATERROR]))
        points = points_of(ws)
        bounds = bounds_of(ws)
        for name in ("staterror_SR_bin_0", "staterror_SR_bin_1"):
            assert points[name] == {"value": 1.0, "const": False}
            assert bounds[name] == (1e-10, 10.0)

    def test_shared_across_samples(self):
        spec = make_spec([STATERROR])
        add_signal_modifiers(spec, {**STATERROR, "data": [1.0, 1.5]})
        ws = from_hifa(spec)
        dist = ws.distributions["model_SR"]
        expected = ["staterror_SR_bin_0", "staterror_SR_bin_1"]
        assert modifiers_of(ws, "signal")[1]["parameters"] == expected
        assert modifiers_of(ws, "background")[0]["parameters"] == expected
        assert dist.samples["signal"].data.errors == [1.0, 1.5]
        assert dist.samples["background"].data.errors == [2.0, 3.0]

    def test_global_bin_index_across_channels(self):
        """pyhf numbers the bins of the channels using the name, sorted by name.

        Channel ``AA`` sorts first but does not use the name, so it takes no
        index; ``CR`` (3 bins) takes 0-2 and ``SR`` (2 bins) takes 3-4.
        """
        spec = make_spec([STATERROR])
        add_channel(spec, "CR", 3, {**STATERROR, "data": [1.0, 1.0, 1.0]})
        add_channel(spec, "AA", 1)
        ws = from_hifa(spec)
        assert modifiers_of(ws, "background", "CR")[0]["parameters"] == [
            "staterror_SR_bin_0",
            "staterror_SR_bin_1",
            "staterror_SR_bin_2",
        ]
        assert modifiers_of(ws, "background")[0]["parameters"] == [
            "staterror_SR_bin_3",
            "staterror_SR_bin_4",
        ]

    def test_sample_missing_from_a_channel_of_the_name_raises(self):
        """pyhf (an ``AssertionError``) needs each sample on every channel's group."""
        spec = make_spec([STATERROR])
        add_signal_modifiers(spec, {**STATERROR, "data": [1.0, 1.5]})
        add_channel(spec, "CR", 3, {**STATERROR, "data": [1.0, 1.0, 1.0]})
        with pytest.raises(
            ValueError, match=r"'staterror_SR'.*'signal'.*\['SR'\].*\['CR', 'SR'\]"
        ):
            from_hifa(spec)

    def test_zero_uncertainty_bin_const(self):
        ws = from_hifa(make_spec([{**STATERROR, "data": [2.0, 0.0]}]))
        points = points_of(ws)
        assert points["staterror_SR_bin_0"] == {"value": 1.0, "const": False}
        assert points["staterror_SR_bin_1"] == {"value": 1.0, "const": True}

    def test_zero_summed_yield_bin_const(self):
        spec = make_spec([{**STATERROR, "data": [2.0, 1.0]}])
        spec["channels"][0]["samples"][1]["data"] = [50.0, 0.0]
        points = points_of(from_hifa(spec))
        assert points["staterror_SR_bin_0"] == {"value": 1.0, "const": False}
        assert points["staterror_SR_bin_1"] == {"value": 1.0, "const": True}

    def test_extra_single_sample_group_becomes_gauss_shapesys(self):
        """A second staterror group on one sample is a Gauss shapesys.

        pyhs3's lite mode holds one staterror group per channel; a group on a
        single sample has relative uncertainty delta / nominal, which a Gauss
        shapesys with absolute ``vals`` = delta reproduces exactly.
        """
        extra = {"name": "stat_extra", "type": "staterror", "data": [4.0, 0.0]}
        spec = make_spec([STATERROR, extra])
        add_signal_modifiers(spec, {**STATERROR, "data": [1.0, 1.5]})
        ws = from_hifa(spec)
        assert modifiers_of(ws, "background")[1] == {
            "name": "stat_extra",
            "type": "shapesys",
            "parameters": ["stat_extra_bin_0", "stat_extra_bin_1"],
            "constraint": "Gauss",
            "data": {"vals": [4.0, 0.0]},
        }
        # The sample's errors stay those of the lite group.
        dist = ws.distributions["model_SR"]
        assert dist.samples["background"].data.errors == [2.0, 3.0]
        points = points_of(ws)
        assert points["stat_extra_bin_0"] == {"value": 1.0, "const": False}
        assert points["stat_extra_bin_1"] == {"value": 1.0, "const": True}
        assert bounds_of(ws)["stat_extra_bin_0"] == (1e-10, 10.0)

    def test_extra_group_negative_uncertainty(self):
        """pyhf squares staterror uncertainties, so their sign does not matter."""
        extra = {"name": "stat_extra", "type": "staterror", "data": [-4.0, 3.0]}
        spec = make_spec([STATERROR, extra])
        add_signal_modifiers(spec, {**STATERROR, "data": [1.0, 1.5]})
        ws = from_hifa(spec)
        assert modifiers_of(ws, "background")[1]["data"] == {"vals": [4.0, 3.0]}

    def test_two_single_sample_groups(self):
        """The first group to appear stays the lite group."""
        spec = make_spec([STATERROR])
        add_signal_modifiers(spec, {**STATERROR, "name": "stat_sig"})
        ws = from_hifa(spec)
        assert modifiers_of(ws, "signal")[1]["type"] == "staterror"
        assert modifiers_of(ws, "background")[0]["type"] == "shapesys"
        dist = ws.distributions["model_SR"]
        assert dist.samples["signal"].data.errors == [2.0, 3.0]
        assert dist.samples["background"].data.model_dump() == {
            "contents": [50.0, 60.0]
        }

    def test_multi_sample_group_preferred_as_lite(self):
        """The lite group is the one on several samples, wherever it appears."""
        extra = {"name": "stat_extra", "type": "staterror", "data": [4.0, 4.0]}
        spec = make_spec([extra, STATERROR])
        add_signal_modifiers(spec, STATERROR)
        ws = from_hifa(spec)
        assert [m["type"] for m in modifiers_of(ws, "background")] == [
            "shapesys",
            "staterror",
        ]

    def test_two_multi_sample_groups_raises(self):
        extra = {"name": "stat_extra", "type": "staterror", "data": [4.0, 4.0]}
        spec = make_spec([STATERROR, extra])
        add_signal_modifiers(spec, STATERROR, extra)
        with pytest.raises(
            ValueError, match=r"'SR'.*staterror_SR.*stat_extra|'SR'.*stat_extra"
        ):
            from_hifa(spec)

    def test_lite_in_one_channel_extra_in_another(self):
        """Lite or extra is decided per channel."""
        extra = {"name": "stat_extra", "type": "staterror", "data": [4.0, 4.0]}
        spec = make_spec([STATERROR, extra])
        add_signal_modifiers(spec, {**STATERROR, "data": [1.0, 1.5]})
        add_channel(spec, "CR", 2, extra)
        ws = from_hifa(spec)
        assert modifiers_of(ws, "background")[1]["type"] == "shapesys"
        cr = modifiers_of(ws, "background", "CR")[0]
        assert cr["type"] == "staterror"
        # CR sorts before SR, so it takes the first global indices.
        assert cr["parameters"] == ["stat_extra_bin_0", "stat_extra_bin_1"]
        assert modifiers_of(ws, "background")[1]["parameters"] == [
            "stat_extra_bin_2",
            "stat_extra_bin_3",
        ]

    def test_model_builds(self):
        extra = {"name": "stat_extra", "type": "staterror", "data": [4.0, 0.0]}
        spec = make_spec([STATERROR, extra, SHAPESYS, SHAPEFACTOR])
        add_signal_modifiers(spec, {**STATERROR, "data": [1.0, 1.5]})
        ws = from_hifa(spec)
        model = ws.model(ws.analyses["meas"], progress=False)
        assert set(model.free_params) == {
            "mu",
            "staterror_SR_bin_0",
            "staterror_SR_bin_1",
            "stat_extra_bin_0",
            "bkg_unc_bin_0",
            "bkg_unc_bin_1",
            "sf_bin_0",
            "sf_bin_1",
        }


class TestPerBinMeasurementConfig:
    """Measurement-config entries for per-bin parameters."""

    def test_inits_bounds_per_bin(self):
        ws = from_hifa(
            make_spec(
                [SHAPESYS],
                parameters=[
                    {
                        "name": "bkg_unc",
                        "inits": [1.1, 0.9],
                        "bounds": [[0.5, 1.5], [0.1, 2.0]],
                    }
                ],
            )
        )
        points = points_of(ws)
        assert points["bkg_unc_bin_0"] == {"value": 1.1, "const": False}
        assert points["bkg_unc_bin_1"] == {"value": 0.9, "const": False}
        bounds = bounds_of(ws)
        assert bounds["bkg_unc_bin_0"] == (0.5, 1.5)
        assert bounds["bkg_unc_bin_1"] == (0.1, 2.0)

    def test_fixed_applies_to_every_bin(self):
        ws = from_hifa(
            make_spec([SHAPEFACTOR], parameters=[{"name": "sf", "fixed": True}])
        )
        points = points_of(ws)
        assert points["sf_bin_0"]["const"] is True
        assert points["sf_bin_1"]["const"] is True

    def test_init_of_unconstrained_bin_used(self):
        """A bin with no constraint stays const, at its configured init."""
        ws = from_hifa(
            make_spec(
                [{**STATERROR, "data": [2.0, 0.0]}],
                parameters=[{"name": "staterror_SR", "inits": [1.2, 1.3]}],
            )
        )
        assert points_of(ws)["staterror_SR_bin_1"] == {"value": 1.3, "const": True}

    def test_fixed_false_with_unconstrained_bin_raises(self):
        """pyhf would free the bin under a dummy constraint HS3 cannot express."""
        with pytest.raises(ValueError, match=r"'staterror_SR'.*fixed.*bins \[1\]"):
            from_hifa(
                make_spec(
                    [{**STATERROR, "data": [2.0, 0.0]}],
                    parameters=[{"name": "staterror_SR", "fixed": False}],
                )
            )

    def test_fixed_false_without_unconstrained_bin_accepted(self):
        ws = from_hifa(
            make_spec(
                [STATERROR], parameters=[{"name": "staterror_SR", "fixed": False}]
            )
        )
        assert points_of(ws)["staterror_SR_bin_1"]["const"] is False

    @pytest.mark.parametrize(
        ("modifier", "entry"),
        [
            (SHAPESYS, {"name": "bkg_unc", "inits": [1.0]}),
            (SHAPESYS, {"name": "bkg_unc", "bounds": [[0.0, 2.0]]}),
            (STATERROR, {"name": "staterror_SR", "inits": [1.0, 1.0, 1.0]}),
            (SHAPEFACTOR, {"name": "sf", "inits": [1.0]}),
        ],
        ids=["shapesys-inits", "shapesys-bounds", "staterror-inits", "shapefactor"],
    )
    def test_wrong_length_raises(self, modifier, entry):
        with pytest.raises(ValueError, match=rf"'{entry['name']}'.*expected 2"):
            from_hifa(make_spec([modifier], parameters=[entry]))

    def test_shapesys_default_aux_accepted(self):
        # tau = (nominal / delta)^2 = (50 / 5)^2 = (60 / 6)^2 = 100.
        ws = from_hifa(
            make_spec(
                [SHAPESYS],
                parameters=[
                    {
                        "name": "bkg_unc",
                        "auxdata": [100.0, 100.0],
                        "factors": [100.0, 100.0],
                    }
                ],
            )
        )
        assert points_of(ws)["bkg_unc_bin_0"]["const"] is False

    def test_shapesys_unconstrained_bin_factor_is_one(self):
        """pyhf sets tau = 1 for a bin it fixes."""
        ws = from_hifa(
            make_spec(
                [{**SHAPESYS, "data": [5.0, 0.0]}],
                parameters=[{"name": "bkg_unc", "auxdata": [100.0, 1.0]}],
            )
        )
        assert points_of(ws)["bkg_unc_bin_1"]["const"] is True

    @pytest.mark.parametrize(
        ("key", "value"),
        [("auxdata", [100.0, 50.0]), ("factors", [1.0, 1.0]), ("sigmas", [0.1, 0.1])],
    )
    def test_shapesys_non_default_aux_raises(self, key, value):
        with pytest.raises(ValueError, match=rf"'bkg_unc'.*{key}"):
            from_hifa(
                make_spec([SHAPESYS], parameters=[{"name": "bkg_unc", key: value}])
            )

    def test_staterror_default_aux_accepted(self):
        # sigma = delta / nominal = [2 / 50, 3 / 60].
        ws = from_hifa(
            make_spec(
                [STATERROR],
                parameters=[
                    {
                        "name": "staterror_SR",
                        "auxdata": [1.0, 1.0],
                        "sigmas": [0.04, 0.05],
                    }
                ],
            )
        )
        assert points_of(ws)["staterror_SR_bin_0"]["const"] is False

    def test_staterror_unconstrained_bin_sigma_is_one(self):
        """pyhf sets sigma = 1 for a bin it fixes."""
        ws = from_hifa(
            make_spec(
                [{**STATERROR, "data": [2.0, 0.0]}],
                parameters=[{"name": "staterror_SR", "sigmas": [0.04, 1.0]}],
            )
        )
        assert points_of(ws)["staterror_SR_bin_1"]["const"] is True

    @pytest.mark.parametrize(
        ("key", "value"),
        [("auxdata", [0.0, 0.0]), ("sigmas", [0.1, 0.1]), ("factors", [1.0, 1.0])],
    )
    def test_staterror_non_default_aux_raises(self, key, value):
        with pytest.raises(ValueError, match=rf"'staterror_SR'.*{key}"):
            from_hifa(
                make_spec(
                    [STATERROR], parameters=[{"name": "staterror_SR", key: value}]
                )
            )

    def test_extra_staterror_group_validated_as_staterror(self):
        """The config of a group emitted as shapesys follows pyhf's staterror."""
        extra = {"name": "stat_extra", "type": "staterror", "data": [4.0, 6.0]}
        spec = make_spec(
            [STATERROR, extra],
            parameters=[{"name": "stat_extra", "sigmas": [0.08, 0.1]}],
        )
        add_signal_modifiers(spec, {**STATERROR, "data": [1.0, 1.5]})
        assert points_of(from_hifa(spec))["stat_extra_bin_0"]["const"] is False

    @pytest.mark.parametrize("key", ["auxdata", "sigmas", "factors"])
    def test_shapefactor_aux_raises(self, key):
        with pytest.raises(ValueError, match=rf"'sf'.*{key}"):
            from_hifa(
                make_spec([SHAPEFACTOR], parameters=[{"name": "sf", key: [1.0, 1.0]}])
            )


class TestMeasurementConfig:
    """``inits``, ``bounds``, ``fixed`` and the POI from the measurement config."""

    def test_inits_bounds_fixed(self):
        ws = from_hifa(
            make_spec(
                [NORMSYS, HISTOSYS],
                parameters=[
                    {"name": "mu", "inits": [2.0], "bounds": [[-1.0, 20.0]]},
                    {"name": "bkg_norm", "fixed": True, "inits": [0.5]},
                ],
            )
        )
        points = points_of(ws)
        assert points == {
            "bkg_norm": {"value": 0.5, "const": True},
            "bkg_shape": {"value": 0.0, "const": False},
            "mu": {"value": 2.0, "const": False},
        }
        bounds = bounds_of(ws)
        assert bounds == {
            "bkg_norm": (-5.0, 5.0),
            "bkg_shape": (-5.0, 5.0),
            "mu": (-1.0, 20.0),
        }

    def test_every_parameter_listed_in_name_order(self):
        ws = from_hifa(make_spec([NORMSYS, HISTOSYS]))
        names = ["bkg_norm", "bkg_shape", "mu"]
        assert [p.name for p in ws.parameter_points["meas"].parameters] == names
        assert ws.domains["meas"].axis_names == names

    def test_parameter_absent_from_model_ignored(self):
        ws = from_hifa(make_spec(parameters=[{"name": "not_in_model", "inits": [3.0]}]))
        assert "not_in_model" not in points_of(ws)
        assert "not_in_model" not in bounds_of(ws)

    def test_poi_absent_from_model_copied(self):
        ws = from_hifa(make_spec(poi="not_in_model"))
        assert ws.analyses["meas"].parameters_of_interest == ["not_in_model"]

    def test_measurements_independent(self):
        spec = make_spec([NORMSYS])
        spec["measurements"].append(
            {
                "name": "fixed_bkg",
                "config": {
                    "poi": "mu",
                    "parameters": [{"name": "bkg_norm", "fixed": True}],
                },
            }
        )
        ws = from_hifa(spec)
        assert points_of(ws, "meas")["bkg_norm"]["const"] is False
        assert points_of(ws, "fixed_bkg")["bkg_norm"]["const"] is True

    @pytest.mark.parametrize(
        "entry",
        [
            {"name": "mu", "inits": [1.0, 2.0]},
            {"name": "mu", "bounds": [[0.0, 1.0], [0.0, 2.0]]},
        ],
        ids=["inits", "bounds"],
    )
    def test_wrong_length_raises(self, entry):
        with pytest.raises(ValueError, match=r"'mu'.*expected 1"):
            from_hifa(make_spec(parameters=[entry]))

    @pytest.mark.parametrize("key", ["auxdata", "sigmas", "factors"])
    def test_normfactor_constraint_keys_raise(self, key):
        with pytest.raises(ValueError, match=rf"'mu'.*{key}"):
            from_hifa(make_spec(parameters=[{"name": "mu", key: [1.0]}]))

    @pytest.mark.parametrize(
        ("key", "value"),
        [("auxdata", [0.5]), ("sigmas", [2.0]), ("factors", [1.0])],
    )
    def test_constrained_non_default_keys_raise(self, key, value):
        with pytest.raises(ValueError, match=rf"'bkg_norm'.*{key}"):
            from_hifa(
                make_spec([NORMSYS], parameters=[{"name": "bkg_norm", key: value}])
            )

    def test_constrained_default_aux_accepted(self):
        ws = from_hifa(
            make_spec(
                [NORMSYS],
                parameters=[{"name": "bkg_norm", "auxdata": [0.0], "sigmas": [1.0]}],
            )
        )
        assert points_of(ws)["bkg_norm"] == {"value": 0.0, "const": False}


class TestModifierSettings:
    """Validation of ``modifier_settings``."""

    @pytest.mark.parametrize(
        "settings",
        [
            {"normsys": {"interpcode": "code0"}},
            {"normsys": {"interpcode": "code2"}},
            {"histosys": {"interpcode": "code1"}},
            {"histosys": {"interpcode": "code4"}},
        ],
    )
    def test_unsupported_code_raises(self, settings):
        with pytest.raises(ValidationError, match="interpcode"):
            from_hifa(make_spec([NORMSYS, HISTOSYS]), modifier_settings=settings)

    def test_unknown_modifier_type_raises(self):
        with pytest.raises(ValidationError, match="shapesys"):
            from_hifa(
                make_spec(), modifier_settings={"shapesys": {"interpcode": "code4"}}
            )

    def test_unknown_key_raises(self):
        with pytest.raises(ValidationError, match="interp"):
            from_hifa(make_spec(), modifier_settings={"normsys": {"interp": "code4"}})


class TestLumi:
    """``lumi``: a normfactor with a Gauss constraint, all set by the config."""

    def test_modifier(self):
        ws = from_hifa(make_spec([LUMI], parameters=[LUMI_CONFIG]))
        assert modifiers_of(ws, "background") == [
            {
                "name": "lumi",
                "type": "normfactor",
                "parameter": "lumi",
                "constraint": None,
            }
        ]

    def test_constraint_distribution(self):
        ws = from_hifa(make_spec([LUMI], parameters=[LUMI_CONFIG]))
        assert ws.distributions["lumi_constraint_meas"].model_dump() == {
            "name": "lumi_constraint_meas",
            "type": "gaussian_dist",
            "x": 1.0,
            "mean": "lumi",
            "sigma": 0.1,
        }
        likelihood = ws.likelihoods["meas"]
        assert [d.name for d in likelihood.aux_distributions] == [
            "lumi_constraint_meas"
        ]
        assert likelihood.distribution_names() == ["model_SR", "lumi_constraint_meas"]

    def test_no_lumi_no_aux_distributions(self):
        ws = from_hifa(make_spec([NORMSYS]))
        assert ws.likelihoods["meas"].aux_distributions is None

    def test_inits_bounds(self):
        ws = from_hifa(make_spec([LUMI], parameters=[LUMI_CONFIG]))
        assert points_of(ws)["lumi"] == {"value": 1.0, "const": False}
        assert bounds_of(ws)["lumi"] == (0.5, 1.5)

    def test_one_parameter_for_every_use(self):
        spec = make_spec([LUMI], parameters=[LUMI_CONFIG])
        add_signal_modifiers(spec, LUMI)
        add_channel(spec, "CR", 3, LUMI)
        ws = from_hifa(spec)
        assert [p.name for p in ws.parameter_points["meas"].parameters] == [
            "lumi",
            "mu",
        ]
        for sample, channel in [
            ("signal", "SR"),
            ("background", "SR"),
            ("background", "CR"),
        ]:
            assert modifiers_of(ws, sample, channel)[-1]["parameter"] == "lumi"
        assert [d.name for d in ws.likelihoods["meas"].aux_distributions] == [
            "lumi_constraint_meas"
        ]

    def test_constraint_per_measurement(self):
        spec = make_spec([LUMI], parameters=[LUMI_CONFIG])
        other = {**LUMI_CONFIG, "auxdata": [1.2], "sigmas": [0.05]}
        spec["measurements"].append(
            {"name": "other", "config": {"poi": "mu", "parameters": [other]}}
        )
        ws = from_hifa(spec)
        for name, x, sigma in [("meas", 1.0, 0.1), ("other", 1.2, 0.05)]:
            dist = ws.distributions[f"lumi_constraint_{name}"]
            assert (dist.x, dist.mean, dist.sigma) == (x, "lumi", sigma)
            assert [d.name for d in ws.likelihoods[name].aux_distributions] == [
                f"lumi_constraint_{name}"
            ]

    def test_fixed_keeps_constraint(self):
        entry = {**LUMI_CONFIG, "inits": [1.1], "fixed": True}
        ws = from_hifa(make_spec([LUMI], parameters=[entry]))
        assert points_of(ws)["lumi"] == {"value": 1.1, "const": True}
        assert [d.name for d in ws.likelihoods["meas"].aux_distributions] == [
            "lumi_constraint_meas"
        ]

    @pytest.mark.parametrize("key", ["inits", "bounds", "auxdata", "sigmas"])
    def test_missing_config_key_raises(self, key):
        entry = {k: v for k, v in LUMI_CONFIG.items() if k != key}
        with pytest.raises(ValueError, match=rf"'meas'.*'lumi'.*{key}"):
            from_hifa(make_spec([LUMI], parameters=[entry]))

    def test_missing_config_entry_raises(self):
        with pytest.raises(ValueError, match=r"'meas'.*'lumi'.*auxdata"):
            from_hifa(make_spec([LUMI]))

    def test_missing_in_second_measurement_raises(self):
        spec = make_spec([LUMI], parameters=[LUMI_CONFIG])
        spec["measurements"].append(
            {"name": "other", "config": {"poi": "mu", "parameters": []}}
        )
        with pytest.raises(ValueError, match=r"'other'.*'lumi'"):
            from_hifa(spec)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("inits", [1.0, 1.0]),
            ("bounds", [[0.5, 1.5], [0.5, 1.5]]),
            ("auxdata", [1.0, 1.0]),
            ("sigmas", [0.1, 0.1]),
        ],
    )
    def test_wrong_length_raises(self, key, value):
        entry = {**LUMI_CONFIG, key: value}
        with pytest.raises(ValueError, match=rf"'lumi'.*2 {key}, expected 1"):
            from_hifa(make_spec([LUMI], parameters=[entry]))

    def test_factors_raises(self):
        entry = {**LUMI_CONFIG, "factors": [1.0]}
        with pytest.raises(ValueError, match=r"'lumi'.*factors"):
            from_hifa(make_spec([LUMI], parameters=[entry]))

    @pytest.mark.parametrize("sigma", [0.0, -0.1])
    def test_non_positive_sigma_raises(self, sigma):
        entry = {**LUMI_CONFIG, "sigmas": [sigma]}
        with pytest.raises(ValueError, match=r"'lumi'.*sigmas.*positive"):
            from_hifa(make_spec([LUMI], parameters=[entry]))

    @pytest.mark.parametrize(
        "modifier",
        [
            {"type": "normfactor", "data": None},
            {"type": "normsys", "data": {"hi": 1.1, "lo": 0.9}},
            {"type": "histosys", "data": HISTOSYS["data"]},
            {"type": "shapefactor", "data": None},
            {"type": "shapesys", "data": [5.0, 6.0]},
            {"type": "staterror", "data": [2.0, 3.0]},
        ],
        ids=lambda m: m["type"],
    )
    def test_name_reused_by_other_type_raises(self, modifier):
        spec = make_spec([LUMI], parameters=[LUMI_CONFIG])
        add_signal_modifiers(spec, {"name": "lumi", **modifier})
        with pytest.raises(ValueError, match=rf"'lumi'.*{modifier['type']}"):
            from_hifa(spec)

    def test_model_builds(self):
        ws = from_hifa(make_spec([LUMI], parameters=[LUMI_CONFIG]))
        model = ws.model(ws.analyses["meas"], progress=False)
        assert set(model.free_params) == {"mu", "lumi"}


class TestInputValidation:
    """The private schema models reject input pyhf's schema rejects."""

    def test_unknown_modifier_type(self):
        with pytest.raises(ValidationError, match="bogus"):
            from_hifa(make_spec([{"name": "x", "type": "bogus", "data": None}]))

    def test_normsys_missing_lo(self):
        with pytest.raises(ValidationError, match="lo"):
            from_hifa(
                make_spec([{"name": "x", "type": "normsys", "data": {"hi": 1.1}}])
            )

    def test_normfactor_with_data(self):
        with pytest.raises(ValidationError):
            from_hifa(make_spec([{"name": "x", "type": "normfactor", "data": 1.0}]))

    def test_extra_top_level_key(self):
        spec = make_spec()
        spec["bogus"] = 1
        with pytest.raises(ValidationError, match="bogus"):
            from_hifa(spec)

    def test_wrong_version(self):
        spec = make_spec()
        spec["version"] = "2.0.0"
        with pytest.raises(ValidationError, match="version"):
            from_hifa(spec)

    @pytest.mark.parametrize("key", ["channels", "observations", "measurements"])
    def test_empty_list(self, key):
        spec = make_spec()
        spec[key] = []
        with pytest.raises(ValidationError, match=key):
            from_hifa(spec)

    def test_missing_poi(self):
        spec = make_spec()
        del spec["measurements"][0]["config"]["poi"]
        with pytest.raises(ValidationError, match="poi"):
            from_hifa(spec)


class TestConsistencyChecks:
    """Checks pyhf makes when it builds a model, made at conversion time."""

    def test_sample_length_mismatch(self):
        spec = make_spec()
        spec["channels"][0]["samples"][1]["data"] = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match=r"'background'.*3 bins.*2"):
            from_hifa(spec)

    def test_observation_length_mismatch(self):
        spec = make_spec()
        spec["observations"][0]["data"] = [1.0]
        with pytest.raises(ValueError, match=r"'SR'.*1 bins.*2"):
            from_hifa(spec)

    def test_histosys_length_mismatch(self):
        bad = {
            "name": "bkg_shape",
            "type": "histosys",
            "data": {"hi_data": [55.0], "lo_data": [45.0, 58.0]},
        }
        with pytest.raises(ValueError, match=r"'bkg_shape'.*hi_data"):
            from_hifa(make_spec([bad]))

    @pytest.mark.parametrize(
        "modifier", [SHAPESYS, STATERROR], ids=["shapesys", "staterror"]
    )
    def test_per_bin_data_length_mismatch(self, modifier):
        bad = {**modifier, "data": [1.0, 2.0, 3.0]}
        with pytest.raises(ValueError, match=rf"'{modifier['name']}'.*3 bins.*2"):
            from_hifa(make_spec([bad]))

    def test_missing_observation(self):
        spec = make_spec()
        spec["channels"].append({**copy.deepcopy(spec["channels"][0]), "name": "CR"})
        with pytest.raises(ValueError, match="channel 'CR' has no observation"):
            from_hifa(spec)

    def test_observation_for_unknown_channel_is_ignored(self, caplog):
        # A signal patch can remove a channel and leave its observation;
        # pyhf ignores the observation, so the conversion does too.
        spec = make_spec()
        spec["observations"].append({"name": "CR", "data": [1.0, 2.0, 3.0]})
        with caplog.at_level(logging.DEBUG, logger="pyhs3.hifa"):
            ws = from_hifa(spec)
        assert [d.name for d in ws.data] == ["obsData_SR"]
        assert "Observation 'CR' has no channel" in caplog.text

    @pytest.mark.parametrize("key", ["channels", "observations", "measurements"])
    def test_duplicate_names(self, key):
        spec = make_spec()
        spec[key].append(copy.deepcopy(spec[key][0]))
        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            from_hifa(spec)

    def test_duplicate_config_parameter(self):
        entry = {"name": "mu", "inits": [2.0]}
        with pytest.raises(ValueError, match=r"Duplicate parameter.*'meas'.*mu"):
            from_hifa(make_spec(parameters=[entry, entry]))

    def test_name_reused_across_incompatible_types(self):
        clash = {"name": "mu", "type": "normsys", "data": {"hi": 1.1, "lo": 0.9}}
        with pytest.raises(ValueError, match=r"'mu'.*normfactor.*normsys"):
            from_hifa(make_spec([clash]))
