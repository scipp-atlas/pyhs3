"""Unit tests for the pyhf (HiFa JSON) to HS3 converter, ``pyhs3.from_hifa``.

Each test feeds a pyhf workspace dict in and checks the HS3 structure that
comes out, without pyhf itself. ``tests/test_hifa_pyhf.py`` compares the
converted models numerically against pyhf.
"""

from __future__ import annotations

import copy
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


def modifiers_of(ws: pyhs3.Workspace, sample: str) -> list[dict[str, Any]]:
    """Serialized modifiers of one sample of channel ``SR``."""
    dist = ws.distributions["model_SR"]
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


class TestUnsupportedModifiers:
    """Modifier types later PRs add raise ``NotImplementedError`` for now."""

    @pytest.mark.parametrize(
        "modifier",
        [
            {"name": "sys", "type": "shapesys", "data": [1.0, 2.0]},
            {"name": "sf", "type": "shapefactor", "data": None},
            {"name": "staterror_SR", "type": "staterror", "data": [1.0, 2.0]},
            {"name": "lumi", "type": "lumi", "data": None},
        ],
        ids=lambda m: m["type"],
    )
    def test_raises(self, modifier):
        with pytest.raises(NotImplementedError, match=modifier["type"]):
            from_hifa(make_spec([modifier]))


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

    def test_missing_observation(self):
        spec = make_spec()
        spec["channels"].append({**copy.deepcopy(spec["channels"][0]), "name": "CR"})
        with pytest.raises(ValueError, match="channel 'CR' has no observation"):
            from_hifa(spec)

    def test_observation_for_unknown_channel(self):
        spec = make_spec()
        spec["observations"].append({"name": "CR", "data": [1.0, 2.0]})
        with pytest.raises(ValueError, match=r"'CR'.*no channel"):
            from_hifa(spec)

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
