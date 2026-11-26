.. diataxis: how-to
.. status: implemented

Convert Between pyhf JSON and HS3
=================================

You have a pyhf workspace (HistFactory JSON), for example a likelihood
published on HEPData, and want to evaluate or fit it with pyhs3, or you have
an HS3 workspace of HistFactory channels and want to hand it to pyhf.
:func:`pyhs3.from_hifa` converts pyhf to HS3, :func:`pyhs3.to_hifa` converts
HS3 to pyhf, and :func:`pyhs3.hifa_modifier_settings` gives the
interpolation codes that pyhf's JSON cannot hold. Their API pages list every
naming rule, default, and error; this page shows the tasks.

Convert a pyhf workspace to HS3
-------------------------------

``from_hifa`` takes the parsed JSON, so read a file with ``json.load``. The
workspace below has one channel ``SR`` with two bins, a signal sample scaled
by ``mu``, and a background with a ``normsys`` and a per-bin ``shapesys``:

.. doctest::

    >>> import json
    >>> import pathlib
    >>> import tempfile
    >>> import pyhs3
    >>> spec = {
    ...     "channels": [
    ...         {
    ...             "name": "SR",
    ...             "samples": [
    ...                 {
    ...                     "name": "signal",
    ...                     "data": [5.0, 10.0],
    ...                     "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    ...                 },
    ...                 {
    ...                     "name": "background",
    ...                     "data": [50.0, 60.0],
    ...                     "modifiers": [
    ...                         {
    ...                             "name": "bkg_norm",
    ...                             "type": "normsys",
    ...                             "data": {"hi": 1.1, "lo": 0.9},
    ...                         },
    ...                         {"name": "bkg_unc", "type": "shapesys", "data": [5.0, 6.0]},
    ...                     ],
    ...                 },
    ...             ],
    ...         }
    ...     ],
    ...     "observations": [{"name": "SR", "data": [53.0, 65.0]}],
    ...     "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    ...     "version": "1.0.0",
    ... }
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = pathlib.Path(tmpdir) / "workspace.json"
    ...     with path.open("w") as f:
    ...         json.dump(spec, f)
    ...     with path.open() as f:
    ...         ws = pyhs3.from_hifa(json.load(f))
    ...

Channel ``SR`` becomes the ``histfactory_dist`` ``model_SR`` with observed
data ``obsData_SR``, and measurement ``meas`` becomes a likelihood, an
analysis, a domain and a parameter point, all named ``meas``. A per-bin
modifier such as ``bkg_unc`` gets one parameter per bin, ``bkg_unc_bin_<i>``
for pyhf's ``bkg_unc[i]``:

.. doctest::

    >>> [d.name for d in ws.distributions], [d.name for d in ws.data]
    (['model_SR'], ['obsData_SR'])
    >>> [p.name for p in ws.parameter_points["meas"]]
    ['bkg_norm', 'bkg_unc_bin_0', 'bkg_unc_bin_1', 'mu']

``from_hifa`` also accepts a ``pyhf.Workspace``, which is a ``dict``.

Build, evaluate, and fit the converted model
--------------------------------------------

The result is an ordinary :class:`~pyhs3.Workspace`, so build a model from
the analysis named after the measurement and compile its negative
log-likelihood:

.. doctest::

    >>> model = ws.model(ws.analyses["meas"], progress=False)
    >>> nll = model.compile_nll()
    >>> sorted(nll.parameter_names)
    ['bkg_norm', 'bkg_unc_bin_0', 'bkg_unc_bin_1', 'mu']

From here, :doc:`evaluate_model` covers evaluating the model and
:doc:`minimize_nll` covers fitting it with SciPy or iminuit.

To check the conversion against pyhf, evaluate both at the same point. pyhf
orders its parameters as ``model.config.par_names``, and builds its model with
the interpolation codes from ``hifa_modifier_settings`` (see the next
section):

.. doctest::

    >>> import numpy as np
    >>> import pyhf
    >>> pyhf_ws = pyhf.Workspace(spec)
    >>> pyhf_model = pyhf_ws.model(modifier_settings=pyhs3.hifa_modifier_settings(ws))
    >>> pyhf_model.config.par_names
    ['mu', 'bkg_norm', 'bkg_unc[0]', 'bkg_unc[1]']
    >>> pars = [1.5, 0.5, 0.95, 1.05]
    >>> point = {"mu": 1.5, "bkg_norm": 0.5, "bkg_unc_bin_0": 0.95, "bkg_unc_bin_1": 1.05}
    >>> pyhf_nll = -pyhf_model.logpdf(pars, pyhf_ws.data(pyhf_model))[0]
    >>> bool(np.isclose(nll.value(nll.to_vector(point)), pyhf_nll))
    True

Choose the interpolation codes
------------------------------

pyhf's JSON holds no interpolation codes: you choose them when you build a
pyhf model, with ``pyhf.Workspace.model(modifier_settings=...)``. Pass the
same dictionary to ``from_hifa``, which records each ``normsys`` and
``histosys`` code in the HS3 modifier's ``data.interpolation``. A modifier
type you leave out keeps pyhs3's default (see :func:`pyhs3.from_hifa` for the
defaults and accepted codes):

.. doctest::

    >>> ws_code1 = pyhs3.from_hifa(spec, modifier_settings={"normsys": {"interpcode": "code1"}})
    >>> background = ws_code1.distributions["model_SR"].samples["background"]
    >>> background.modifiers[0].data.interpolation
    'code1'

``hifa_modifier_settings`` reads the codes back, in the form pyhf takes:

.. doctest::

    >>> pyhs3.hifa_modifier_settings(ws_code1)
    {'normsys': {'interpcode': 'code1'}, 'histosys': {'interpcode': 'code4p'}}

It always returns both modifier types. Pass its result whole to pyhf: a
``modifier_settings`` dictionary that leaves a type out makes pyhf fall back
to that type's own default code, which differs from pyhs3's.

Convert an HS3 workspace to pyhf
--------------------------------

``to_hifa`` returns a ``dict`` that follows pyhf's ``workspace.json``
schema. A workspace that ``from_hifa`` produced converts back to the
original pyhf workspace, up to measurement-config entries equal to pyhf's
defaults:

.. doctest::

    >>> pyhf_spec = pyhs3.to_hifa(ws)
    >>> pyhf_spec == spec
    True
    >>> pyhf_model = pyhf.Workspace(pyhf_spec).model(
    ...     modifier_settings=pyhs3.hifa_modifier_settings(ws)
    ... )

The channels come from a likelihood, the observations from that likelihood's
data, and each analysis on it becomes a pyhf measurement. Three keyword
arguments resolve what pyhf cannot hold more than one of:

- ``likelihood`` (a name, or an index into ``ws.likelihoods``) picks the
  likelihood to convert. Without it, ``to_hifa`` converts every likelihood,
  and raises a ``ValueError`` if they pair the channels with different data,
  since pyhf holds one observation per channel.
- ``poi`` picks the parameter of interest, for an analysis that lists none
  or several. pyhf allows one.
- ``parameter_set`` names the parameter point that gives the measurement's
  initial values and ``fixed`` flags, in place of each analysis's ``init``.

Pass the same ``likelihood`` to ``hifa_modifier_settings``, so it reads the
codes of the channels ``to_hifa`` converts.

Convert a workspace exported by ROOT
------------------------------------

Two of these arguments matter for a HistFactory workspace that ROOT exports
to HS3. If the export has a likelihood for the observed data and one for an
Asimov dataset, the two pair the channels with different data, so pick one
with ``likelihood``; without it, ``to_hifa`` raises a ``ValueError`` that
names the likelihoods.

ROOT's analyses also have no ``init``, and ROOT holds ``Lumi`` ``const`` only
in its parameter points. Without ``parameter_set``, ``to_hifa`` takes the first
parameter point in which every parameter outside the analysis's domains is
``const``:

.. code-block:: python

    root_ws = pyhs3.Workspace.load("simplemodel_hs3.json")
    root_spec = pyhs3.to_hifa(root_ws, likelihood="simPdf_obsData")
    # root_spec["measurements"][0]["config"]["parameters"] is
    # [{"name": "Lumi", "fixed": True}]

and logs a warning (logger ``pyhs3.hifa._to_hifa``) naming the parameter point it
chose:

.. code-block:: text

    Analysis 'simPdf_obsData' has no init, and ['Lumi'] are in none of its
    domains; taking initial values and const flags from parameter point
    'default_values', the first in which they are all const. Pass
    parameter_set= to choose another.

Pass ``parameter_set="default_values"`` (or another parameter point) to choose
it yourself and silence the warning. If no parameter point holds those
parameters ``const``, ``to_hifa`` raises a ``ValueError`` asking for
``parameter_set``.

ROOT also names per-bin parameters differently. ``to_hifa`` groups a per-bin
modifier's parameters into one pyhf modifier ``x`` when every name in its
list is ``x`` followed by the same bin suffix, with indices counting up by one:

- ``x_bin_<i>``, as ``from_hifa`` writes them;
- ``x_bin<i>``, as pyhs3 names the parameters of a ``staterror`` that lists
  none: ``staterror_<channel>_bin<i>``, where ``<channel>`` is the
  ``histfactory_dist`` name, so pyhf gets a ``staterror`` named
  ``staterror_<channel>``;
- ``x_<i>``, as ROOT writes them.

Any other parameter list raises a ``ValueError``.

Convert from the command line
-----------------------------

``pyhs3 convert to-hs3`` runs ``from_hifa`` on a file, with ``--interpcode``
(repeatable) for the interpolation codes; ``pyhs3 convert to-pyhf`` runs
``to_hifa`` and ``hifa_modifier_settings``, with ``--likelihood``, ``--poi``
and ``--parameter-set``. Both write to stdout unless you pass ``--output``
(``-o``):

.. code-block:: console

   $ pyhs3 convert to-hs3 workspace.json --interpcode normsys=code1 -o hs3.json
   $ pyhs3 convert to-pyhf hs3.json -o pyhf.json
   modifier_settings: {"normsys": {"interpcode": "code1"}, "histosys": {"interpcode": "code4p"}}

``to-pyhf`` prints the ``modifier_settings`` to stderr so that stdout holds
only the workspace; pass ``--modifier-settings settings.json`` to write them
to a file instead. See :doc:`../reference/cli` for every flag.

Apply a pyhf patch before converting
------------------------------------

A published likelihood often comes as a background-only workspace and one
JSON patch per signal hypothesis. ``from_hifa`` converts a workspace with
its patches already applied, so apply the patch first, for example with
``jsonpatch`` (which pyhf depends on):

.. doctest::

    >>> import jsonpatch
    >>> bkg_only = {
    ...     "channels": [
    ...         {
    ...             "name": "SR",
    ...             "samples": [
    ...                 {
    ...                     "name": "background",
    ...                     "data": [50.0, 60.0],
    ...                     "modifiers": [
    ...                         {
    ...                             "name": "bkg_norm",
    ...                             "type": "normsys",
    ...                             "data": {"hi": 1.1, "lo": 0.9},
    ...                         }
    ...                     ],
    ...                 }
    ...             ],
    ...         }
    ...     ],
    ...     "observations": [{"name": "SR", "data": [53.0, 65.0]}],
    ...     "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    ...     "version": "1.0.0",
    ... }
    >>> signal_patch = [
    ...     {
    ...         "op": "add",
    ...         "path": "/channels/0/samples/0",
    ...         "value": {
    ...             "name": "signal",
    ...             "data": [5.0, 10.0],
    ...             "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    ...         },
    ...     }
    ... ]
    >>> ws_signal = pyhs3.from_hifa(jsonpatch.apply_patch(bkg_only, signal_patch))
    >>> [s.name for s in ws_signal.distributions["model_SR"].samples]
    ['signal', 'background']

A patch that removes a channel can leave that channel's observation behind.
``from_hifa`` ignores an observation whose name matches no channel, as pyhf
does, and logs it at debug level:

.. doctest::

    >>> leftover = {
    ...     **spec,
    ...     "observations": [*spec["observations"], {"name": "VR", "data": [12.0]}],
    ... }
    >>> [d.name for d in pyhs3.from_hifa(leftover).data]
    ['obsData_SR']
