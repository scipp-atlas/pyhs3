.. diataxis: reference
.. status: implemented

Workspace Reference
====================

:class:`~pyhs3.Workspace` is the top-level container for an HS3
specification. It accepts the fields below as keyword arguments, or as an
equivalent JSON file loaded via :meth:`~pyhs3.Workspace.load`.

Metadata
--------

- **Name:** ``metadata``
- **Type:** :class:`~pyhs3.metadata.Metadata`
- **Default:** required
- **Constraints:** must carry ``hs3_version``. ``authors`` (``list[str]``),
  ``publications`` (``list[str]``), and ``description`` (``str``) are
  optional. ``packages`` is an optional ``list`` of
  :class:`~pyhs3.metadata.PackageInfo` records (each a ``{"name": ...,
  "version": ...}`` pair) — not a mapping keyed by package name.

Distributions
-------------

- **Name:** ``distributions``
- **Type:** list of distribution specs
- **Default:** empty
- **Constraints:** each entry needs ``name`` and ``type``; the remaining
  fields depend on ``type``. See :mod:`pyhs3.distributions` for every
  supported ``type``.

Functions
---------

- **Name:** ``functions``
- **Type:** list of function specs
- **Default:** empty
- **Constraints:** each entry needs ``name`` and ``type``; the remaining
  fields depend on ``type``. See :mod:`pyhs3.functions` for every supported
  ``type``.

Domains
-------

- **Name:** ``domains``
- **Type:** list of domain specs
- **Default:** empty
- **Constraints:** each domain has ``name``, ``type``, and an ``axes`` list.
  Each axis is one of two types: a :class:`~pyhs3.axes.DomainCoordinateAxis`
  (optional ``min``/``max`` bounds, unbounded when omitted) or a
  :class:`~pyhs3.axes.ConstantAxis` (``const: true``). A parameter named by a
  ``ConstantAxis`` is currently treated as unbounded, the same as a parameter
  the domain doesn't name at all — pyhs3 does not yet use this axis type to
  fix a parameter's value at model-construction time. To fix a parameter's
  value, set ``const: true`` on its entry in ``parameter_points`` instead; see
  :doc:`model_reference`. See :doc:`data_reference` for the separate axis
  types used by data (which additionally support ``edges``/``nbins``
  binning — domain axes never do).

Parameter Points
----------------

- **Name:** ``parameter_points``
- **Type:** list of named parameter sets
- **Default:** empty
- **Constraints:** each set has ``name`` and a ``parameters`` list of
  ``{"name": ..., "value": ...}`` entries. Optional — parameters absent from
  every parameter set are still discovered from the distributions and
  functions that reference them, and default to unbounded scalars.

Data
----

- **Name:** ``data``
- **Type:** list of data specs (``point``, ``unbinned``, or ``binned``)
- **Default:** empty
- **Constraints:** see :doc:`data_reference` for the exact fields each data
  type accepts.

Likelihoods
-----------

- **Name:** ``likelihoods``
- **Type:** list of likelihood specs
- **Default:** empty
- **Constraints:** each likelihood has ``name``, a ``distributions`` list, a
  ``data`` list of the same length, and an optional ``aux_distributions``
  list of constraint terms.

Analyses
--------

- **Name:** ``analyses``
- **Type:** list of analysis specs
- **Default:** empty
- **Constraints:** each analysis has ``name``, a ``likelihood`` name, a
  ``domains`` list, and optional ``parameters_of_interest``, ``init``
  (a parameter-point name), and ``prior``.

Accessing a collection
-----------------------

``workspace.distributions``, ``.functions``, ``.domains``,
``.parameter_points``, ``.data``, ``.likelihoods``, and ``.analyses`` all
share the same collection interface:

- **``len(collection)``** — number of entries.
- **``collection[key]``** — entry by integer index or by ``name`` string;
  raises ``KeyError``/``IndexError`` if absent.
- **``collection.get(key)``** — same lookup, returns ``None`` instead of
  raising if absent.
- **``key in collection``** — ``True`` if ``key`` names an entry.
- **iteration** — ``for entry in collection:`` yields entries in declaration
  order.

.. doctest:: workspace-reference

   >>> import pyhs3
   >>> workspace = pyhs3.Workspace(
   ...     metadata={"hs3_version": "0.2"},
   ...     distributions=[
   ...         {
   ...             "name": "signal",
   ...             "type": "gaussian_dist",
   ...             "x": "obs",
   ...             "mean": "mu",
   ...             "sigma": "sigma",
   ...         }
   ...     ],
   ... )
   >>> len(workspace.distributions)
   1
   >>> "signal" in workspace.distributions
   True
   >>> workspace.distributions.get("missing") is None
   True

See :doc:`model_reference` for how ``workspace.model(target, ...)`` builds a
:class:`~pyhs3.Model` from these components.
