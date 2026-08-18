.. diataxis: reference
.. status: implemented

Data Reference
================

pyhs3 supports three data types for observed data used in likelihood
evaluation.

Point data
-----------

A single measurement with an optional uncertainty.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Notes
   * - ``name``
     - ``str``
     - required
   * - ``type``
     - ``"point"``
     - required
   * - ``value``
     - ``float``
     - required
   * - ``uncertainty``
     - ``float | None``
     - optional
   * - ``axes``
     - ``list[UnbinnedAxis] | None``
     - optional

.. code-block:: python

   {
       "name": "higgs_mass_measurement",
       "type": "point",
       "value": 125.09,
       "uncertainty": 0.24,
   }

Unbinned data
--------------

Individual data points in a multi-dimensional space.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Notes
   * - ``name``
     - ``str``
     - required
   * - ``type``
     - ``"unbinned"``
     - required
   * - ``entries``
     - ``list[list[float]]``
     - required; one row per event
   * - ``axes``
     - ``list[UnbinnedAxis]``
     - required; one per column of ``entries``
   * - ``weights``
     - ``list[float] | None``
     - optional, one per event
   * - ``entries_uncertainties``
     - ``list[list[float]] | None``
     - optional, same shape as ``entries``

.. code-block:: python

   {
       "name": "particle_tracks",
       "type": "unbinned",
       "entries": [[120.5, 0.8], [125.1, 1.2]],  # [mass, momentum] per event
       "axes": [
           {"name": "mass", "min": 100.0, "max": 150.0},
           {"name": "momentum", "min": 0.0, "max": 5.0},
       ],
       "weights": [0.8, 1.0],
   }

Binned data
------------

Histogram contents with an optional uncertainty.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Notes
   * - ``name``
     - ``str``
     - required
   * - ``type``
     - ``"binned"``
     - required
   * - ``contents``
     - ``list[float]``
     - required; bin contents
   * - ``axes``
     - ``list[BinnedAxis]``
     - required
   * - ``uncertainty``
     - ``GaussianUncertainty | None``
     - optional

A ``BinnedAxis`` is either regular (``min``, ``max``, ``nbins``) or irregular
(``edges``); pyhs3 picks the form based on which keys are present. A
``GaussianUncertainty`` has ``sigma`` (one value per bin) and ``correlation``
(optional, default the literal ``0`` for uncorrelated bins, or a full
correlation matrix).

.. code-block:: python

   {
       "name": "mass_spectrum",
       "type": "binned",
       "contents": [45.0, 67.0, 52.0, 38.0],
       "axes": [{"name": "mass", "edges": [110.0, 120.0, 130.0, 140.0, 150.0]}],
       "uncertainty": {
           "type": "gaussian_uncertainty",
           "sigma": [6.7, 8.2, 7.2, 6.2],
           "correlation": 0,
       },
   }

   # Regular binning
   {
       "name": "pt_spectrum",
       "type": "binned",
       "contents": [100.0, 80.0],
       "axes": [{"name": "pt", "min": 0.0, "max": 100.0, "nbins": 2}],
   }

Domain axes are a separate type from ``BinnedAxis`` above: a domain axis
takes only ``min``/``max`` bounds (or ``const`` — see :doc:`workspace_reference`).
The ``edges``/``nbins`` binning duality applies to data axes only, never to a
domain.
