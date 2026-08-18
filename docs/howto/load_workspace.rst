.. diataxis: how-to
.. status: implemented

Load a Workspace from an HS3 JSON File
========================================

You have an HS3 workspace as a JSON file — exported from RooFit, HistFactory,
or ``combine``, or written by hand — and need it as a ``pyhs3.Workspace``.

Load the file
--------------

Use ``Workspace.load()``, passing the file's path:

.. doctest::

    >>> import json
    >>> import tempfile
    >>> import pyhs3
    >>> workspace_json = {
    ...     "metadata": {"hs3_version": "0.2"},
    ...     "distributions": [
    ...         {
    ...             "name": "gauss",
    ...             "type": "gaussian_dist",
    ...             "x": "x",
    ...             "mean": "mu",
    ...             "sigma": "sigma",
    ...         }
    ...     ],
    ... }
    >>> import pathlib
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = str(pathlib.Path(tmpdir) / "workspace.json")
    ...     with open(path, "w") as f:
    ...         json.dump(workspace_json, f)
    ...     ws = pyhs3.Workspace.load(path)
    ...
    >>> ws.distributions[0].name
    'gauss'

``Workspace.load()`` reads and parses the JSON, then validates it exactly as
``pyhs3.Workspace(**data)`` does. It is a convenience for the common case of
starting from a file instead of an in-memory dictionary.

Handle a workspace that fails to validate
-------------------------------------------

A file that doesn't match the HS3 schema raises
``pyhs3.exceptions.WorkspaceValidationError`` rather than a raw parsing
error. By default, the error message lists the first 20 problems and
summarizes how many more exist; pass ``verbose=True`` to see every one:

.. doctest::

    >>> import pyhs3
    >>> broken_json = {"distributions": [{"name": "gauss", "type": "gaussian_dist"}]}
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     broken_path = str(pathlib.Path(tmpdir) / "broken.json")
    ...     with open(broken_path, "w") as f:
    ...         json.dump(broken_json, f)
    ...     ws = pyhs3.Workspace.load(
    ...         broken_path, verbose=True, suppress_traceback=False
    ...     )  # doctest: +ELLIPSIS
    ...
    Traceback (most recent call last):
        ...
    pyhs3.exceptions.WorkspaceValidationError: ...

The example above is missing ``metadata`` and the distribution's required
``x``/``mean``/``sigma`` parameters, all of which show up in the validation
error.

.. warning::
    ``suppress_traceback`` defaults to ``True``, which sets
    ``sys.tracebacklimit = 0`` for the rest of the process on a validation
    failure, not just for this call. Pass ``suppress_traceback=False``, as
    above, in a long-running process (a notebook, a service) where you don't
    want a single bad workspace to suppress every later traceback.

Select a domain and parameter set when building the model
-------------------------------------------------------------

A loaded workspace is used the same way as one built from a dictionary. If
the file defines named domains and parameter sets, select them by name:

.. code-block:: python

    ws = pyhs3.Workspace.load("my_analysis.json")
    model = ws.model("signal_region", parameter_set="best_fit")

See :meth:`~pyhs3.Workspace.model` for how it selects a domain and
parameter set for each kind of target you can pass it.
