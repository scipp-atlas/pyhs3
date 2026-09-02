.. diataxis: reference
.. status: implemented

pyhs3 CLI
=========

``pyhs3`` installs a command-line tool of the same name, wired to
``pyhs3.cli:app``. Run ``pyhs3 --help`` or ``python -m pyhs3 --help`` for the
tool's own summary of the commands below.

.. note::

   Each command's arguments and options below are hand-maintained, verified
   against the actual flag definitions and behavior in ``src/pyhs3/cli/`` and
   its test suite -- not literal ``--help`` transcripts (Typer renders
   ``--help`` through Rich, as boxed panels, which don't reproduce cleanly in
   a documentation page) and not ``sphinx-click`` autodoc. ``sphinx-click``
   (this project's docs dependency for exactly this purpose) requires each
   command to be an instance of ``click.Command``/``click.Group``; the
   installed Typer release builds its commands from Typer's own internal
   command classes instead, so the two aren't currently compatible.

Every command accepts the workspace as a positional argument, a path to an
HS3 JSON file. Passing ``-``, or omitting the argument entirely when standard
input is not a terminal, reads the workspace from stdin instead, so ``cat
workspace.json | pyhs3 validate -`` and ``pyhs3 validate workspace.json`` are
equivalent.

``pyhs3 validate``
------------------

Load and validate an HS3 workspace, reporting success or the errors found.

Arguments
   ``WORKSPACE`` (optional)
      Path to an HS3 workspace JSON file. Use ``-`` or omit to read from
      stdin.

Options
   ``-v``, ``--verbose``
      Show every schema error, not just the first 20.
   ``--help``
      Show the command's help and exit.

On success, prints ``<source> is a valid HS3 workspace.`` and exits with code
``0``. On failure, prints the error(s) to stderr and exits with code ``1``:
a schema error (a field missing or the wrong type) is reported through
``Workspace.format_validation_error``; an unresolved cross-reference (a
likelihood naming a distribution that doesn't exist, for example) is
reported through ``pyhs3.exceptions.WorkspaceValidationError``. Without
``--verbose``, only the first 20 schema errors are shown, with the remainder
summarized by count.

``pyhs3 inspect``
-----------------

Summarize a workspace's distributions, domains, data, likelihoods, analyses,
and parameter points, without building a computation graph.

Arguments
   ``WORKSPACE`` (optional)
      Path to an HS3 workspace JSON file. Use ``-`` or omit to read from
      stdin.

Options
   ``--json`` / ``--no-json``
      Force JSON or table output. Default: autodetect -- JSON when stdout
      isn't an interactive terminal, a table when it is.
   ``-v``, ``--verbose``
      Show every row in table output, without the 10-row-per-section cap.
      Has no effect on JSON output, which is always complete.
   ``--help``
      Show the command's help and exit.

Prints a Rich table when attached to an interactive terminal, JSON
otherwise. Table output caps each section at 10 rows by default (the module
constant ``_MAX_ROWS_DEFAULT``), with a trailing ``… N more (use -v to show
all)`` row; ``--verbose``/``-v`` removes the cap entirely. **JSON output is
never truncated, with or without --verbose/-v** -- it exists for scripts,
which must never silently receive a partial summary.

The JSON payload has one key per section (``metadata``, ``distributions``,
``domains``, ``data``, ``likelihoods``, ``analyses``, ``parameter_points``);
each section is a list of dicts summarizing the corresponding workspace
entries. This is a read-only summary for humans and scripts, not a
round-trippable serialization of the workspace.

``pyhs3 nll``
-------------

Compute the negative log-likelihood at a point in parameter space.

Arguments
   ``WORKSPACE`` (optional)
      Path to an HS3 workspace JSON file. Use ``-`` or omit to read from
      stdin.

Options
   ``-p``, ``--param NAME=VALUE``
      Override a parameter value. Repeatable.
   ``--params-file FILE``
      JSON file mapping parameter names to values. Must exist and be a
      regular file, or the command reports a usage error before running.
   ``--analysis TEXT``
      Name of the analysis (or likelihood) to evaluate. Defaults to the
      sole one.
   ``--help``
      Show the command's help and exit.

The printed value is ``-2 * log_prob`` -- twice the negated joint
log-probability -- not the plain ``-log_prob`` some other tools (including
RooFit's own ``createNLL()``) report under the same name. Compare against a
RooFit/combine-side number with that factor of two in mind.

Parameter values are layered, later layers winning:

#. the workspace's own ``parameter_points`` (so an analysis without an
   ``init`` set is still evaluable),
#. the model's free (non-const) parameter values,
#. ``--params-file``,
#. repeated ``--param`` (a name given both in the file and on the command
   line takes the command-line value).

A ``--param`` naming an observable rather than a free parameter is rejected
with a warning and ignored -- it can never override the workspace's actual
observed data -- and the command still exits ``0``, since this is a
recoverable mismatch, not a fatal error. A ``--param`` naming something that
is neither an observable nor a free parameter of the model (a typo, for
example) is likewise rejected with a warning and ignored, also exiting
``0`` -- check stderr, not just the exit code, when a ``--param`` might not
have applied. A free parameter with no value available anywhere in this
layering -- absent from the workspace's ``parameter_points``, the model's
free parameters, and every override -- is a fatal error naming the missing
parameter, exiting non-zero.

``--analysis`` accepts either an analysis name or a likelihood name
directly. With neither given, the workspace's sole analysis is used, or,
absent any analyses, its sole likelihood; having more than one of either
without ``--analysis`` is a fatal error.

``pyhs3 graph``
---------------

Render a distribution's computation graph to an image file.

Arguments
   ``WORKSPACE`` (optional)
      Path to an HS3 workspace JSON file. Use ``-`` or omit to read from
      stdin.
   ``NAME`` (required)
      Name of the distribution to visualize.

Options
   ``--analysis TEXT``
      Name of the analysis (or likelihood) to build the model from.
      Defaults to the sole one.
   ``--fmt TEXT``
      Output format: ``svg`` (default), ``png``, or ``pdf``.
   ``--outfile TEXT``
      Output file path. Defaults to ``{name}_graph.{fmt}``.
   ``--path TEXT``
      Directory to write the output file in. Ignored if ``--outfile`` is
      set.
   ``--show-id``
      Append each op's toposort index to its label.
   ``--show-dtype``
      Keep dtype annotations in node labels.
   ``--show-shape``
      Keep shape annotations in node labels.
   ``--help``
      Show the command's help and exit.

Builds the model the same way ``nll`` does, then delegates to
:meth:`~pyhs3.model.Model.visualize_graph` -- see there for what the display
flags control. Prints the rendered file's path to stdout. Requires the
``graph`` extra (``pip install 'pyhs3[graph]'``, which installs ``pydot``);
without it, or if the named distribution doesn't exist in the model, the
command reports the problem through a clean error and exits non-zero,
never a raw traceback.

``pyhs3 plot``
--------------

Render one named data entry from a workspace as a matplotlib figure.

Arguments
   ``WORKSPACE`` (optional)
      Path to an HS3 workspace JSON file. Use ``-`` or omit to read from
      stdin.
   ``DATA_NAME`` (required)
      Name of the entry in the workspace's ``data`` list to plot.

Options
   ``--outfile PATH``
      Output file path. Defaults to ``{data-name}.{fmt}`` in the current
      directory.
   ``--fmt TEXT``
      Output image format: ``png`` (default), ``pdf``, or ``svg``.
   ``--help``
      Show the command's help and exit.

Requires the ``plot`` extra (``pip install 'pyhs3[plot]'``), which installs
``hist[plot]`` -- matplotlib **and** ``mplhep``, since plotting a
``hist.Hist`` needs both; matplotlib alone is not enough. If matplotlib
itself is missing, the command reports this through a clean error rather
than a raw traceback; if matplotlib is present but ``mplhep`` specifically
is missing (an unusual partial install), plotting can still raise a raw
``ModuleNotFoundError`` -- installing the ``plot`` extra as documented
avoids this case entirely.

A 1D ``BinnedData``/``UnbinnedData`` entry renders as a filled histogram; a
2D ``BinnedData`` entry renders as a ``pcolormesh`` heatmap. Data with three
or more axes, and any ``PointData`` entry, raise a "not yet supported"
error rather than a misleading plot, and a ``--fmt`` outside
``png``/``pdf``/``svg`` is rejected the same way.
