.. diataxis: how-to
.. status: implemented

Evaluate and Visualize a Model from the Command Line
=======================================================

You have a valid HS3 workspace (see :doc:`check_a_workspace_from_the_cli` to
confirm that first) and want to compute its likelihood or see its model
without writing Python. ``pyhs3``'s ``nll``, ``graph``, and ``plot``
subcommands cover this; see :doc:`../reference/cli` for their full flag
reference. Like every ``pyhs3`` subcommand, each reads the workspace as a
file path, or from standard input when you pass ``-`` or omit the argument
while piping.

Compute a negative log-likelihood
-------------------------------------

``pyhs3 nll`` builds the model and evaluates ``-2 * log_prob`` at a point in
parameter space:

.. code-block:: console

   $ pyhs3 nll workspace.json
   12.763631199228033

With no parameter values given, it uses the workspace's own nominal values.
Override one at a time with ``--param name=value`` (repeatable):

.. code-block:: console

   $ pyhs3 nll workspace.json --param mu=1.0
   6.763631199228032

or many at once from a JSON file, with any ``--param`` still taking
precedence over the file for names given both ways:

.. code-block:: console

   $ cat params.json
   {"mu": 1.0, "sigma": 2.0}
   $ pyhs3 nll workspace.json --params-file params.json

If the workspace defines more than one analysis (or, absent any analyses,
more than one likelihood), name which one to evaluate with ``--analysis``.

Render a distribution's computation graph
---------------------------------------------

``pyhs3 graph`` draws a distribution's PyTensor computation graph -- the
same figure :meth:`~pyhs3.model.Model.visualize_graph` produces from Python
-- useful when a model isn't behaving as expected and you want to see the
graph pyhs3 actually built. It needs the ``graph`` extra:

.. code-block:: console

   $ pip install 'pyhs3[graph]'
   $ pyhs3 graph workspace.json gauss
   gauss_graph.svg

Choose the output format and location with ``--fmt``/``--outfile``:

.. code-block:: console

   $ pyhs3 graph workspace.json gauss --fmt png --outfile /tmp/gauss.png

Plot a workspace's data
--------------------------

``pyhs3 plot`` renders one named entry from the workspace's ``data`` list --
a filled histogram for 1D data, a heatmap for 2D binned data. It needs the
``plot`` extra:

.. code-block:: console

   $ pip install 'pyhs3[plot]'
   $ pyhs3 plot workspace.json obs
   obs.png

Data with three or more axes, and single-value ``PointData`` entries, aren't
supported yet: the command reports this rather than producing a misleading
plot.
