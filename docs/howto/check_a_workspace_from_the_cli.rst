.. diataxis: how-to
.. status: implemented

Check a Workspace from the Command Line
==========================================

You have an HS3 workspace as a JSON file and want to confirm it's valid, or
see what's in it, before writing any Python. ``pip install pyhs3`` installs
a ``pyhs3`` command covering both with ``validate`` and ``inspect``; see
:doc:`../reference/cli` for their full flag reference.

Both, like every ``pyhs3`` subcommand, read the workspace as a file path, or
from standard input when you pass ``-`` or omit the argument while piping:

.. code-block:: console

   $ pyhs3 validate workspace.json
   $ curl -s https://example.com/workspace.json | pyhs3 validate -

Validate a workspace
----------------------

.. code-block:: console

   $ pyhs3 validate workspace.json
   workspace.json is a valid HS3 workspace.

A workspace that fails schema validation, or references something that
doesn't exist (a likelihood naming an undefined distribution, for example),
reports the problem to stderr and exits with a non-zero status -- usable as
a pass/fail check in a script:

.. code-block:: console

   $ pyhs3 validate broken.json || echo "invalid workspace"

Pass ``--verbose`` to see every schema error instead of just the first 20.

Inspect a workspace's contents
----------------------------------

Once a workspace validates, ``pyhs3 inspect`` summarizes it -- distributions,
domains, data, likelihoods, analyses, parameter points -- without building a
computation graph:

.. code-block:: console

   $ pyhs3 inspect --no-json workspace.json
   HS3 workspace (hs3_version 0.2)
   Distributions
   ┏━━━━━━━┳━━━━━━━━━━━━━━━┓
   ┃ name  ┃ type          ┃
   ┡━━━━━━━╇━━━━━━━━━━━━━━━┩
   │ gauss │ gaussian_dist │
   └───────┴───────────────┘
   ...

Attached to a terminal, ``inspect`` renders this Rich table; piped, or
redirected, it emits JSON instead, autodetecting which one applies (override
either way with ``--json``/``--no-json``):

.. code-block:: console

   $ pyhs3 inspect workspace.json | jq '.distributions'

A large workspace's table view caps each section at 10 rows by default,
with a trailing ``... N more`` line; pass ``--verbose``/``-v`` to see
everything. This cap never applies to the JSON output, which is always
complete.
