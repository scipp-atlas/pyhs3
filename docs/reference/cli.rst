.. diataxis: reference
.. status: implemented

pyhs3 CLI
=========

``pyhs3`` installs a command-line tool of the same name, wired to
``pyhs3.cli:app``. Run ``pyhs3 --help`` or ``python -m pyhs3 --help`` for the
tool's own summary of the commands below.

Every command accepts the workspace as a positional argument, a path to an
HS3 JSON file. Passing ``-``, or omitting the argument entirely when standard
input is not a terminal, reads the workspace from stdin instead, so ``cat
workspace.json | pyhs3 validate -`` and ``pyhs3 validate workspace.json`` are
equivalent.

.. No theme option is set below: preferred=text captures each command's help
   as a plain, colorless literal-block transcript, so a theme (which colors
   the console output) has nothing to apply to -- it only takes effect with
   preferred=svg or html.
.. typer:: pyhs3.cli:app
   :prog: pyhs3
   :width: 65
   :preferred: text
   :make-sections:
   :show-nested:
