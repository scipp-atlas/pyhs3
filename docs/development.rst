Development Workflow
====================

This guide covers the day-to-day development workflow for pyhs3, including tools, commands, and best practices.

Development Tools
-----------------

We use several tools for development:

- **hatch**: Project management and task running
- **nox**: Automated testing across environments
- **pre-commit**: Git hooks for code quality
- **pytest**: Testing framework
- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **sphinx**: Documentation generation

Pre-commit Hooks
----------------

Setting Up Pre-commit
~~~~~~~~~~~~~~~~~~~~~

Install and configure pre-commit hooks:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

This installs git hooks that automatically run before each commit.

Running Pre-commit
~~~~~~~~~~~~~~~~~~

Run hooks on changed files:

.. code-block:: bash

   pre-commit run

Run hooks on all files:

.. code-block:: bash

   pre-commit run --all-files

What Pre-commit Checks
~~~~~~~~~~~~~~~~~~~~~~

Our pre-commit configuration runs:

- **blacken-docs**: Format code in documentation
- **check-added-large-files**: Prevent large file commits
- **check-case-conflict**: Detect case conflicts
- **check-merge-conflict**: Detect merge conflict markers
- **check-yaml**: Validate YAML files
- **debug-statements**: Detect debug statements
- **end-of-file-fixer**: Ensure files end with newline
- **trailing-whitespace**: Remove trailing whitespace
- **prettier**: Format YAML, Markdown, JSON, CSS
- **ruff-check**: Lint Python code (with auto-fix)
- **ruff-format**: Format Python code
- **mypy**: Type check Python code
- **codespell**: Check for common misspellings
- **shellcheck**: Lint shell scripts
- **validate-pyproject**: Validate pyproject.toml

Pre-commit Workflow
~~~~~~~~~~~~~~~~~~~

**Let pre-commit handle imports and formatting automatically:**

1. Make your code changes
2. Add new imports if needed (don't remove unused imports manually)
3. Commit your changes
4. Pre-commit will automatically:
   - Remove unused imports
   - Sort imports
   - Format code
   - Fix other issues
5. If pre-commit makes changes, re-add files and commit again

If unstaged changes conflict with hook fixes:

.. code-block:: bash

   git add <modified-files>
   git commit -m "your message"

Code Linting
------------

Using ruff
~~~~~~~~~~

ruff is our primary linter and formatter. It's extremely fast and handles most code quality issues.

Check code:

.. code-block:: bash

   ruff check src/pyhs3 tests

Auto-fix issues:

.. code-block:: bash

   ruff check --fix src/pyhs3 tests

Format code:

.. code-block:: bash

   ruff format src/pyhs3 tests

Configured Rules
~~~~~~~~~~~~~~~~

We enable many ruff rules (see ``pyproject.toml``):

- **B**: flake8-bugbear (detect common bugs)
- **I**: isort (import sorting)
- **ARG**: flake8-unused-arguments
- **C4**: flake8-comprehensions
- **EM**: flake8-errmsg
- **PL**: pylint rules
- **PT**: flake8-pytest-style
- **SIM**: flake8-simplify
- **UP**: pyupgrade (modernize Python code)
- And many more...

Type Checking
-------------

Using mypy
~~~~~~~~~~

We enforce strict type checking with mypy:

.. code-block:: bash

   mypy src/pyhs3

Configuration
~~~~~~~~~~~~~

Our mypy configuration (from ``pyproject.toml``):

- **Python version**: 3.10+ (target version)
- **Strict mode**: Enabled
- **Paths**: ``src/`` and ``tests/``
- **Package-specific**: Strict checking for ``pyhs3.*`` modules

Type Hints Requirements
~~~~~~~~~~~~~~~~~~~~~~~

All code must include type hints:

.. code-block:: python

   from __future__ import annotations

   from typing import Any


   def process_data(data: dict[str, Any], normalize: bool = True) -> tuple[float, float]:
       """Process input data and return mean and std."""
       ...

Using pylint
~~~~~~~~~~~~

For deeper static analysis, run pylint:

.. code-block:: bash

   pylint src/pyhs3

Or using nox:

.. code-block:: bash

   nox -s pylint

Working with hatch
------------------

hatch is our project management tool.

Running Tests
~~~~~~~~~~~~~

Run tests:

.. code-block:: bash

   hatch run test

Run doctests:

.. code-block:: bash

   hatch run doctest

Run specific tests:

.. code-block:: bash

   hatch run test tests/test_distributions.py::TestGaussianDistribution

Viewing Environments
~~~~~~~~~~~~~~~~~~~~

See configured environments:

.. code-block:: bash

   hatch env show

Available Scripts
~~~~~~~~~~~~~~~~~

From ``pyproject.toml`` hatch configuration:

- ``test``: Run pytest
- ``doctest``: Run doctests on source code

Working with nox
----------------

nox provides reproducible testing across environments.

Available Sessions
~~~~~~~~~~~~~~~~~~

View all available sessions:

.. code-block:: bash

   nox --list

Default sessions (run automatically with ``nox``):

- ``lint``: Run pre-commit hooks
- ``pylint``: Run pylint
- ``tests``: Run pytest

Optional sessions:

- ``docs``: Build documentation
- ``build``: Build package distributions

Running Sessions
~~~~~~~~~~~~~~~~

Run all default sessions:

.. code-block:: bash

   nox

Run specific session:

.. code-block:: bash

   nox -s lint
   nox -s tests
   nox -s pylint

Run with arguments:

.. code-block:: bash

   nox -s tests -- tests/test_distributions.py -v
   nox -s docs -- --serve

Building Documentation
----------------------

Local Documentation Build
~~~~~~~~~~~~~~~~~~~~~~~~~

Build documentation:

.. code-block:: bash

   nox -s docs

Build and serve with live reload:

.. code-block:: bash

   nox -s docs -- --serve

This will:

- Build the documentation
- Start a local web server
- Open your browser automatically
- Reload when you make changes

Manual Documentation Build
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install documentation dependencies:

.. code-block:: bash

   pip install -e .[docs]

Build with Sphinx:

.. code-block:: bash

   cd docs
   sphinx-build -b html . _build/html

View built documentation:

.. code-block:: bash

   open docs/_build/html/index.html

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~

Documentation sources are in ``docs/``:

.. code-block:: text

   docs/
   ├── conf.py                    # Sphinx configuration
   ├── index.rst                  # Main documentation page
   ├── api.rst                    # API reference
   ├── structure.rst              # HS3 structure guide
   ├── workspace.rst              # Workspace documentation
   ├── model.rst                  # Model documentation
   ├── broadcasting.rst           # Broadcasting guide
   ├── defining_components.rst    # Component definition guide
   ├── contributing.rst           # This guide
   ├── testing.rst                # Testing guide
   ├── development.rst            # Development workflow
   ├── architecture.rst           # Architecture overview
   └── code_of_conduct.rst        # Code of conduct

Testing Workflow
----------------

Quick Testing
~~~~~~~~~~~~~

During development, run tests frequently:

.. code-block:: bash

   pytest -v

Run specific test file:

.. code-block:: bash

   pytest tests/test_distributions.py

Run specific test:

.. code-block:: bash

   pytest tests/test_distributions.py::TestGaussianDistribution::test_pdf_evaluation

Skip slow tests:

.. code-block:: bash

   pytest -m "not slow"

With Coverage
~~~~~~~~~~~~~

Run tests with coverage:

.. code-block:: bash

   pytest --cov=pyhs3

Generate HTML coverage report:

.. code-block:: bash

   pytest --cov=pyhs3 --cov-report=html
   open htmlcov/index.html

See :doc:`testing` for comprehensive testing guide.

Common Development Tasks
------------------------

Adding a New Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create distribution class in ``src/pyhs3/distributions/``
2. Add type hints and docstrings
3. Write unit tests in ``tests/test_distributions.py``
4. Add integration test if needed
5. Update documentation if it's a public API
6. Run tests and linting
7. Commit with semantic message: ``feat: add XYZ distribution``

Adding a New Function
~~~~~~~~~~~~~~~~~~~~~

1. Create function class in ``src/pyhs3/functions/``
2. Add type hints and docstrings
3. Write unit tests in ``tests/test_functions.py``
4. Update documentation
5. Run tests and linting
6. Commit: ``feat: add XYZ function``

Fixing a Bug
~~~~~~~~~~~~

1. Write a failing test that reproduces the bug
2. Run test to confirm failure
3. Fix the bug with minimal changes
4. Run test to confirm fix
5. Check for regression with full test suite
6. Commit: ``fix: correct XYZ behavior``

Updating Documentation
~~~~~~~~~~~~~~~~~~~~~~

1. Edit relevant ``.rst`` files in ``docs/``
2. Build docs locally to preview: ``nox -s docs -- --serve``
3. Check for broken links and formatting
4. Commit: ``docs: update XYZ documentation``

Troubleshooting
---------------

Pre-commit Hook Failures
~~~~~~~~~~~~~~~~~~~~~~~~~

If pre-commit hooks fail:

1. **Read the error message** - it usually explains what's wrong
2. **Let hooks fix issues** - many hooks auto-fix problems
3. **Re-add and re-commit** - after auto-fixes, stage changes again
4. **Check for conflicts** - ensure you don't have uncommitted changes

Common issues:

- **Import sorting**: Let ruff/isort handle it
- **Formatting**: Let ruff format handle it
- **Type errors**: Fix mypy errors before committing
- **Trailing whitespace**: Auto-fixed by hooks

mypy Errors
~~~~~~~~~~~

If mypy reports type errors:

1. Add missing type hints
2. Use ``from __future__ import annotations`` at the top of files
3. Import types from ``typing`` module
4. Use ``Any`` sparingly for truly dynamic types
5. Check mypy documentation: https://mypy.readthedocs.io/

Test Failures
~~~~~~~~~~~~~

If tests fail:

1. **Read the error message and traceback**
2. **Run the specific test** to isolate the issue
3. **Use pytest verbosity**: ``pytest -vv`` for more detail
4. **Check for environment issues** - ensure clean venv
5. **Verify test data** - ensure test files are present

CI Failures
~~~~~~~~~~~

If CI fails but tests pass locally:

1. **Check the CI logs** on GitHub
2. **Look for platform-specific issues** (Windows vs Linux vs macOS)
3. **Verify all dependencies** are in ``pyproject.toml``
4. **Run pre-commit**: ``pre-commit run --all-files``
5. **Test with nox** to match CI environment: ``nox``

Documentation Build Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If documentation fails to build:

1. **Check Sphinx warnings** - treat warnings as errors
2. **Verify reStructuredText syntax** - check for formatting errors
3. **Test locally**: ``nox -s docs``
4. **Check for broken links** in documentation
5. **Validate cross-references** to ensure they resolve

Environment Issues
~~~~~~~~~~~~~~~~~~

If you have dependency or environment issues:

1. **Create a fresh virtual environment**:

   .. code-block:: bash

      rm -rf .venv
      python -m venv .venv
      source .venv/bin/activate
      pip install -e .[dev,test,docs]

2. **Update dependencies**:

   .. code-block:: bash

      pip install --upgrade pip
      pip install -e .[dev,test,docs] --upgrade

3. **Clear caches**:

   .. code-block:: bash

      rm -rf .pytest_cache .mypy_cache .ruff_cache
      rm -rf docs/_build

Performance Tips
----------------

Faster Testing
~~~~~~~~~~~~~~

- Run specific tests instead of entire suite
- Skip slow tests: ``pytest -m "not slow"``
- Use pytest-xdist for parallel testing: ``pytest -n auto``
- Use ``--lf`` to run last failed tests first: ``pytest --lf``

Faster Linting
~~~~~~~~~~~~~~

- Run ruff (very fast) instead of full pre-commit during iteration
- Use ``ruff check --fix`` for quick auto-fixes
- Run full ``pre-commit run --all-files`` before pushing

Editor Integration
------------------

VS Code
~~~~~~~

Recommended extensions:

- Python (Microsoft)
- Pylance
- Ruff
- mypy

Recommended settings:

.. code-block:: json

   {
     "python.linting.enabled": true,
     "python.linting.ruffEnabled": true,
     "python.formatting.provider": "ruff",
     "python.linting.mypyEnabled": true,
     "editor.formatOnSave": true
   }

PyCharm
~~~~~~~

Configure:

1. Enable ruff as external tool
2. Configure mypy as external tool
3. Enable pytest as test runner
4. Set up pre-commit as file watcher

Best Practices
--------------

Daily Workflow
~~~~~~~~~~~~~~

1. **Pull latest changes**: ``git pull origin main``
2. **Create/switch to feature branch**: ``git checkout -b feat/my-feature``
3. **Make small, focused changes**
4. **Write tests as you go**
5. **Run tests frequently**: ``pytest``
6. **Commit often** with semantic messages
7. **Run pre-commit before pushing**: ``pre-commit run --all-files``
8. **Push and create PR** when ready

Code Quality
~~~~~~~~~~~~

- Write simple, readable code over clever code
- Add type hints to all functions
- Write docstrings for public APIs
- Test edge cases and error conditions
- Keep functions small and focused
- Avoid code duplication through refactoring

Git Hygiene
~~~~~~~~~~~

- Commit frequently with logical changes
- Write descriptive commit messages
- Don't commit large files or secrets
- Use ``.gitignore`` to exclude generated files
- Keep branches up to date with main

Getting Help
------------

If you're stuck:

- Check this guide and :doc:`contributing` guide
- Review :doc:`testing` for testing questions
- Read :doc:`architecture` to understand the codebase
- Open a discussion on GitHub
- Check tool documentation:
  - pytest: https://docs.pytest.org/
  - mypy: https://mypy.readthedocs.io/
  - ruff: https://docs.astral.sh/ruff/
  - sphinx: https://www.sphinx-doc.org/
