Development Workflow
====================

This guide covers the day-to-day development workflow for pyhs3, including tools, commands, and best practices.

Development Tools
-----------------

We use several tools for development:

- **pixi**: Package management, environment management, and task running
- **pre-commit**: Git hooks for code quality
- **pytest**: Testing framework
- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **sphinx**: Documentation generation

Pre-commit Hooks
----------------

Setting Up Pre-commit
~~~~~~~~~~~~~~~~~~~~~

After running ``pixi install``, configure pre-commit hooks:

.. code-block:: bash

   pixi run pre-commit-install

This installs git hooks that automatically run before each commit. All tools (ruff, mypy, etc.) are managed by pixi, so you don't need to install them separately.

Running Pre-commit
~~~~~~~~~~~~~~~~~~

Run hooks on changed files:

.. code-block:: bash

   pixi run pre-commit

Run hooks on all files:

.. code-block:: bash

   pixi run pre-commit --all-files

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

The easiest way to run ruff is through pre-commit:

.. code-block:: bash

   pixi run pre-commit

For manual use, ruff is available in the pixi environment. You can run it directly after ``pixi install``:

.. code-block:: bash

   # Check code
   ruff check src/pyhs3 tests

   # Auto-fix issues
   ruff check --fix src/pyhs3 tests

   # Format code
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

We enforce strict type checking with mypy. It runs automatically with pre-commit, or you can run it manually after ``pixi install``:

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

.. code-block:: pycon

   >>> from __future__ import annotations
   >>> from typing import Any
   >>> def process_data(data: dict[str, Any], normalize: bool = True) -> tuple[float, float]:
   ...     """Process input data and return mean and std."""
   ...     values = list(data.values())
   ...     if not values:
   ...         return 0.0, 0.0
   ...     mean = sum(values) / len(values) if isinstance(values[0], (int, float)) else 0.0
   ...     variance = (
   ...         sum((x - mean) ** 2 for x in values) / len(values)
   ...         if isinstance(values[0], (int, float))
   ...         else 0.0
   ...     )
   ...     std = variance**0.5
   ...     return (mean, std) if normalize else (mean * 100, std * 100)
   ...
   >>> # Example usage
   >>> process_data({"a": 1.0, "b": 2.0, "c": 3.0})
   (2.0, 0.816496580927726)
   >>> process_data({"a": 1.0, "b": 2.0, "c": 3.0}, normalize=False)
   (200.0, 81.6496580927726)

Using pylint
~~~~~~~~~~~~

For deeper static analysis, run pylint:

.. code-block:: bash

   pylint src/pyhs3

Or using nox:

.. code-block:: bash

   nox -s pylint

Working with pixi
----------------

pixi provides reproducible development environments and task automation. Tasks automatically use the correct environment, so you don't need to specify environments when running tasks.

Installing pixi
~~~~~~~~~~~~~~~

Follow installation instructions at https://pixi.sh/latest/

Quick Start
~~~~~~~~~~~

After cloning the repository:

.. code-block:: bash

   pixi install  # Install all dependencies and pyhs3 in editable mode

This sets up all environments (test, docs, dev) and installs pyhs3 automatically.

Available Tasks
~~~~~~~~~~~~~~~

View all available tasks:

.. code-block:: bash

   pixi task list

**Testing tasks:**

- ``test``: Run quick tests (skip slow and pydot)
- ``test-cov``: Run quick tests with coverage
- ``test-slow``: Run tests including slow tests
- ``test-pydot``: Run tests including pydot tests
- ``test-all``: Run all tests with coverage (slow + pydot)
- ``test-docstrings``: Run doctests in source modules
- ``test-docs``: Run doctests in README and docs/
- ``check-docstrings``: Check docstring style

**Documentation tasks:**

- ``docs-build``: Build documentation (static)
- ``docs-linkcheck``: Check documentation for broken links
- ``docs-serve``: Build and serve documentation
- ``docs-watch``: Build and serve documentation with live reload (recommended for development)
- ``docs-clean``: Clean documentation build artifacts
- ``docs-api``: Regenerate API documentation

**Linting tasks:**

- ``pre-commit``: Run pre-commit hooks on all files
- ``pylint``: Run PyLint on pyhs3 package
- ``lint``: Run all linting (pre-commit + pylint)

**Development tasks:**

- ``pre-commit-install``: Install pre-commit git hooks
- ``pre-commit-update``: Update pre-commit hook versions
- ``build-clean``: Clean build artifacts
- ``build``: Build SDist and wheel distributions

**Composite tasks:**

- ``check``: Quick check (lint + basic tests)
- ``check-all``: Comprehensive check (lint + all tests)

Running Tasks
~~~~~~~~~~~~~

Run tasks using ``pixi run``:

.. code-block:: bash

   # Testing
   pixi run test
   pixi run test-cov
   pixi run test-all

   # Linting
   pixi run lint
   pixi run pre-commit
   pixi run pylint

   # Documentation
   pixi run docs-watch
   pixi run docs-build

   # Quick checks
   pixi run check
   pixi run check-all

Pass additional arguments directly:

.. code-block:: bash

   pixi run test tests/test_distributions.py -v
   pixi run pylint --output-format=json

Environments
~~~~~~~~~~~~

pixi manages three environments automatically:

- ``test``: Testing environment (includes ROOT, cms-combine, jax, pytest)
- ``docs``: Documentation environment (includes Sphinx and extensions)
- ``dev``: Development environment (combines test + docs + dev tools)

You don't need to manually activate or switch environments - tasks automatically use the correct environment.

Testing with Specific Python Versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run tests with a specific Python version, use the environment name:

.. code-block:: bash

   # Python 3.10
   pixi run -e py310 test

   # Python 3.11
   pixi run -e py311 test

   # Python 3.12
   pixi run -e py312 test

   # Python 3.13
   pixi run -e py313 test

   # Python 3.14
   pixi run -e py314 test

The default ``test`` task runs quick tests (skips slow and pydot tests). For comprehensive testing:

.. code-block:: bash

   # Run all tests (including slow and pydot)
   pixi run -e py311 test-all

   # Run with coverage
   pixi run -e py311 test-cov

Building Documentation
----------------------

Local Documentation Build
~~~~~~~~~~~~~~~~~~~~~~~~~

Build documentation:

.. code-block:: bash

   pixi run docs-build

Build and serve with live reload:

.. code-block:: bash

   pixi run docs-watch

This will:

- Build the documentation
- Start a local web server
- Open your browser automatically
- Reload when you make changes

Regenerate API Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regenerate API docs from source code:

.. code-block:: bash

   pixi run docs-api

Clean Documentation Build
~~~~~~~~~~~~~~~~~~~~~~~~~~

Clean documentation build artifacts:

.. code-block:: bash

   pixi run docs-clean

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

During development, run tests frequently using pixi tasks:

.. code-block:: bash

   # Quick tests (skip slow and pydot)
   pixi run test

   # All tests with coverage
   pixi run test-all

   # Tests with coverage report
   pixi run test-cov

For specific tests, pass additional arguments directly:

.. code-block:: bash

   # Run specific test file
   pixi run test tests/test_distributions.py -v

   # Run specific test
   pixi run test tests/test_distributions.py::TestGaussianDistribution::test_pdf_evaluation

Advanced: Direct pytest Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``pixi install``, pytest is available in the environment for direct use:

.. code-block:: bash

   pytest -v
   pytest tests/test_distributions.py
   pytest -m "not slow"

With Coverage
~~~~~~~~~~~~~

Use pixi tasks for coverage:

.. code-block:: bash

   # Quick tests with coverage
   pixi run test-cov

   # All tests with coverage
   pixi run test-all

Or run pytest directly with coverage options:

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
6. Run tests and linting:

   .. code-block:: bash

      pixi run check

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
2. Build docs locally to preview: ``pixi run docs-serve``
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
3. **Verify all dependencies** are in ``pyproject.toml`` and ``pixi.toml``
4. **Run pre-commit**: ``pixi run pre-commit``
5. **Test with pixi** to match CI environment: ``pixi run check-all``

Documentation Build Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If documentation fails to build:

1. **Check Sphinx warnings** - treat warnings as errors
2. **Verify reStructuredText syntax** - check for formatting errors
3. **Test locally**: ``pixi run docs-build``
4. **Check for broken links** in documentation
5. **Validate cross-references** to ensure they resolve

Environment Issues
~~~~~~~~~~~~~~~~~~

If you have dependency or environment issues:

1. **Reinstall with pixi**:

   .. code-block:: bash

      pixi clean
      pixi install

2. **Update pixi**:

   .. code-block:: bash

      pixi self-update

3. **Clear caches**:

   .. code-block:: bash

      rm -rf .pytest_cache .mypy_cache .ruff_cache
      rm -rf docs/_build
      pixi clean cache

Performance Tips
----------------

Faster Testing
~~~~~~~~~~~~~~

- Use ``pixi run test`` for quick tests (skips slow and pydot tests)
- Run specific tests: ``pixi run test tests/test_specific.py``
- After ``pixi install``, pytest is available for direct use when you need fine-grained control
- Use pytest-xdist for parallel testing: ``pytest -n auto``
- Use ``--lf`` to run last failed tests first: ``pytest --lf``

Faster Linting
~~~~~~~~~~~~~~

- After ``pixi install``, ruff is available directly for very fast iteration
- Use ``ruff check --fix`` for quick auto-fixes during development
- Run full ``pixi run pre-commit`` before pushing to ensure all checks pass

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
5. **Run tests frequently**: ``pixi run test``
6. **Commit often** with semantic messages
7. **Run checks before pushing**: ``pixi run check``
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
