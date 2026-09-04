Testing Guide
=============

This guide covers how to write and run tests for pyhs3.

Testing Philosophy
------------------

We follow Test-Driven Development (TDD) principles:

1. **Write a failing test** that validates the desired functionality
2. **Run the test** to confirm it fails as expected
3. **Write minimal code** to make the test pass
4. **Run the test** to confirm success
5. **Refactor** if needed while keeping tests green

Test Requirements
~~~~~~~~~~~~~~~~~

**NO EXCEPTIONS**: All projects MUST have:

- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests with real data

The only exception is if explicitly authorized to skip tests for a specific change.

Running Tests
-------------

Basic Test Execution
~~~~~~~~~~~~~~~~~~~~

Run all tests using pytest:

.. code-block:: bash

   pytest

Run tests with coverage reporting:

.. code-block:: bash

   pytest --cov=pyhs3

Using hatch
~~~~~~~~~~~

Run tests using hatch:

.. code-block:: bash

   hatch run test

Run doctests:

.. code-block:: bash

   hatch run doctest

Using nox
~~~~~~~~~

Run tests across multiple Python versions:

.. code-block:: bash

   nox -s tests

Run specific tests:

.. code-block:: bash

   nox -s tests -- tests/test_distributions.py

HS3TestSuite Conformance
~~~~~~~~~~~~~~~~~~~~~~~~

The external `HS3TestSuite <https://github.com/hep-statistics-serialization-standard/HS3TestSuite>`_
checks pyhs3 against RooFit-generated reference values stored with the suite.
ROOT is not run during these tests: each tested suite revision is a frozen
comparison with the RooFit values and tolerances committed at that revision.

The suite is intentionally pinned to an exact commit. To run the current pin
locally, check out that commit and point the test harness at it:

.. code-block:: bash

   hs3suite_checkout="$(mktemp -d)"
   git clone https://github.com/hep-statistics-serialization-standard/HS3TestSuite.git \
       "$hs3suite_checkout"
   git -C "$hs3suite_checkout" checkout --detach \
       9d04e321ae6fddd283a35507f14ecf852eb7df61
   export HS3TESTSUITE_ROOT="$hs3suite_checkout"
   export HS3TESTSUITE_REPORT_DIR="$(mktemp -d)"
   export PYTENSOR_FLAGS="base_compiledir=$(mktemp -d)"
   pixi run -e py312 test-hs3suite

The integration test is skipped when ``HS3TESTSUITE_ROOT`` is not set, so the
normal local and platform test matrix does not fetch external test data.
Machine-readable results and a Markdown summary are written to
``HS3TESTSUITE_REPORT_DIR`` when it is set.

The upstream snapshot is recorded in ``tests/hs3testsuite/pin.json`` and
unsupported checks are recorded in ``tests/hs3testsuite/known_failures.json``
as strict expected failures. A supported check that regresses fails normally,
and an expected failure that starts passing is an unexpected pass and also
fails. A known failure that moves to a different stage, such as from numerical
comparison to workspace import, is also a regression. This forces the ledger
to describe the currently pinned suite rather than silently accumulating stale
entries.

Updating the suite pin is a deliberate compatibility review, because the HS3
specification, fixtures, schemas, and RooFit reference values evolve. In a
dedicated pull request:

1. Check out the candidate HS3TestSuite commit explicitly; never use a moving
   branch or tag in CI.
2. Audit changes to schemas, fixture and check identifiers, feature tags,
   tolerances, and RooFit reference vectors.
3. Run the candidate checkout without modifying the tracked pin or ledger:

   .. code-block:: bash

      pixi run -e py312 python tools/audit_hs3testsuite.py \
          --suite-root /path/to/HS3TestSuite \
          --output-dir /tmp/pyhs3-hs3-audit

   This explicit audit mode records the candidate commit and manifest digest in
   its reports without weakening the exact-pin check used by CI. Compare its
   JSON and Markdown reports with the report from the current pin.
4. Review every added or removed check and every pass-to-fail or fail-to-pass
   transition. New expected failures require an explicit reason in the ledger;
   newly supported checks must be removed from it.
5. Update the upstream commit and manifest digest in the pin file, the CI
   checkout ref, this page's local checkout command, and the strict
   expected-failure ledger together. CI must validate that exact snapshot.

Test Organization
-----------------

Test File Structure
~~~~~~~~~~~~~~~~~~~

Tests are organized in the ``tests/`` directory:

.. code-block:: text

   tests/
   ├── test_distributions.py      # Unit tests for distributions
   ├── test_functions.py          # Unit tests for functions
   ├── test_workspace.py          # Integration tests for workspaces
   ├── test_histfactory.py        # Integration tests for HistFactory
   ├── test_realworld.py          # End-to-end tests with real data
   └── test_histfactory/          # Test data directory
       ├── simplemodel_uncorrelated-background_hifa.json
       └── simplemodel_uncorrelated-background_hs3.json

Test Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~

- **Test files**: Must start with ``test_`` (e.g., ``test_distributions.py``)
- **Test classes**: Use descriptive names (e.g., ``TestGaussianDistribution``, ``TestDiHiggsIssue41Workspace``)
- **Test functions**: Must start with ``test_`` (e.g., ``test_gaussian_pdf_evaluation``)

Test Markers
------------

We use pytest markers to categorize tests:

slow
~~~~

Mark tests that take significant time to run:

.. code-block:: python

   import pytest


   @pytest.mark.slow
   def test_large_scale_optimization():
       """Test that runs for several seconds."""
       ...

Run slow tests:

.. code-block:: bash

   pytest --runslow

Skip slow tests (default):

.. code-block:: bash

   pytest

pydot
~~~~~

Mark tests that require the pydot dependency:

.. code-block:: python

   import pytest


   @pytest.mark.pydot
   def test_graph_visualization():
       """Test that requires pydot for graph rendering."""
       ...

Run pydot tests:

.. code-block:: bash

   pytest --runpydot

Writing Tests
-------------

Unit Tests
~~~~~~~~~~

Test individual components in isolation:

.. code-block:: python

   from __future__ import annotations

   import numpy as np
   import pytest
   from pyhs3.distributions import GaussianDist


   class TestGaussianDistribution:
       """Unit tests for Gaussian distribution."""

       def test_gaussian_pdf_at_mean(self):
           """Test that PDF is maximized at the mean."""
           dist = GaussianDist(name="gauss", x="x", mean="mu", sigma="sigma")
           # Test implementation...

       def test_gaussian_invalid_sigma(self):
           """Test that negative sigma raises an error."""
           with pytest.raises(ValueError, match="sigma must be positive"):
               ...
           # Test implementation...

Integration Tests
~~~~~~~~~~~~~~~~~

Test interactions between components:

.. code-block:: python

   from __future__ import annotations

   import pyhs3
   from pyhs3.distributions import GaussianDist
   from pyhs3.parameter_points import ParameterSet, ParameterPoint


   class TestWorkspaceIntegration:
       """Integration tests for workspace operations."""

       def test_workspace_model_creation(self):
           """Test creating a model from workspace."""
           ws = pyhs3.Workspace(
               metadata={"hs3_version": "0.2"},
               distributions=[GaussianDist(name="model", x="x", mean="mu", sigma="sigma")],
               parameter_points=[
                   ParameterSet(
                       name="default",
                       parameters=[
                           ParameterPoint(name="x", value=0.0),
                           ParameterPoint(name="mu", value=0.0),
                           ParameterPoint(name="sigma", value=1.0),
                       ],
                   )
               ],
           )
           model = ws.model()
           assert len(model.parameters) == 3

End-to-End Tests
~~~~~~~~~~~~~~~~

Test with real data and complete workflows:

.. code-block:: python

   from __future__ import annotations

   import json
   import pyhs3


   class TestRealWorldWorkspace:
       """End-to-end tests with real data."""

       def test_histfactory_workspace_loading(self):
           """Test loading a real HistFactory workspace."""
           with open(
               "tests/test_histfactory/simplemodel_uncorrelated-background_hs3.json"
           ) as f:
               workspace_data = json.load(f)

           ws = pyhs3.Workspace(**workspace_data)
           model = ws.model()

           # Validate against expected results
           assert model is not None
           # Run full PDF evaluation...

Test Data
~~~~~~~~~

For integration and end-to-end tests:

- **Use external test data repositories** like ``scikit-hep-testdata`` when possible
- **Store small test files** in ``tests/test_*/`` directories
- **Never commit large files** to the repository

Doctests
--------

We support doctests in both code and documentation:

In Code
~~~~~~~

Add doctests to docstrings:

.. code-block:: python

   def calculate_nll(logpdf: float) -> float:
       """Calculate negative log-likelihood from log PDF.

       Parameters
       ----------
       logpdf : float
           The log probability density function value

       Returns
       -------
       float
           The negative log-likelihood (-2 * logpdf)

       Examples
       --------
       ...

       """
       return -2 * logpdf

In Documentation
~~~~~~~~~~~~~~~~

Add code examples to ``.rst`` files with ``.. doctest::`` directive or in code blocks
that are automatically tested.

Run doctests:

.. code-block:: bash

   pytest --doctest-modules src/pyhs3
   hatch run doctest

Coverage Requirements
---------------------

Test Coverage Goals
~~~~~~~~~~~~~~~~~~~

We aim for high test coverage:

- **Overall coverage**: >85%
- **Core modules**: >90%
- **New code**: 100% coverage expected

Viewing Coverage Reports
~~~~~~~~~~~~~~~~~~~~~~~~

Generate and view HTML coverage report:

.. code-block:: bash

   pytest --cov=pyhs3 --cov-report=html
   open htmlcov/index.html  # or start htmlcov/index.html on Windows

View coverage in terminal:

.. code-block:: bash

   pytest --cov=pyhs3 --cov-report=term-missing

Coverage is automatically reported in CI and uploaded to CodeCov.

Test Output Standards
---------------------

Clean Output
~~~~~~~~~~~~

**Test output must be pristine to pass.**

- No unexpected warnings
- No error messages in logs (unless explicitly tested)
- All expected log messages should be captured and tested

Capturing Expected Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

If your code is expected to log errors, capture them in tests:

.. code-block:: python

   import pytest
   import logging


   def test_expected_error_logging(caplog):
       """Test that expected errors are logged correctly."""
       with caplog.at_level(logging.ERROR):
           # Code that logs an error
           ...

       assert "Expected error message" in caplog.text

Testing Best Practices
----------------------

Fixtures
~~~~~~~~

Use pytest fixtures for common test setup:

.. code-block:: python

   import pytest
   from pyhs3 import Workspace


   @pytest.fixture
   def simple_workspace():
       """Create a simple workspace for testing."""
       return Workspace(
           metadata={"hs3_version": "0.2"},
           distributions=[...],
       )


   def test_with_fixture(simple_workspace):
       """Test using the fixture."""
       model = simple_workspace.model()
       assert model is not None

Parametrized Tests
~~~~~~~~~~~~~~~~~~

Test multiple cases efficiently:

.. code-block:: python

   import pytest


   @pytest.mark.parametrize(
       "value,expected",
       [
           (0.0, 1.83787706),
           (1.0, 2.83787706),
           (-1.0, 2.83787706),
       ],
   )
   def test_nll_values(value, expected):
       """Test NLL calculation for various inputs."""
       # Test implementation...

Mocking
~~~~~~~

**Important**: We DO NOT use mocks in end-to-end tests. Always use real data and real APIs.

For unit tests, mocking is acceptable when testing external dependencies:

.. code-block:: python

   from unittest.mock import Mock, patch


   def test_with_mock():
       """Test with mocked external dependency."""
       with patch("external_module.function") as mock_func:
           mock_func.return_value = 42
           # Test implementation...

Continuous Integration
----------------------

Our CI pipeline runs:

1. **Linting** via pre-commit hooks
2. **Type checking** with mypy
3. **Unit tests** across multiple Python versions
4. **Coverage reporting** to CodeCov
5. **Documentation build** to ensure docs compile

All checks must pass before a PR can be merged.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Tests pass locally but fail in CI**
   - Ensure you've run ``pre-commit run --all-files``
   - Check for platform-specific issues
   - Verify all dependencies are specified

**Coverage too low**
   - Add tests for uncovered code paths
   - Use ``pytest --cov=pyhs3 --cov-report=term-missing`` to see what's missing

**Slow test suite**
   - Mark slow tests with ``@pytest.mark.slow``
   - Run only fast tests during development: ``pytest -m "not slow"``

**Flaky tests**
   - Investigate timing issues or race conditions
   - Use fixed random seeds for reproducibility
   - Avoid tests that depend on external services

Getting Help
~~~~~~~~~~~~

If you're stuck:

- Check CI logs for detailed error messages
- Review similar tests in the codebase
- Ask in GitHub discussions
- Read pytest documentation: https://docs.pytest.org/
