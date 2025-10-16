Architecture Overview
=====================

This guide provides a high-level overview of the pyhs3 architecture to help contributors understand the codebase structure.

Project Goals
-------------

pyhs3 is a pure-Python implementation of the HS3 (High-level Statistics Serialization Standard) specification that:

- Deserializes HS3 JSON to Python objects
- Builds computational graphs using PyTensor for automatic differentiation
- Evaluates statistical models efficiently with tensor operations
- Supports complex statistical models from particle physics (e.g., HistFactory)

Core Concepts
-------------

The architecture follows the HS3 specification structure:

Workspace
~~~~~~~~~

The ``Workspace`` is the top-level container that holds all components of a statistical analysis:

- **Metadata**: Version and configuration information
- **Distributions**: Probability distributions (PDFs)
- **Functions**: Mathematical functions and transformations
- **Domains**: Parameter bounds and constraints
- **Parameter Points**: Named parameter sets (values, errors, bounds)
- **Data**: Observable data for fits
- **Likelihoods**: Likelihood definitions combining distributions and data
- **Analyses**: Complete analysis specifications

Located in: ``src/pyhs3/core.py``

Model
~~~~~

The ``Model`` is built from a ``Workspace`` and represents a compiled computational graph:

- Converts symbolic expressions to PyTensor ops
- Manages parameter ordering and dependencies
- Provides methods for PDF evaluation (``logpdf``, ``pdf``)
- Handles parameter transformations and constraints

Located in: ``src/pyhs3/core.py``

Distributions
~~~~~~~~~~~~~

Distributions represent probability density functions. The distribution system is hierarchical:

**Base Classes** (``src/pyhs3/distributions/core.py``):

- ``Distribution``: Abstract base for all distributions
- ``Distributions``: Container for managing multiple distributions

**Distribution Types**:

- **Basic** (``basic.py``): Gaussian, Poisson, Exponential, Uniform, etc.
- **Mathematical** (``mathematical.py``): LogNormal, Gamma, Beta, etc.
- **Physics** (``physics.py``): CrystalBall, Landau (particle physics specific)
- **CMS** (``cms.py``): CMS experiment specific distributions
- **Composite** (``composite.py``): Product, Mixture, Sum distributions
- **Histogram** (``histogram.py``): Binned data distributions with interpolation

Each distribution implements:

- Validation of input parameters
- Conversion to PyTensor computational graph
- Support for broadcasting over batch dimensions

Functions
~~~~~~~~~

Functions provide parameterized transformations and calculations:

**Base Classes** (``src/pyhs3/functions/core.py``):

- ``Function``: Abstract base for all functions
- ``Functions``: Container for managing multiple functions

**Function Types** (``src/pyhs3/functions/standard.py``):

- ``GenericFunction``: Custom expressions using SymPy
- ``InterpolationFunction``: Multi-dimensional interpolation
- ``ProductFunction``: Product of multiple functions
- ``SumFunction``: Sum of multiple functions
- ``ProcessNormalizationFunction``: HistFactory normalization (``src/pyhs3/distributions/histfactory/``)

Domains
~~~~~~~

Domains define parameter spaces and constraints:

- ``Axis``: Single parameter with min/max bounds
- ``ProductDomain``: Multi-dimensional parameter space

Located in: ``src/pyhs3/domains.py``

Parameter Points
~~~~~~~~~~~~~~~~

Parameter management system:

- ``ParameterPoint``: Single parameter with value, bounds, error
- ``ParameterSet``: Named collection of parameters
- ``ParameterPoints``: Container for multiple parameter sets

Located in: ``src/pyhs3/parameter_points.py``

Project Structure
-----------------

.. code-block:: text

   pyhs3/
   ├── src/pyhs3/                    # Source code
   │   ├── __init__.py               # Public API (Workspace, Model)
   │   ├── core.py                   # Workspace and Model classes
   │   ├── base.py                   # Base classes and utilities
   │   ├── context.py                # PyTensor context management
   │   ├── networks.py               # Computational graph construction
   │   ├── generic_parse.py          # SymPy expression parsing
   │   ├── exceptions.py             # Custom exceptions
   │   ├── logging.py                # Logging configuration
   │   ├── metadata.py               # Metadata handling
   │   ├── domains.py                # Domain definitions
   │   ├── parameter_points.py       # Parameter management
   │   ├── data.py                   # Data handling
   │   ├── likelihoods.py            # Likelihood definitions
   │   ├── analyses.py               # Analysis specifications
   │   ├── distributions/            # Distribution implementations
   │   │   ├── __init__.py           # Distribution exports
   │   │   ├── core.py               # Base distribution classes
   │   │   ├── basic.py              # Basic distributions
   │   │   ├── mathematical.py       # Mathematical distributions
   │   │   ├── physics.py            # Physics distributions
   │   │   ├── cms.py                # CMS-specific distributions
   │   │   ├── composite.py          # Composite distributions
   │   │   ├── histogram.py          # Histogram distributions
   │   │   └── histfactory/          # HistFactory support
   │   │       ├── __init__.py
   │   │       └── modifiers.py      # HistFactory modifiers
   │   └── functions/                # Function implementations
   │       ├── __init__.py           # Function exports
   │       ├── core.py               # Base function classes
   │       └── standard.py           # Standard functions
   │
   ├── tests/                        # Test suite
   │   ├── test_distributions.py     # Distribution tests
   │   ├── test_functions.py         # Function tests
   │   ├── test_workspace.py         # Workspace tests
   │   ├── test_histfactory.py       # HistFactory tests
   │   ├── test_cross_distribution_dependencies.py
   │   └── test_histfactory/         # Test data
   │
   ├── docs/                         # Documentation
   │   ├── conf.py                   # Sphinx configuration
   │   ├── index.rst                 # Documentation index
   │   ├── api.rst                   # API reference
   │   ├── structure.rst             # HS3 structure
   │   ├── workspace.rst             # Workspace guide
   │   ├── model.rst                 # Model guide
   │   ├── broadcasting.rst          # Broadcasting guide
   │   ├── defining_components.rst   # Component definition
   │   ├── contributing.rst          # Contributor guide
   │   ├── testing.rst               # Testing guide
   │   ├── development.rst           # Development workflow
   │   └── architecture.rst          # This file
   │
   ├── pyproject.toml                # Project configuration
   ├── noxfile.py                    # Nox automation
   ├── .pre-commit-config.yaml       # Pre-commit hooks
   ├── README.rst                    # Project README
   └── LICENSE                       # Apache 2.0 license

Key Dependencies
----------------

Core Dependencies
~~~~~~~~~~~~~~~~~

- **pytensor**: Tensor computation and automatic differentiation
- **numpy**: Numerical computing
- **sympy**: Symbolic mathematics (for expression parsing)
- **pydantic**: Data validation and serialization
- **rustworkx**: Graph algorithms (dependency management)
- **rich**: Terminal formatting

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **mypy**: Static type checking
- **ruff**: Linting and formatting
- **pre-commit**: Git hooks
- **sphinx**: Documentation generation
- **nox**: Task automation

Data Flow
---------

The typical flow through pyhs3:

1. **Deserialization**

   .. code-block:: python

      # HS3 JSON → Pydantic models
      ws = pyhs3.Workspace(**json_data)

2. **Validation**

   - Pydantic validates structure and types
   - Custom validators check constraints
   - Parameter references are resolved

3. **Graph Construction**

   .. code-block:: python

      # Build computational graph
      model = ws.model()

   - Distributions converted to PyTensor ops
   - Functions resolved and linked
   - Dependencies tracked via rustworkx graph
   - Parameters ordered topologically

4. **Evaluation**

   .. code-block:: python

      # Evaluate PDF
      result = model.logpdf("dist_name", **parameters)

   - PyTensor compiles optimized computation
   - Automatic differentiation available
   - Batch evaluation supported via broadcasting

Computational Graph
-------------------

pyhs3 builds a computational graph using PyTensor:

Graph Nodes
~~~~~~~~~~~

- **Parameters**: Input variables
- **Functions**: Transformations and calculations
- **Distributions**: PDF evaluations
- **Constraints**: Parameter bounds and relationships

Graph Construction
~~~~~~~~~~~~~~~~~~

Located in ``src/pyhs3/networks.py``:

1. Parse all components (distributions, functions, parameters)
2. Build dependency graph with rustworkx
3. Topologically sort to determine evaluation order
4. Convert to PyTensor ops maintaining dependencies
5. Compile for efficient evaluation

Context Management
~~~~~~~~~~~~~~~~~~

Located in ``src/pyhs3/context.py``:

- Manages PyTensor compilation modes (FAST_RUN, FAST_COMPILE)
- Handles optimization flags
- Configures numerical stability settings

Expression Parsing
------------------

Generic expressions are handled by ``src/pyhs3/generic_parse.py``:

1. **Parse**: Convert string expression to SymPy
2. **Analyze**: Extract variables and dependencies
3. **Convert**: Transform SymPy to PyTensor ops
4. **Validate**: Check for unsupported operations

This allows users to specify custom mathematical expressions in HS3 JSON.

HistFactory Support
-------------------

HistFactory is a major use case, with dedicated support:

**Modifiers** (``src/pyhs3/distributions/histfactory/modifiers.py``):

- ``histosys``: Histogram systematic variations
- ``normsys``: Normalization systematics
- ``normfactor``: Normalization factors
- ``shapesys``: Shape systematics
- ``staterror``: Statistical uncertainties

**Histogram Distribution** (``src/pyhs3/distributions/histogram.py``):

- Binned data representation
- Interpolation between variations
- Integration with modifiers

Error Handling
--------------

Custom exceptions in ``src/pyhs3/exceptions.py``:

- ``HS3Exception``: Base exception
- ``ExpressionParseError``: Expression parsing failures
- ``ExpressionEvaluationError``: Runtime evaluation errors
- ``UnknownInterpolationCodeError``: Invalid interpolation codes

Extension Points
----------------

To extend pyhs3:

Adding New Distributions
~~~~~~~~~~~~~~~~~~~~~~~~

1. Create class inheriting from ``Distribution`` in appropriate file
2. Implement validation in ``model_validate_extra``
3. Implement ``_to_pytensor`` method
4. Add to ``Distributions.model_validate`` discriminator
5. Write unit tests
6. Update documentation

Example skeleton:

.. code-block:: python

   from __future__ import annotations

   from pyhs3.distributions.core import Distribution


   class MyDistribution(Distribution):
       """My custom distribution."""

       type: Literal["my_dist"] = "my_dist"
       param1: str
       param2: str

       def _to_pytensor(
           self,
           tensors: dict[str, TensorVariable],
           distributions: dict[str, TensorVariable],
           functions: dict[str, TensorVariable],
       ) -> TensorVariable:
           """Convert to PyTensor operation."""
           # Implementation here
           ...

Adding New Functions
~~~~~~~~~~~~~~~~~~~~

1. Create class inheriting from ``Function``
2. Implement ``_to_pytensor`` method
3. Add to ``Functions.model_validate`` discriminator
4. Write unit tests
5. Update documentation

Broadcasting
~~~~~~~~~~~~

See :doc:`broadcasting` for details on batch dimension handling.

Testing Strategy
----------------

See :doc:`testing` for comprehensive testing guide.

Test Categories
~~~~~~~~~~~~~~~

- **Unit tests**: Individual component testing
- **Integration tests**: Component interaction testing
- **End-to-end tests**: Real-world workspace loading and evaluation
- **Doctests**: Documentation examples validation

Performance Considerations
--------------------------

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

- **Graph optimization**: PyTensor optimizes computational graph
- **Batch evaluation**: Use broadcasting for multiple evaluations
- **Compilation modes**: Choose appropriate PyTensor mode
- **Caching**: PyTensor caches compiled functions

Bottlenecks
~~~~~~~~~~~

- First evaluation (compilation overhead)
- Large HistFactory models (many parameters)
- Deep dependency chains (topological sorting)

Best Practices for Contributors
--------------------------------

Code Organization
~~~~~~~~~~~~~~~~~

- Keep related functionality together
- Use clear, descriptive names
- Minimize inter-module dependencies
- Follow existing patterns

Type Hints
~~~~~~~~~~

- Use ``from __future__ import annotations``
- Provide complete type hints
- Use ``typing`` module types appropriately
- Document complex types in docstrings

Documentation
~~~~~~~~~~~~~

- Write numpy-style docstrings
- Include examples in docstrings
- Update relevant documentation files
- Add doctests for public APIs

Testing
~~~~~~~

- Write tests before implementation (TDD)
- Test edge cases and error conditions
- Use parametrized tests for multiple cases
- Keep tests focused and readable

Resources
---------

External Documentation
~~~~~~~~~~~~~~~~~~~~~~

- `HS3 Specification <https://hep-statistics-serialization-standard.github.io/>`_
- `PyTensor Documentation <https://pytensor.readthedocs.io/>`_
- `Pydantic Documentation <https://docs.pydantic.dev/>`_
- `Scientific Python Developer Guide <https://learn.scientific-python.org/development/>`_

Internal Documentation
~~~~~~~~~~~~~~~~~~~~~~

- :doc:`contributing` - Contributor guide
- :doc:`testing` - Testing guide
- :doc:`development` - Development workflow
- :doc:`api` - API reference
- :doc:`structure` - HS3 structure guide

Getting Help
------------

If you need architectural guidance:

- Open a discussion on GitHub
- Review existing code for patterns
- Check HS3 specification for requirements
- Ask specific questions with context
