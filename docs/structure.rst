PyHS3 Structure and Components
==============================

This guide explains the overall structure of PyHS3 and how its main components work together to create statistical models.

Overview
--------

PyHS3 follows a hierarchical structure:

.. mermaid::
   :config: {"theme": "forest", "darkMode": "true"}

   %%{
     init: {
       'theme': 'forest',
       'themeVariables': {
         'primaryColor': '#fefefe',
         'lineColor': '#aaa'
       }
     }
   }%%

   flowchart TD
       A[HS3 JSON File] <--> B["**Workspace** <br>container"]
       B --> C["**Model** <br>computational representation"]
       C --> D["PDF Evaluation / Analysis"]

The data flows from JSON specification → Workspace → Model → Results, with each level adding computational capabilities.

Core Components
---------------

Distributions
~~~~~~~~~~~~~

**Distributions** are probability density functions that form the core of your statistical model. PyHS3 supports all :hs3:label:`HS3 distribution types <hs3.sec:distributions>`:

.. code-block:: python

   {
       "name": "signal",
       "type": "gaussian_dist",
       "x": "observable",  # What variable this distribution describes
       "mean": "mu",  # Parameter dependencies
       "sigma": "sigma",
   }

For available distribution types, see the :doc:`api` documentation under :mod:`pyhs3.distributions`.

Functions
~~~~~~~~~

**Functions** compute derived parameters from other parameters following the :hs3:label:`HS3 functions specification <hs3.sec:functions>`:

.. code-block:: python

   {"name": "total_rate", "type": "sum", "summands": ["signal_rate", "background_rate"]}

For available function types, see the :doc:`api` documentation under :mod:`pyhs3.functions`.

Domains
~~~~~~~

**Domains** define parameter spaces and constraints:

.. code-block:: python

   {
       "name": "physical_region",
       "type": "product_domain",
       "axes": [
           {"name": "mass", "min": 100.0, "max": 150.0},
           {"name": "rate", "min": 0.0, "max": 1000.0},
       ],
   }

Domains serve multiple purposes:

- Define valid parameter ranges for optimization
- Apply bounds to automatically discovered parameters
- Specify integration regions
- Validate parameter values

Parameter Points
~~~~~~~~~~~~~~~~

**Parameter Points** define named sets of parameter values:

.. code-block:: python

   {
       "name": "best_fit",
       "parameters": [
           {"name": "mass", "value": 125.0},
           {"name": "rate", "value": 100.0},
           {"name": "background", "value": 1000.0},
       ],
   }

Parameter points are optional - parameters can be automatically discovered from distributions and functions.

Likelihoods
~~~~~~~~~~~

**Likelihoods** map distributions to observed data to create likelihood functions:

.. code-block:: python

   {
       "name": "signal_likelihood",
       "distributions": ["signal_model", "background_model"],
       "data": ["observed_data", "sideband_data"],
       "aux_distributions": ["nuisance_constraint"],  # Optional regularization
   }

Likelihoods represent the mathematical construct: :math:`\mathcal{L}(\theta) = \prod_i \text{PDF}(m_i(\theta_i), x_i)`

Where:
- ``distributions`` are the parameterized models :math:`m_i(\theta_i)`
- ``data`` are the observations :math:`x_i`
- ``aux_distributions`` are penalty terms for regularization

Analyses
~~~~~~~~

**Analyses** define complete analysis configurations linking likelihoods with parameter domains:

.. code-block:: python

   {
       "name": "higgs_analysis",
       "likelihood": "signal_likelihood",
       "domains": ["nuisance_parameters", "parameters_of_interest"],
       "parameters_of_interest": ["mu_higgs"],
       "init": "starting_values",  # Optional initial parameter point
       "prior": "bayesian_prior",  # Optional for Bayesian analyses
   }

Analyses specify:
- Which likelihood to use for inference
- Parameter domains for optimization/integration
- Parameters of primary interest for the analysis
- Initial values and priors (both optional)

Metadata
~~~~~~~~

**Metadata** provides version information and documentation:

.. code-block:: python

   {
       "hs3_version": "0.2",
       "authors": ["Analysis Team"],
       "description": "H→γγ signal extraction model",
       "packages": {"pyhs3": "0.2.0"},
   }

How Components Interact
-----------------------

Dependency Resolution
~~~~~~~~~~~~~~~~~~~~~

PyHS3 automatically builds a dependency graph to determine evaluation order:

.. mermaid::
   :config: {"theme": "forest", "darkMode": "true"}

   %%{
     init: {
       'theme': 'forest',
       'themeVariables': {
         'primaryColor': '#fefefe',
         'lineColor': '#aaa'
       }
     }
   }%%

   flowchart TD
       Parameters["Parameters"]
       Functions["Functions"]
       Distributions["Distributions"]
       Parameters --> Functions
       Functions --> Functions
       Functions --> Distributions
       Distributions --> Parameters
       Distributions --> Functions
       Distributions --> Distributions

For example:

.. code-block:: python

   # This creates the dependency: signal_events → total_events → signal_fraction
   {
       "functions": [
           {
               "name": "total_events",
               "type": "sum",
               "summands": ["signal_events", "background_events"],
           },
           {
               "name": "signal_fraction",
               "type": "generic_function",
               "expression": "signal_events / total_events",  # Depends on function above
           },
       ],
       "distributions": [
           {
               "name": "measured_fraction",
               "type": "gaussian_dist",
               "x": "observed_fraction",
               "mean": "signal_fraction",  # Depends on function
               "sigma": "uncertainty",
           }
       ],
   }

Parameter Discovery
~~~~~~~~~~~~~~~~~~~

When parameters are not explicitly defined in ``parameter_points``, PyHS3 discovers them:

1. **Scan distributions and functions** for parameter references
2. **Create tensor variables** for each discovered parameter
3. **Apply domain bounds** if available
4. **Use default scalar type** unless specified otherwise

.. code-block:: python

   # This will discover: obs, mu, sigma automatically
   {
       "distributions": [
           {
               "name": "model",
               "type": "gaussian_dist",
               "x": "obs",
               "mean": "mu",
               "sigma": "sigma",
           }
       ]
       # No parameter_points needed!
   }

Tensor Types and Bounds
~~~~~~~~~~~~~~~~~~~~~~~

Parameters become **bounded tensor variables**:

- **Scalar tensors** (default): Single values
- **Vector tensors**: Arrays of values
- **Bounded**: Constrained by domain specifications

.. code-block:: python

   # Domain bounds automatically applied:
   {"name": "sigma", "min": 0.1, "max": 5.0}  # σ ∈ [0.1, 5.0]

   # Results in bounded tensor variable:
   sigma_tensor = clip(raw_sigma, 0.1, 5.0)

Compilation and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Models compile into optimized computational graphs:

1. **Build dependency graph** from all components
2. **Topological sort** to determine evaluation order
3. **Compile with PyTensor** for efficient evaluation
4. **Cache compiled functions** for reuse

Data Flow Example
-----------------

Here's how data flows through a complete PyHS3 model:

.. doctest::

   >>> # 1. JSON/Dict specification
   >>> model_spec = {
   ...     "metadata": {"hs3_version": "0.2"},
   ...     "distributions": [
   ...         {
   ...             "name": "signal",
   ...             "type": "gaussian_dist",
   ...             "x": "mass",
   ...             "mean": "higgs_mass",
   ...             "sigma": "resolution",
   ...         },
   ...         {
   ...             "name": "background",
   ...             "type": "generic_dist",
   ...             "x": "mass",
   ...             "expression": "exp(-mass/slope)",
   ...         },
   ...     ],
   ...     "functions": [
   ...         {
   ...             "name": "total_yield",
   ...             "type": "sum",
   ...             "summands": ["signal_yield", "background_yield"],
   ...         }
   ...     ],
   ...     "parameter_points": [
   ...         {
   ...             "name": "physics",
   ...             "parameters": [
   ...                 {"name": "higgs_mass", "value": 125.0},
   ...                 {"name": "resolution", "value": 2.5},
   ...                 {"name": "signal_yield", "value": 100.0},
   ...                 {"name": "background_yield", "value": 1000.0},
   ...                 {"name": "slope", "value": 50.0},
   ...             ],
   ...         }
   ...     ],
   ...     "domains": [
   ...         {
   ...             "name": "search_region",
   ...             "type": "product_domain",
   ...             "axes": [
   ...                 {"name": "mass", "min": 110.0, "max": 140.0},
   ...                 {"name": "higgs_mass", "min": 120.0, "max": 130.0},
   ...             ],
   ...         }
   ...     ],
   ...     "data": [
   ...         {
   ...             "name": "observed_mass_data",
   ...             "type": "binned",
   ...             "contents": [10, 20, 15],
   ...             "axes": [{"name": "mass", "edges": [110.0, 120.0, 130.0, 140.0]}],
   ...         }
   ...     ],
   ...     "likelihoods": [
   ...         {
   ...             "name": "higgs_likelihood",
   ...             "distributions": ["signal", "background"],
   ...             "data": ["observed_mass_data", "observed_mass_data"],
   ...         }
   ...     ],
   ...     "analyses": [
   ...         {
   ...             "name": "higgs_discovery",
   ...             "likelihood": "higgs_likelihood",
   ...             "domains": ["search_region"],
   ...             "parameters_of_interest": ["higgs_mass", "signal_yield"],
   ...         }
   ...     ],
   ... }
   >>> # 2. Create Workspace (validates and organizes)
   >>> import pyhs3
   >>> import numpy as np
   >>> ws = pyhs3.Workspace(**model_spec)
   >>> # 3. Create Model (builds computational graph)
   >>> model = ws.model(domain="search_region", parameter_set="physics")
   <BLANKLINE>
   >>> # 4. Evaluate (compile and compute)
   >>> signal_pdf = model.pdf(
   ...     "signal", mass=np.array(125.0), higgs_mass=np.array(125.0), resolution=np.array(2.5)
   ... )
   >>> background_pdf = model.pdf("background", mass=np.array(125.0), slope=np.array(50.0))

Common Patterns
---------------

Signal + Background Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "distributions": [
           {
               "name": "signal",
               "type": "gaussian_dist",
               "x": "mass",
               "mean": "mu",
               "sigma": "sigma",
           },
           {
               "name": "background",
               "type": "generic_dist",
               "x": "mass",
               "expression": "exp(-x)",
           },
       ],
       "functions": [
           {
               "name": "total_events",
               "type": "sum",
               "summands": ["signal_events", "background_events"],
           }
       ],
   }

Systematic Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "functions": [
           {
               "name": "corrected_rate",
               "type": "product",
               "factors": ["nominal_rate", "systematic_factor"],
           }
       ],
       "distributions": [
           {
               "name": "systematic_constraint",
               "type": "gaussian_dist",
               "x": "systematic_factor",
               "mean": "1.0",
               "sigma": "0.1",
           }
       ],
   }

Multi-channel Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "distributions": [
           {
               "name": "channel_1",
               "type": "gaussian_dist",
               "x": "obs1",
               "mean": "mu1",
               "sigma": "sigma1",
           },
           {"name": "channel_2", "type": "poisson_dist", "x": "obs2", "rate": "lambda2"},
           {
               "name": "combined",
               "type": "product_dist",
               "dists": ["channel_1", "channel_2"],
           },
       ]
   }
