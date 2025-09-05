.. _workspace_tutorial:

Working with Workspaces
=======================

This tutorial covers how to work with PyHS3 workspaces - loading, exploring, and understanding their structure.

What is a Workspace?
-------------------

A **Workspace** is the main container in PyHS3 that holds all the components needed to define a statistical model:

- **Distributions**: Probability distributions (:ref:`hs3:hs3.gaussian-normal-distribution`, :ref:`hs3:hs3.dist:poisson`, etc.)
- **Functions**: Mathematical functions that compute parameter values (:ref:`hs3:hs3.sum`, :ref:`hs3:hs3.product`, :ref:`hs3:hs3.generic-function`)
- **Domains**: Parameter space constraints and bounds
- **Parameter Points**: Named sets of parameter values
- **Data**: Observed data specifications (point data, unbinned data, binned/histogram data)
- **Likelihoods**: Mappings between distributions and data
- **Analyses**: Complete analysis configurations
- **Metadata**: Version information and documentation

Loading a Workspace
-------------------

You can create a workspace from a dictionary or load it from a JSON file. The following example shows a simple workspace with a :ref:`Gaussian distribution <hs3:hs3.gaussian-normal-distribution>`:

.. code-block:: python

   import pyhs3

   # From a dictionary
   workspace_data = {
       "metadata": {"hs3_version": "0.2"},
       "distributions": [
           {
               "name": "signal",
               "type": "gaussian_dist",
               "x": "obs",
               "mean": "mu",
               "sigma": "sigma",
           }
       ],
       "parameter_points": [
           {
               "name": "nominal",
               "parameters": [
                   {"name": "obs", "value": 0.0},
                   {"name": "mu", "value": 0.0},
                   {"name": "sigma", "value": 1.0},
               ],
           }
       ],
       "domains": [
           {
               "name": "physics_region",
               "type": "product_domain",
               "axes": [
                   {"name": "obs", "min": -5.0, "max": 5.0},
                   {"name": "mu", "min": -2.0, "max": 2.0},
                   {"name": "sigma", "min": 0.1, "max": 3.0},
               ],
           }
       ],
   }

   ws = pyhs3.Workspace(**workspace_data)

   # From a JSON file
   # ws = pyhs3.Workspace.load("my_model.json")

Exploring Workspace Contents
----------------------------

Once you have a workspace, you can explore its contents:

.. code-block:: pycon

   >>> import pyhs3
   >>> workspace_data = {
   ...     "metadata": {"hs3_version": "0.2"},
   ...     "distributions": [
   ...         {
   ...             "name": "signal",
   ...             "type": "gaussian_dist",
   ...             "x": "obs",
   ...             "mean": "mu",
   ...             "sigma": "sigma",
   ...         }
   ...     ],
   ...     "parameter_points": [
   ...         {
   ...             "name": "nominal",
   ...             "parameters": [
   ...                 {"name": "obs", "value": 0.0},
   ...                 {"name": "mu", "value": 0.0},
   ...                 {"name": "sigma", "value": 1.0},
   ...             ],
   ...         }
   ...     ],
   ...     "domains": [
   ...         {
   ...             "name": "physics_region",
   ...             "type": "product_domain",
   ...             "axes": [
   ...                 {"name": "obs", "min": -5.0, "max": 5.0},
   ...                 {"name": "mu", "min": -2.0, "max": 2.0},
   ...                 {"name": "sigma", "min": 0.1, "max": 3.0},
   ...             ],
   ...         }
   ...     ],
   ... }
   >>> ws = pyhs3.Workspace(**workspace_data)
   >>> # Print workspace structure
   >>> print(f"Workspace contains:")
   Workspace contains:
   >>> print(f"- {len(ws.distributions)} distributions")
   - 1 distributions
   >>> print(f"- {len(ws.functions)} functions")
   - 0 functions
   >>> print(f"- {len(ws.domains)} domains")
   - 1 domains
   >>> print(f"- {len(ws.parameter_points)} parameter sets")
   - 1 parameter sets
   >>> print(f"- {len(ws.data)} data components")
   - 0 data components
   >>> print(f"- {len(ws.likelihoods)} likelihoods")
   - 0 likelihoods
   >>> print(f"- {len(ws.analyses)} analyses")
   - 0 analyses

   # Access distributions
   print("\\nDistributions:")
   for dist in ws.distributions:
       print(f"  {dist.name} ({dist.type})")
       print(f"    Parameters: {list(dist.parameters.values())}")

   # Access parameter sets
   print("\\nParameter sets:")
   for param_set in ws.parameter_points:
       print(f"  {param_set.name}:")
       for param in param_set.parameters:
           print(f"    {param.name} = {param.value}")

   # Access domains
   print("\\nDomains:")
   for domain in ws.domains:
       print(f"  {domain.name}:")
       for axis in domain.axes:
           print(f"    {axis.name}: [{axis.min}, {axis.max}]")

Understanding Workspace Structure
--------------------------------

The workspace follows a hierarchical structure:

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

   classDiagram
       class Workspace {
           +metadata: Metadata
           +distributions: list[Distribution]
           +functions: list[Function]
           +domains: list[Domain]
           +parameter_points: list[ParameterSet]
           +data: list[Data]
           +likelihoods: Likelihoods
           +analyses: Analyses
       }

       class Metadata {
           +hs3_version: str
           +authors: optional[list]
           +description: optional[str]
       }

       class Distribution {
           +name: str
           +type: str
           +parameters: dict
       }

       class Function {
           +name: str
           +type: str
           +parameters: dict
       }

       class Domain {
           +name: str
           +type: str
           +axes: list[Axis]
       }

       class ParameterSet {
           +name: str
           +parameters: list[ParameterPoint]
       }

       class Likelihood {
           +name: str
           +distributions: list[str]
           +data: list[str|float|int]
           +aux_distributions: optional[list[str]]
       }

       class Analysis {
           +name: str
           +likelihood: str
           +domains: list[str]
           +parameters_of_interest: optional[list[str]]
           +init: optional[str]
           +prior: optional[str]
       }

       class Datum {
           +name: str
           +type: str
       }

       class PointData {
           +name: str
           +type: "point"
           +value: float
           +uncertainty: optional[float]
       }

       class UnbinnedData {
           +name: str
           +type: "unbinned"
           +entries: list[list[float]]
           +axes: list[Axis]
           +weights: optional[list[float]]
           +entries_uncertainties: optional[list[list[float]]]
       }

       class BinnedData {
           +name: str
           +type: "binned"
           +contents: list[float]
           +axes: list[Axis]
           +uncertainty: optional[GaussianUncertainty]
       }

       Workspace --> Metadata : contains
       Workspace --> Distribution : contains
       Workspace --> Function : contains
       Workspace --> Domain : contains
       Workspace --> ParameterSet : contains
       Workspace --> Datum : contains
       Workspace --> Likelihood : contains
       Workspace --> Analysis : contains
       Datum <|-- PointData : inherits
       Datum <|-- UnbinnedData : inherits
       Datum <|-- BinnedData : inherits

Creating Models from Workspaces
------------------------------

The main purpose of a workspace is to create models that you can evaluate:

.. code-block:: python

   # Create a model using specific domain and parameter set
   model = ws.model(domain="physics_region", parameter_set="nominal")

   # Or use defaults (index 0)
   model = ws.model()

   # Evaluate the model
   result = model.pdf("signal", obs=0.5, mu=0.0, sigma=1.0)
   print(f"PDF value: {result}")

Example: Complete Physics Model
------------------------------

Here's a more realistic example of a workspace for a physics analysis using both :ref:`Gaussian distributions <hs3:hs3.gaussian-normal-distribution>` and :ref:`generic expressions <hs3:hs3.sec:generic_expression>` with a :ref:`sum function <hs3:hs3.sum>`:

.. code-block:: python

   physics_model = {
       "metadata": {
           "hs3_version": "0.2",
           "authors": ["Physics Analysis Team"],
           "description": "Signal + background model for Higgs search",
       },
       "distributions": [
           {
               "name": "signal",
               "type": "gaussian_dist",
               "x": "mass",
               "mean": "higgs_mass",
               "sigma": "resolution",
           },
           {
               "name": "background",
               "type": "generic_dist",
               "x": "mass",
               "expression": "exp(-mass/lifetime) / norm",
           },
       ],
       "functions": [
           {
               "name": "total_events",
               "type": "sum",
               "summands": ["signal_yield", "background_yield"],
           }
       ],
       "parameter_points": [
           {
               "name": "best_fit",
               "parameters": [
                   {"name": "higgs_mass", "value": 125.0},
                   {"name": "resolution", "value": 2.5},
                   {"name": "signal_yield", "value": 100.0},
                   {"name": "background_yield", "value": 1000.0},
                   {"name": "lifetime", "value": 50.0},
                   {"name": "norm", "value": 1.0},
               ],
           }
       ],
       "domains": [
           {
               "name": "search_window",
               "type": "product_domain",
               "axes": [
                   {"name": "mass", "min": 110.0, "max": 140.0},
                   {"name": "higgs_mass", "min": 120.0, "max": 130.0},
                   {"name": "resolution", "min": 1.0, "max": 5.0},
                   {"name": "signal_yield", "min": 0.0, "max": 500.0},
                   {"name": "background_yield", "min": 100.0, "max": 5000.0},
               ],
           }
       ],
       "data": [
           {
               "name": "observed_mass_spectrum",
               "type": "binned",
               "contents": [50, 75, 45],
               "axes": [{"name": "mass", "edges": [110.0, 120.0, 125.0, 130.0, 140.0]}],
               "uncertainty": {"type": "gaussian_uncertainty", "sigma": [7.1, 8.7, 6.7]},
           }
       ],
       "likelihoods": [
           {
               "name": "higgs_likelihood",
               "distributions": ["signal", "background"],
               "data": ["observed_mass_spectrum", "observed_mass_spectrum"],
           }
       ],
       "analyses": [
           {
               "name": "higgs_discovery",
               "likelihood": "higgs_likelihood",
               "domains": ["search_window"],
               "parameters_of_interest": ["higgs_mass", "signal_yield"],
               "init": "best_fit",
           }
       ],
   }

   physics_ws = pyhs3.Workspace(**physics_model)

   # Explore the workspace
   print(
       f"Workspace contains {len(physics_ws.likelihoods)} likelihoods and {len(physics_ws.analyses)} analyses"
   )
   print(
       f"Analysis '{physics_ws.analyses[0].name}' uses likelihood '{physics_ws.analyses[0].likelihood}'"
   )

   physics_model = physics_ws.model()

   # Evaluate signal and background separately
   signal_pdf = physics_model.pdf("signal", mass=125.0, higgs_mass=125.0, resolution=2.5)
   background_pdf = physics_model.pdf("background", mass=125.0, lifetime=50.0, norm=1.0)

   print(f"Signal PDF at 125 GeV: {signal_pdf}")
   print(f"Background PDF at 125 GeV: {background_pdf}")

Working with Likelihoods and Analyses
-------------------------------------

Likelihoods and analyses are optional but important components for statistical inference:

.. code-block:: python

   # Access likelihood information
   likelihood = physics_ws.likelihoods["higgs_likelihood"]
   print(f"Likelihood '{likelihood.name}' connects:")
   print(f"  - Distributions: {likelihood.distributions}")
   print(f"  - To data: {likelihood.data}")

   # Access analysis configuration
   analysis = physics_ws.analyses["higgs_discovery"]
   print(f"Analysis '{analysis.name}' configuration:")
   print(f"  - Uses likelihood: {analysis.likelihood}")
   print(f"  - Parameter domains: {analysis.domains}")
   print(f"  - Parameters of interest: {analysis.parameters_of_interest}")
   print(f"  - Initial values from: {analysis.init}")

   # These components provide structured access to the complete statistical model
   # for use with fitting and inference tools

Working with Data Components
----------------------------

The data component in PyHS3 provides structured specifications for observed data used in likelihood evaluations. There are three types of data supported:

**Point Data**: Single measurements with optional uncertainties (see :mod:`HS3 data specification <hs3:chapters.2.3_data>`)

.. code-block:: python

   point_data_example = {
       "name": "higgs_mass_measurement",
       "type": "point",
       "value": 125.09,
       "uncertainty": 0.24,
   }

**Unbinned Data**: Individual data points in multi-dimensional space

.. code-block:: python

   unbinned_data_example = {
       "name": "particle_tracks",
       "type": "unbinned",
       "entries": [
           [120.5, 0.8],  # [mass, momentum] for event 1
           [125.1, 1.2],  # [mass, momentum] for event 2
           [122.3, 0.9],  # [mass, momentum] for event 3
       ],
       "axes": [
           {"name": "mass", "min": 100.0, "max": 150.0},
           {"name": "momentum", "min": 0.0, "max": 5.0},
       ],
       "weights": [0.8, 1.0, 0.9],  # optional event weights
       "entries_uncertainties": [  # optional uncertainties for each coordinate
           [0.1, 0.05],
           [0.2, 0.08],
           [0.15, 0.06],
       ],
   }

**Binned Data**: Histogram data with bin contents and optional uncertainties

.. code-block:: python

   binned_data_example = {
       "name": "mass_spectrum",
       "type": "binned",
       "contents": [45.0, 67.0, 52.0, 38.0],  # bin contents
       "axes": [
           {
               "name": "mass",
               "edges": [110.0, 120.0, 130.0, 140.0, 150.0],  # irregular binning
           }
       ],
       "uncertainty": {
           "type": "gaussian_uncertainty",
           "sigma": [6.7, 8.2, 7.2, 6.2],  # uncertainties for each bin
           "correlation": 0,  # or correlation matrix for correlated uncertainties
       },
   }

   # Regular binning alternative
   regular_binned_example = {
       "name": "pt_spectrum",
       "type": "binned",
       "contents": [100.0, 80.0, 60.0, 40.0, 20.0],
       "axes": [
           {
               "name": "pt",
               "min": 0.0,
               "max": 100.0,
               "nbins": 5,  # regular binning: 5 bins from 0 to 100
           }
       ],
   }

**Accessing Data in Workspaces**

.. code-block:: python

   # Access data components
   print(f"\\nData components ({len(physics_ws.data)}):")
   for datum in physics_ws.data:
       print(f"  {datum.name} ({datum.type})")
       if hasattr(datum, "value"):
           print(f"    Value: {datum.value}")
       elif hasattr(datum, "contents"):
           print(f"    Bins: {len(datum.contents)}")
       elif hasattr(datum, "entries"):
           print(f"    Events: {len(datum.entries)}")

   # Get specific data by name
   mass_data = physics_ws.data["observed_mass_spectrum"]
   print(f"Data '{mass_data.name}' has {len(mass_data.contents)} bins")

   # Check if data exists
   if "observed_mass_spectrum" in physics_ws.data:
       print("Mass spectrum data is available")

Data components integrate with likelihoods to define the complete statistical model for parameter estimation and hypothesis testing.
