.. _workspace_tutorial:

Working with Workspaces
=======================

This tutorial covers how to work with PyHS3 workspaces - loading, exploring, and understanding their structure.

What is a Workspace?
-------------------

A **Workspace** is the main container in PyHS3 that holds all the components needed to define a statistical model:

- **Distributions**: Probability distributions (Gaussian, Poisson, etc.)
- **Functions**: Mathematical functions that compute parameter values
- **Domains**: Parameter space constraints and bounds
- **Parameter Points**: Named sets of parameter values
- **Metadata**: Version information and documentation

Loading a Workspace
-------------------

You can create a workspace from a dictionary or load it from a JSON file:

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

   classDiagram
       class Workspace {
           +metadata: Metadata
           +distributions: list[Distribution]
           +functions: list[Function]
           +domains: list[Domain]
           +parameter_points: list[ParameterSet]
           +data: optional
           +likelihoods: optional
           +analyses: optional
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

       Workspace ||--|| Metadata : contains
       Workspace ||--o{ Distribution : contains
       Workspace ||--o{ Function : contains
       Workspace ||--o{ Domain : contains
       Workspace ||--o{ ParameterSet : contains

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

Here's a more realistic example of a workspace for a physics analysis:

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
   }

   physics_ws = pyhs3.Workspace(**physics_model)
   physics_model = physics_ws.model()

   # Evaluate signal and background separately
   signal_pdf = physics_model.pdf("signal", mass=125.0, higgs_mass=125.0, resolution=2.5)
   background_pdf = physics_model.pdf("background", mass=125.0, lifetime=50.0, norm=1.0)

   print(f"Signal PDF at 125 GeV: {signal_pdf}")
   print(f"Background PDF at 125 GeV: {background_pdf}")
