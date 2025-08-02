.. _model_tutorial:

Understanding and Exploring Models
==================================

This tutorial covers how to work with PyHS3 models - understanding their structure, exploring their contents, and evaluating them.

What is a Model?
---------------

A **Model** is the computational representation of your statistical model created from a workspace. It contains:

- **Parameters**: Symbolic tensor variables representing your model parameters
- **Distributions**: Compiled probability distribution functions
- **Functions**: Compiled mathematical functions
- **Computational Graph**: The dependency structure between all components

Models are created from workspaces and provide the interface for evaluating PDFs, generating samples, and performing statistical analysis.

Creating Models
---------------

Models are created from workspaces using the ``.model()`` method:

.. code-block:: python

   import pyhs3

   # Create a workspace (see workspace tutorial)
   workspace_data = {
       "metadata": {"hs3_version": "0.2"},
       "distributions": [
           {
               "name": "gaussian_model",
               "type": "gaussian_dist",
               "x": "observable",
               "mean": "mu",
               "sigma": "sigma",
           }
       ],
       "parameter_points": [
           {
               "name": "default_params",
               "parameters": [
                   {"name": "observable", "value": 0.0},
                   {"name": "mu", "value": 0.0},
                   {"name": "sigma", "value": 1.0},
               ],
           }
       ],
       "domains": [
           {
               "name": "valid_range",
               "type": "product_domain",
               "axes": [
                   {"name": "observable", "min": -5.0, "max": 5.0},
                   {"name": "mu", "min": -2.0, "max": 2.0},
                   {"name": "sigma", "min": 0.1, "max": 3.0},
               ],
           }
       ],
   }

   ws = pyhs3.Workspace(**workspace_data)

   # Create model with specific domain and parameter set
   model = ws.model(domain="valid_range", parameter_set="default_params")

   # Or use defaults (first domain and parameter set)
   model = ws.model()

Exploring Model Structure
-------------------------

Once you have a model, you can explore its structure:

.. code-block:: python

   # Print model overview
   print(model)  # Shows parameters, distributions, functions count

   # Access model components
   print("Parameters:")
   for name, tensor in model.parameters.items():
       print(f"  {name}: {tensor}")

   print("\\nDistributions:")
   for name, tensor in model.distributions.items():
       print(f"  {name}: {tensor}")

   print("\\nFunctions:")
   for name, tensor in model.functions.items():
       print(f"  {name}: {tensor}")

   # Get detailed graph information for a specific distribution
   summary = model.graph_summary("gaussian_model")
   print(f"\\nGraph summary for gaussian_model:\\n{summary}")

Understanding the Computational Graph
------------------------------------

PyHS3 models are built as computational graphs where:

- **Parameters** are leaf nodes (input variables)
- **Functions** transform parameters into intermediate values
- **Distributions** depend on parameters and/or function outputs
- **Dependencies** define the evaluation order

You can visualize the computational graph:

.. code-block:: python

   # Generate a visual graph (requires pydot)
   try:
       model.visualize_graph("gaussian_model", output_file="model_graph.png")
       print("Graph saved to model_graph.png")
   except ImportError:
       print("Install pydot to visualize graphs: pip install pydot")

Parameter Discovery and Bounds
------------------------------

PyHS3 automatically discovers parameters from your distributions and functions. Parameters are created with domain bounds applied:

.. code-block:: python

   # Parameters are automatically bounded based on domain constraints
   # For example, with domain axes:
   # {"name": "sigma", "min": 0.1, "max": 3.0}
   # The sigma parameter will be automatically constrained to [0.1, 3.0]

   # Parameters not in parameter_points are discovered and use default bounds
   minimal_workspace = {
       "metadata": {"hs3_version": "0.2"},
       "distributions": [
           {
               "name": "discovered_model",
               "type": "gaussian_dist",
               "x": "data",
               "mean": "discovered_mu",
               "sigma": "discovered_sigma",
           }
       ],
       "domains": [
           {
               "name": "constraints",
               "type": "product_domain",
               "axes": [{"name": "discovered_sigma", "min": 0.5, "max": 2.0}],
           }
       ],
       # Note: no parameter_points defined
   }

   ws_minimal = pyhs3.Workspace(**minimal_workspace)
   model_minimal = ws_minimal.model()

   print("Discovered parameters:")
   for param_name in model_minimal.parameters:
       print(f"  {param_name}")

Evaluating Models
----------------

The primary use of models is to evaluate probability density functions:

.. code-block:: python

   # Evaluate PDF at specific parameter values
   pdf_value = model.pdf("gaussian_model", observable=0.0, mu=0.0, sigma=1.0)
   print(f"PDF(0.0) = {pdf_value}")

   # Evaluate at different points
   pdf_at_1 = model.pdf("gaussian_model", observable=1.0, mu=0.0, sigma=1.0)
   pdf_at_2 = model.pdf("gaussian_model", observable=2.0, mu=0.0, sigma=1.0)

   print(f"PDF(1.0) = {pdf_at_1}")
   print(f"PDF(2.0) = {pdf_at_2}")

   # Vectorized evaluation
   import numpy as np

   x_values = np.linspace(-3, 3, 100)
   pdf_values = [
       model.pdf("gaussian_model", observable=x, mu=0.0, sigma=1.0) for x in x_values
   ]

Model Compilation and Performance
--------------------------------

Models use PyTensor for fast compilation and evaluation:

.. code-block:: python

   # Models support different compilation modes
   fast_model = ws.model(mode="FAST_RUN")  # Maximum optimization
   debug_model = ws.model(mode="FAST_COMPILE")  # Faster compilation

   # Check compilation status
   print(f"Model mode: {model.mode}")
   summary = model.graph_summary("gaussian_model")
   print("Compiled:" in summary)  # Shows if function is compiled

Working with Complex Models
---------------------------

For models with multiple distributions and functions:

.. code-block:: python

   complex_model = {
       "metadata": {"hs3_version": "0.2"},
       "distributions": [
           {
               "name": "signal",
               "type": "gaussian_dist",
               "x": "mass",
               "mean": "signal_mean",
               "sigma": "resolution",
           },
           {
               "name": "background",
               "type": "generic_dist",
               "x": "mass",
               "expression": "exp(-mass/slope)",
           },
       ],
       "functions": [
           {
               "name": "total_yield",
               "type": "sum",
               "summands": ["signal_events", "background_events"],
           },
           {
               "name": "signal_fraction",
               "type": "generic_function",
               "expression": "signal_events / total_yield",
           },
       ],
       "parameter_points": [
           {
               "name": "physics_point",
               "parameters": [
                   {"name": "signal_mean", "value": 125.0},
                   {"name": "resolution", "value": 2.5},
                   {"name": "signal_events", "value": 100.0},
                   {"name": "background_events", "value": 1000.0},
                   {"name": "slope", "value": 50.0},
               ],
           }
       ],
   }

   complex_ws = pyhs3.Workspace(**complex_model)
   complex_model = complex_ws.model()

   # Evaluate individual components
   signal_pdf = complex_model.pdf("signal", mass=125.0, signal_mean=125.0, resolution=2.5)
   background_pdf = complex_model.pdf("background", mass=125.0, slope=50.0)

   # Evaluate functions
   total = complex_model.pdf("total_yield", signal_events=100.0, background_events=1000.0)
   fraction = complex_model.pdf(
       "signal_fraction", signal_events=100.0, background_events=1000.0
   )

   print(f"Signal PDF: {signal_pdf}")
   print(f"Background PDF: {background_pdf}")
   print(f"Total yield: {total}")
   print(f"Signal fraction: {fraction}")

Debugging and Troubleshooting
-----------------------------

When working with models, you can debug issues using:

.. code-block:: python

   # 1. Check model structure
   print(model)

   # 2. Examine computational graph
   summary = model.graph_summary("distribution_name")
   print(summary)

   # 3. Use debug compilation mode
   debug_model = ws.model(mode="DebugMode")

   # 4. Visualize dependencies
   try:
       model.visualize_graph("distribution_name")
   except ImportError:
       print("Install pydot for graph visualization")

   # 5. Check parameter discovery
   print("Available parameters:", list(model.parameters.keys()))
   print("Available distributions:", list(model.distributions.keys()))
   print("Available functions:", list(model.functions.keys()))

Advanced Topics
--------------

Tensor Types
~~~~~~~~~~~~

Parameters can have different tensor types based on their intended use:

.. code-block:: python

   # In parameter_points, you can specify tensor kinds:
   vector_params = {
       "name": "vector_params",
       "parameters": [
           {"name": "scalar_param", "value": 1.0, "kind": "scalar"},  # Default
           {"name": "vector_param", "value": [1.0, 2.0], "kind": "vector"},
       ],
   }

Custom Functions
~~~~~~~~~~~~~~~

You can define custom mathematical expressions:

.. code-block:: python

   custom_function = {
       "name": "custom_calc",
       "type": "generic_function",
       "expression": "sqrt(x**2 + y**2)",  # Uses SymPy syntax
   }

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

For better performance:

- Use ``mode="FAST_RUN"`` for production models
- Avoid repeated model creation
- Cache compiled functions when possible
- Use appropriate tensor types for your data

.. code-block:: python

   # Good: reuse model
   model = ws.model(mode="FAST_RUN")
   results = []
   for x in data_points:
       result = model.pdf("my_dist", observable=x, mu=0.0, sigma=1.0)
       results.append(result)

   # Less efficient: recreate model each time
   # for x in data_points:
   #     model = ws.model()  # Don't do this
   #     result = model.pdf("my_dist", observable=x, mu=0.0, sigma=1.0)
