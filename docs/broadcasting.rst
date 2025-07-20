Broadcasting with Vector Parameters
====================================

PyHS3 supports broadcasting operations by allowing parameters to be vectors instead of scalars. This is useful when you want to evaluate the same model at multiple parameter points simultaneously.

Basic Usage
-----------

By default, all parameters are scalar tensors:

.. code-block:: python

    import pyhs3
    import pytensor.tensor as pt
    import numpy as np

    # Create workspace with a simple Gaussian
    workspace_data = {
        "distributions": [
            {
                "mean": "mu",
                "name": "model",
                "sigma": "sigma",
                "type": "gaussian_dist",
                "x": "x",
            }
        ],
        "parameter_points": [
            {
                "name": "default_values",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 0.0},
                    {"name": "sigma", "value": 1.0},
                ],
            }
        ],
    }

    ws = pyhs3.Workspace(workspace_data)
    model = ws.model()

    # Scalar evaluation
    parameters = {"x": 0.0, "mu": 0.0, "sigma": 1.0}
    result = model.logpdf("model", **parameters)
    print(f"Scalar result: {result}")

Converting Parameters to Vectors
--------------------------------

To enable broadcasting, you need to modify the parameter's ``kind`` before creating the model:

.. code-block:: python

    # Get the parameter set
    parameterset = ws.parameter_collection[0]

    # Convert 'x' parameter to vector
    parameterset["x"].kind = pt.vector

    # Create new model with vector parameter
    new_model = ws.model(parameter_point=parameterset)

Now you can pass vector values for the ``x`` parameter:

.. code-block:: python

    # Vector evaluation - multiple x values at once
    parameters = {"x": [0.0, 1.0, 2.0], "mu": 0.0, "sigma": 1.0}
    results = new_model.logpdf("model", **parameters)
    print(f"Vector results: {results}")
    # Output: Vector results: [-0.91893853, -1.41893853, -2.91893853]

Understanding the Model Structure
---------------------------------

You can inspect the model to understand its parameter structure:

.. code-block:: python

    print(model)
    # Output:
    # Model(
    #     mode: FAST_RUN
    #     parameters: 3 (x, mu, sigma)
    #     distributions: 1 (model)
    #     functions: 0 ()
    # )

    # Check parameter types
    print(f"Parameters: {model.parameters}")
    print(f"x parameter type: {type(model.parameters['x'])}")

    # Check parameter set configuration
    print(f"Parameter set: {model.parameterset}")
    print(f"x parameter config: {model.parameterset['x']}")

Error Handling
--------------

If you try to pass vector values to a scalar parameter, you'll get a dimension mismatch error:

.. code-block:: python

    # This will fail - trying to pass vector to scalar parameter
    parameters = {"x": [0.0, 1.0], "mu": 0.0, "sigma": 1.0}
    try:
        model.logpdf("model", **parameters)
    except TypeError as e:
        print(f"Error: {e}")
        # Error: Wrong number of dimensions: expected 0, got 1 with shape (2,).

Complete Example
----------------

Here's a complete working example:

.. code-block:: python

    import pyhs3
    import pytensor.tensor as pt
    import numpy as np

    # Define workspace
    workspace_data = {
        "distributions": [
            {
                "mean": "mu",
                "name": "gaussian",
                "sigma": "sigma",
                "type": "gaussian_dist",
                "x": "x",
            }
        ],
        "parameter_points": [
            {
                "name": "default_values",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 0.0},
                    {"name": "sigma", "value": 1.0},
                ],
            }
        ],
    }

    ws = pyhs3.Workspace(workspace_data)

    # Method 1: Scalar evaluation
    scalar_model = ws.model()
    scalar_result = scalar_model.logpdf("gaussian", x=0.0, mu=0.0, sigma=1.0)
    print(f"Scalar: {scalar_result}")

    # Method 2: Vector evaluation
    parameterset = ws.parameter_collection[0]
    parameterset["x"].kind = pt.vector
    vector_model = ws.model(parameter_point=parameterset)

    # Evaluate at multiple x values
    x_values = np.linspace(-2, 2, 5)
    vector_results = vector_model.logpdf("gaussian", x=x_values, mu=0.0, sigma=1.0)
    print(f"Vector: {vector_results}")

Current Limitations
------------------

- Users must manually specify which parameters should be vectors
- The ``kind`` must be set before creating the model
- No automatic inference of parameter dimensionality

.. note::
   A more user-friendly API for automatic broadcasting detection is planned for future releases.
