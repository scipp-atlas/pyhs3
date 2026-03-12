Broadcasting with Vector Parameters
====================================

PyHS3 supports broadcasting operations by allowing parameters to be vectors instead of scalars. This is useful when you want to evaluate the same model at multiple parameter points simultaneously.

Basic Usage
-----------

By default, most parameters are scalar tensors. However, parameters identified as observables (via likelihood data axes) are automatically created as 1D vectors to enable batched evaluation and numerical integration:

.. doctest::

    >>> import pyhs3
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> # Create workspace with a simple Gaussian
    >>> workspace_data = {
    ...     "metadata": {"hs3_version": "0.2"},
    ...     "distributions": [
    ...         {
    ...             "mean": "mu",
    ...             "name": "model",
    ...             "sigma": "sigma",
    ...             "type": "gaussian_dist",
    ...             "x": "x",
    ...         }
    ...     ],
    ...     "parameter_points": [
    ...         {
    ...             "name": "default_values",
    ...             "parameters": [
    ...                 {"name": "x", "value": 0.0},
    ...                 {"name": "mu", "value": 0.0},
    ...                 {"name": "sigma", "value": 1.0},
    ...             ],
    ...         }
    ...     ],
    ... }
    >>> ws = pyhs3.Workspace(**workspace_data)
    >>> model = ws.model()
    <BLANKLINE>
    >>> # Scalar evaluation
    >>> parameters = {"x": 0.0, "mu": 0.0, "sigma": 1.0}
    >>> result = model.logpdf_unsafe("model", **parameters)
    >>> print(f"Scalar result: {result}")
    Scalar result: -0.9189385332046727

Converting Parameters to Vectors
--------------------------------

To enable broadcasting, you need to modify the parameter's ``kind`` before creating the model:

.. doctest::

    :options: +ELLIPSIS

    >>> import warnings
    >>> # Get the parameter set
    >>> parameterset = ws.parameter_points[0]
    >>> # Convert 'x' parameter to vector
    >>> parameterset["x"].kind = pt.vector
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     # Create new model with vector parameter
    ...     new_model = ws.model(parameter_set=parameterset)
    ...     print(w[0].message)  # shows the warning in the docs
    ...
    <BLANKLINE>
    Parameter 'x' has kind override vector (default would be scalar)

Now you can pass vector values for the ``x`` parameter:

.. code-block:: pycon

    >>> # Vector evaluation - multiple x values at once
    >>> parameters = {"x": [0.0, 1.0, 2.0], "mu": 0.0, "sigma": 1.0}
    >>> results = new_model.logpdf_unsafe("model", **parameters)
    >>> print(f"Vector results: {results}")
    Vector results: [-0.91893853 -1.41893853 -2.91893853]

Understanding the Model Structure
---------------------------------

You can inspect the model to understand its parameter structure:

.. code-block:: pycon

    >>> print(model)  # doctest: +ELLIPSIS
    Model(
        mode: FAST_RUN
        parameters: 3 (...)
        distributions: 1 (model)
        functions: 0 ()
    )
    >>> # Check parameter types
    >>> print(f"Parameters: {sorted(model.parameters)}")
    Parameters: ['mu', 'sigma', 'x']
    >>> print(f"x parameter type: {type(model.parameters['x'])}")
    x parameter type: <class 'pytensor.tensor.variable.TensorVariable'>
    >>> # Check parameter set configuration
    >>> print(f"Parameter set: {model.parameterset}")  # doctest: +ELLIPSIS
    Parameter set: ...
    >>> print(f"x parameter config: {model.parameterset['x']}")
    x parameter config: name='x'

Behavior Difference: Scalar vs Vector Parameters
------------------------------------------------

When you pass vector values to a scalar parameter model, it will only use the first element:

.. code-block:: pycon

    >>> # Scalar model with vector input - only uses first element
    >>> parameters = {"x": [0.0, 1.0, 2.0], "mu": 0.0, "sigma": 1.0}
    >>> result = model.pdf_unsafe("model", **parameters)
    >>> print(f"Result with scalar model: {result}")
    Result with scalar model: 0.3989422804014327
    >>> # Compare with vector model - processes all elements
    >>> result_vector = new_model.pdf_unsafe("model", **parameters)
    >>> print(f"Result with vector model: {result_vector}")
    Result with vector model: [0.39894228 0.24197072 0.05399097]

Complete Example
----------------

Here's a complete working example:

.. doctest::

    :options: +ELLIPSIS

    >>> import pyhs3
    >>> import pytensor.tensor as pt
    >>> import numpy as np
    >>> import warnings
    >>> # Define workspace
    >>> workspace_data = {
    ...     "metadata": {"hs3_version": "0.2"},
    ...     "distributions": [
    ...         {
    ...             "mean": "mu",
    ...             "name": "gaussian",
    ...             "sigma": "sigma",
    ...             "type": "gaussian_dist",
    ...             "x": "x",
    ...         }
    ...     ],
    ...     "parameter_points": [
    ...         {
    ...             "name": "default_values",
    ...             "parameters": [
    ...                 {"name": "x", "value": 0.0},
    ...                 {"name": "mu", "value": 0.0},
    ...                 {"name": "sigma", "value": 1.0},
    ...             ],
    ...         }
    ...     ],
    ... }
    >>> ws = pyhs3.Workspace(**workspace_data)
    >>> # Method 1: Scalar evaluation
    >>> scalar_model = ws.model()
    <BLANKLINE>
    >>> scalar_result = scalar_model.logpdf_unsafe("gaussian", x=0.0, mu=0.0, sigma=1.0)
    >>> print(f"Scalar: {scalar_result}")
    Scalar: -0.9189385332046727
    >>> # Method 2: Vector evaluation
    >>> parameterset = ws.parameter_points[0]
    >>> parameterset["x"].kind = pt.vector
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     vector_model = ws.model(parameter_set=parameterset)
    ...     print(w[0].message)  # shows the warning in the docs
    ...
    <BLANKLINE>
    Parameter 'x' has kind override vector (default would be scalar)
    >>> # Evaluate at multiple x values
    >>> x_values = np.linspace(-2, 2, 5)
    >>> vector_results = vector_model.logpdf_unsafe(
    ...     "gaussian", x=x_values.tolist(), mu=0.0, sigma=1.0
    ... )
    >>> print(f"Vector: {vector_results}")
    Vector: [-2.91893853 -1.41893853 -0.91893853 -1.41893853 -2.91893853]

Current Behavior
-------------------

- Parameters identified as observables (via likelihood data axes) are automatically
  created as 1D vectors (``pt.vector``). This enables batched evaluation and
  numerical integration over the observable domain.
- Non-observable parameters default to scalars (``pt.scalar``).
- Users can override the default ``kind`` on a ``ParameterPoint`` before model
  creation. A warning is emitted when the override differs from the automatically
  determined default.
- The ``kind`` must be set before creating the model.
