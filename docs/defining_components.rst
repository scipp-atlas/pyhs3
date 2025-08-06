Defining Custom Functions and Distributions
==========================================

This guide shows how to create custom Functions and Distributions for use with pyhs3.
Both Functions and Distributions inherit from the ``Evaluable`` base class, which provides
automatic parameter preprocessing to eliminate boilerplate validation code.

Quick Start
-----------

Creating a custom distribution or function requires:

1. **Inherit** from ``Distribution`` or ``Function``
2. **Define fields** with appropriate type annotations
3. **Implement** the ``expression()`` method
4. **Register** your component (optional)

The ``Evaluable`` base class automatically handles parameter processing based on your field type annotations.

Basic Distribution Example
--------------------------

Here's a simple custom Gaussian distribution:

.. code-block:: python

    from typing import Literal
    import pytensor.tensor as pt
    from pyhs3.distributions.core import Distribution
    from pyhs3.context import Context
    from pyhs3.typing.aliases import TensorVar


    class CustomGaussianDist(Distribution):
        """Custom Gaussian distribution implementation."""

        type: Literal["custom_gaussian"] = "custom_gaussian"
        mean: str | float  # Parameter name or numeric value
        sigma: str | float  # Parameter name or numeric value

        def expression(self, context: Context) -> TensorVar:
            """Evaluate the Gaussian PDF."""
            # Get processed parameters from context
            mean_val = context[self._parameters["mean"]]
            sigma_val = context[self._parameters["sigma"]]

            # Assume 'x' is the observable variable
            x = context["x"]  # Would come from domain/data definition

            # Gaussian PDF formula
            norm = 1.0 / (sigma_val * pt.sqrt(2 * pt.pi))
            exp_term = pt.exp(-0.5 * ((x - mean_val) / sigma_val) ** 2)
            return norm * exp_term

**What happens automatically:**

- If ``mean="mu_param"``, then ``self._parameters["mean"] == "mu_param"``
- If ``mean=1.5``, then ``self._parameters["mean"] == "constant_myname_mean"`` and a constant is created
- The ``parameters`` property returns all parameter names this distribution depends on
- The ``constants`` property provides PyTensor constants for numeric values

Basic Function Example
----------------------

Here's a custom product function:

.. code-block:: python

    from typing import Literal
    import pytensor.tensor as pt
    from pyhs3.functions.core import Function
    from pyhs3.context import Context
    from pyhs3.typing.aliases import TensorVar


    class WeightedProductFunction(Function):
        """Product function with weights."""

        type: Literal["weighted_product"] = "weighted_product"
        factors: list[str | float]  # Mix of parameter names and values
        weights: list[str | float]  # Corresponding weights

        def expression(self, context: Context) -> TensorVar:
            """Evaluate weighted product: prod(factor[i] ** weight[i])."""
            # Get parameter lists in original order
            factor_vals = self.get_parameter_list(context, "factors")
            weight_vals = self.get_parameter_list(context, "weights")

            result = pt.constant(1.0)
            for factor, weight in zip(factor_vals, weight_vals, strict=True):
                result = result * (factor**weight)
            return result

**What happens automatically:**

- ``factors=["param1", 2.0, "param2"]`` creates indexed parameters: ``factors[0]``, ``factors[1]``, ``factors[2]``
- ``get_parameter_list(context, "factors")`` reconstructs the original list from context
- Constants are generated for numeric values: ``constant_myname_factors[1]`` for ``2.0``

Automatic Parameter Processing
------------------------------

The ``Evaluable`` base class automatically processes field annotations:

**Supported Field Types:**

.. code-block:: python

    class MyComponent(Evaluable):
        # String fields -> direct parameter mapping
        param_name: str  # -> self._parameters["param_name"] = field_value

        # Numeric fields -> generate constants
        numeric_val: (
            float  # -> self._parameters["numeric_val"] = "constant_name_numeric_val"
        )

        # Union types -> runtime detection
        mixed_param: str | float  # -> string or constant depending on value
        flexible: str | int | float  # -> handles any combination

        # Lists -> indexed processing
        string_list: list[str]  # -> param_name[0], param_name[1], ...
        mixed_list: list[str | float]  # -> mix of strings and generated constants

        # Excluded fields
        config_flag: bool  # -> automatically excluded
        internal_val: float = Field(  # -> explicitly excluded
            default=1.0, json_schema_extra={"preprocess": False}
        )

**Exclusion Rules:**

- **Boolean fields** are automatically excluded (not parameters)
- **Fields marked** with ``json_schema_extra={"preprocess": False}`` are excluded
- **Base class fields** (``name``, ``type``) are excluded
- **None values** are skipped

Advanced Examples
-----------------

**Complex Distribution with Mixed Parameters:**

.. code-block:: python

    from pydantic import Field


    class FlexibleDist(Distribution):
        type: Literal["flexible"] = "flexible"

        # Core parameters (will be processed)
        location: str | float
        scale: str | float
        coefficients: list[str | float]

        # Configuration (excluded from processing)
        use_log_scale: bool = False
        tolerance: float = Field(default=1e-6, json_schema_extra={"preprocess": False})

        def expression(self, context: Context) -> TensorVar:
            loc = context[self._parameters["location"]]
            scale = context[self._parameters["scale"]]

            # Get coefficient list
            coeffs = self.get_parameter_list(context, "coefficients")

            # Use configuration values directly
            if self.use_log_scale:
                scale = pt.exp(scale)

            # ... implementation
            return result

**Function with Validation:**

.. code-block:: python

    from pydantic import model_validator


    class ValidatedFunction(Function):
        type: Literal["validated"] = "validated"
        inputs: list[str]
        weights: list[float] = Field(json_schema_extra={"preprocess": False})

        @model_validator(mode="after")
        def validate_lengths(self) -> "ValidatedFunction":
            """Custom validation after auto-processing."""
            if len(self.inputs) != len(self.weights):
                raise ValueError("inputs and weights must have same length")
            return self

        def expression(self, context: Context) -> TensorVar:
            # inputs were auto-processed into indexed parameters
            input_vals = self.get_parameter_list(context, "inputs")

            result = pt.constant(0.0)
            for inp, weight in zip(input_vals, self.weights, strict=True):
                result = result + inp * weight
            return result

Registration and Discovery
--------------------------

**Option 1: Manual Registration**

Add your components to the appropriate registry:

.. code-block:: python

    # For distributions
    from pyhs3.distributions.core import registered_distributions

    registered_distributions["custom_gaussian"] = CustomGaussianDist

    # For functions
    from pyhs3.functions.core import registered_functions

    registered_functions["weighted_product"] = WeightedProductFunction

**Option 2: Plugin System** (if available)

Check if pyhs3 supports a plugin entry point system for automatic discovery.

Usage in Workspaces
-------------------

Once defined, your custom components work like built-in ones:

.. code-block:: python

    # In JSON/YAML workspace definition
    {
        "distributions": [
            {
                "name": "signal_pdf",
                "type": "custom_gaussian",
                "mean": "mu_signal",  # Parameter reference
                "sigma": 0.1,  # Numeric constant
            }
        ],
        "functions": [
            {
                "name": "weighted_norm",
                "type": "weighted_product",
                "factors": ["norm1", "norm2", 1.5],  # Mixed types
                "weights": [2.0, 1.0, 0.5],  # Config values
            }
        ],
    }

    # In Python
    workspace = Workspace.from_file("my_workspace.json")
    model = workspace.model()

    # Your components are automatically instantiated and available

Error Handling and Debugging
-----------------------------

**Common Issues:**

1. **Unsupported field type:**

   .. code-block:: python

       class BadDist(Distribution):
           type: Literal["bad"] = "bad"
           complex_field: dict  # Not supported!

   **Fix:** Add ``json_schema_extra={"preprocess": False}`` or use supported types.

2. **Missing implementation:**

   .. code-block:: python

       dist = MyDist(name="test", param="value")
       # RuntimeError: Component type my_type expression not implemented

   **Fix:** Implement the ``expression()`` method.

3. **Context key errors:**

   .. code-block:: python

       def expression(self, context):
           return context["missing_param"]  # KeyError!

   **Fix:** Use ``self._parameters`` or ``self.get_parameter_list()`` to get correct keys.

**Debugging Tips:**

.. code-block:: python

    # Inspect what was auto-processed
    dist = MyDist(name="test", param1="alpha", param2=1.5)

    print("Parameters:", dist.parameters)  # All parameter names
    print("Internal mapping:", dist._parameters)  # Field -> parameter mapping
    print("Constants:", list(dist.constants.keys()))  # Generated constant names
    print("Constant values:", dist._constants_values)  # Stored numeric values

Best Practices
--------------

1. **Use descriptive type literals** for easy identification
2. **Document your components** with clear docstrings and examples
3. **Handle edge cases** in your ``expression()`` method
4. **Test thoroughly** with different parameter combinations
5. **Consider performance** - PyTensor operations should be efficient
6. **Follow naming conventions** - use clear, descriptive field names
7. **Validate inputs** when auto-processing isn't sufficient

The automatic parameter processing handles most common cases, letting you focus on the mathematical implementation rather than parameter management boilerplate.
