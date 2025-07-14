Python API
==========

Top-Level
---------

.. currentmodule:: pyhs3

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Workspace
   Model

Modules
-------

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   distributions
   functions
   generic_parse
   exceptions

Functions
---------

.. currentmodule:: pyhs3.functions

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Function
   ProductFunction
   GenericFunction
   InterpolationFunction

Distributions
-------------

.. currentmodule:: pyhs3.distributions

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Distribution
   GaussianDist
   MixtureDist
   ProductDist
   CrystalBallDist
   GenericDist

Parsing
-------

.. currentmodule:: pyhs3.generic_parse

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   parse_expression
   sympy_to_pytensor
   analyze_sympy_expr

Exceptions
----------

.. currentmodule:: pyhs3.exceptions

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   HS3Exception
   ExpressionParseError
   ExpressionEvaluationError
   UnknownInterpolationCodeError
