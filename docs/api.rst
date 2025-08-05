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
   domains
   parameter_points
   likelihoods
   analyses
   generic_parse
   exceptions

Functions
---------

.. currentmodule:: pyhs3.functions

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Functions
   Function
   ProductFunction
   SumFunction
   GenericFunction
   InterpolationFunction
   ProcessNormalizationFunction

Distributions
-------------

.. currentmodule:: pyhs3.distributions

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Distributions
   Distribution
   GaussianDist
   MixtureDist
   ProductDist
   CrystalBallDist
   GenericDist
   PoissonDist

Domains
-------

.. currentmodule:: pyhs3.domains

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Domains
   Domain
   ProductDomain
   Axis

Parameter Points
----------------

.. currentmodule:: pyhs3.parameter_points

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   ParameterPoints
   ParameterSet
   ParameterPoint

Parsing
-------

.. currentmodule:: pyhs3.generic_parse

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   parse_expression
   sympy_to_pytensor
   analyze_sympy_expr

Likelihoods
-----------

.. currentmodule:: pyhs3.likelihoods

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Likelihoods
   Likelihood

Analyses
--------

.. currentmodule:: pyhs3.analyses

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Analyses
   Analysis

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
