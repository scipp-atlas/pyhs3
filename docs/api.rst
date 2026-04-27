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
   jaxify
   JaxifiedGraph

Modules
-------

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   distributions
   functions
   axes
   domains
   parameter_points
   data
   likelihoods
   analyses
   generic_parse
   exceptions
   normalization
   transpile

Base Classes
------------

.. currentmodule:: pyhs3.base

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Evaluable

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
   UniformDist
   PoissonDist
   ExponentialDist
   LogNormalDist
   LandauDist
   MixtureDist
   ProductDist
   HistogramDist
   GenericDist
   PolynomialDist
   BernsteinPolyDist
   CrystalBallDist
   AsymmetricCrystalBallDist
   ArgusDist
   HistFactoryDistChannel
   FastVerticalInterpHistPdf2Dist
   GGZZBackgroundDist
   QQZZBackgroundDist
   FastVerticalInterpHistPdf2D2Dist

HistFactory
-----------

.. currentmodule:: pyhs3.distributions.histfactory

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   samples.Sample
   samples.Samples
   data.SampleData
   modifiers.Modifier
   modifiers.ModifierData
   modifiers.HasConstraint
   modifiers.ParameterModifier
   modifiers.ParametersModifier
   modifiers.NormFactorModifier
   modifiers.NormSysModifier
   modifiers.HistoSysModifier
   modifiers.ShapeFactorModifier
   modifiers.ShapeSysModifier
   modifiers.StatErrorModifier
   modifiers.NormSysData
   modifiers.HistoSysData
   modifiers.HistoSysDataContents
   modifiers.ShapeSysData
   modifiers.StatErrorData
   modifiers.Modifiers

Axes
----

.. currentmodule:: pyhs3.axes

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Axis
   BoundedAxis
   UnbinnedAxis
   ConstantAxis
   RegularAxis
   IrregularAxis
   DomainCoordinateAxis
   BinnedAxis
   BinnedAxes
   UnbinnedAxes
   Axes
   DomainAxes
   DomainAxis

Domains
-------

.. currentmodule:: pyhs3.domains

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Domains
   Domain
   ProductDomain

Parameter Points
----------------

.. currentmodule:: pyhs3.parameter_points

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   ParameterPoints
   ParameterSet
   ParameterPoint

Data
----

.. currentmodule:: pyhs3.data

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Data
   Datum
   PointData
   UnbinnedData
   BinnedData
   GaussianUncertainty

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
   WorkspaceValidationError

Utilities
---------

.. currentmodule:: pyhs3

.. autosummary::
   :toctree: _generated/
   :nosignatures:

    normalization.gauss_legendre_integral

Typing
------

.. currentmodule:: pyhs3.typing

.. autosummary::
   :toctree: _generated/
   :nosignatures:

    TensorVar
