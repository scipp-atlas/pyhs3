.. diataxis: reference
.. status: implemented

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
   metadata.Metadata
   metadata.PackageInfo

Modules
-------

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   compiled
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
   tensorutils
   transpile

Base Classes
------------

.. currentmodule:: pyhs3

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   base.Evaluable
   collections.NamedModel
   collections.NamedCollection
   context.Context

Compiled Likelihoods
--------------------

.. currentmodule:: pyhs3.compiled

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   compile_nll
   default_nll_mode
   CompiledNLL
   BatchedCompiledNLL

Functions
---------

.. currentmodule:: pyhs3.functions

.. autosummary::
   :toctree: _generated/
   :nosignatures:

   Functions
   Function
   BatchableFunction
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
   BatchableDistribution
   LogProbTerms
   GaussianDist
   UniformDist
   PoissonDist
   ExponentialDist
   LogNormalDist
   LandauDist
   MixtureDist
   ProductDist
   HistogramDist
   histogram.HistogramData
   GenericDist
   PolynomialDist
   BernsteinPolyDist
   CrystalBallDist
   AsymmetricCrystalBallDist
   ArgusDist
   RelativisticBreitWignerDist
   physics.RelativisticBreitWignerChannel
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
   GenericExpressionMixin

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
   UnlistedParameterError
   DuplicateEntityError
   custom_error_msg

Utilities
---------

.. currentmodule:: pyhs3

.. autosummary::
   :toctree: _generated/
   :nosignatures:

    normalization.gauss_legendre_integral
    normalization.substitute_variable
    tensorutils.create_bounded_tensor
    tensorutils.ensure_array
    tensorutils.is_scalar_multiplicative_identity
    tensorutils.merge_duplicate_nodes

Typing
------

.. currentmodule:: pyhs3.typing.aliases

.. autosummary::
   :toctree: _generated/
   :nosignatures:

    EntityType
    TensorVar
