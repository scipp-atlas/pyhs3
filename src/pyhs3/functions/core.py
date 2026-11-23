"""
HS3 Functions implementation.

Provides classes for handling HS3 functions including product functions,
generic functions with mathematical expressions, and interpolation functions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Self

from pyhs3.base import Evaluable

if TYPE_CHECKING:
    from pyhs3.context import Context
    from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)


class Function(Evaluable, ABC):
    """
    Base class for functions in HS3.

    Provides the foundation for all function implementations,
    handling parameter management and constant generation.

    Inherits parameter processing functionality from Evaluable.
    Subclasses must implement _expression() to define computation logic.
    """


class BatchableFunction(Function, ABC):
    """
    A function type whose instances can be evaluated together, as one family.

    :class:`~pyhs3.Model` groups the instances of a batchable type that feed
    the same likelihood channel into one *family* node of its dependency graph,
    and builds the whole family with a single call to
    :meth:`batch_expressions` instead of one :meth:`~pyhs3.base.Evaluable.expression`
    per instance.  A family's graph does not grow with the number of members,
    which keeps large workspaces quick to compile.  This is the role pyhf's
    combined modifiers (e.g. ``histosys_combined``) play for modifiers of one
    type.

    Subclasses implement :meth:`batch_expressions` in addition to
    ``_expression``; members it cannot batch fall back to their own
    ``expression``.
    """

    @classmethod
    @abstractmethod
    def batch_expressions(
        cls,
        members: Sequence[Self],
        context: Context,
        functions: Mapping[str, Function],
    ) -> dict[str, TensorVar]:
        """
        Build the expression of every member, sharing one computation where possible.

        Args:
            members: Instances of this class, in a fixed order.
            context: Built tensors by name; it holds every name that any
                member's ``parameters`` references.
            functions: Every function of the model by name, for members whose
                inputs are themselves functions.

        Returns:
            dict[str, TensorVar]: For each member name, an expression equal to
            that member's ``expression(context)``.
        """
