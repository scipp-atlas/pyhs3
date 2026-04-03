"""
HistFactory data classes.

Provides data structures for HistFactory distributions including sample data
with bin contents and errors.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class SampleData(BaseModel):
    """Sample data containing bin contents and optional per-bin uncertainties.

    This class represents the binned values for a single sample in a HistFactory-
    style model, along with optional statistical uncertainties ("errors") per bin.

    .. note::

        - The ``errors`` field is optional in serialized form. If omitted, it is
          interpreted as an array of zeros with the same length as ``contents``.
          This follows the convention used in ROOT HistFactory/HS3 workflows,
          where some samples may not carry explicit bin-by-bin uncertainties
          (e.g. when BBlight is not applied).
        - Internally, errors are always materialized and accessible via the
          ``errors`` property.
        - If provided, ``errors`` must have the same length as ``contents``.

    Parameters:
        contents : list[float]
            The bin contents (yields) for the sample.
        errors : list[float] | None
            Optional serialized representation of per-bin uncertainties. This is
            aliased to ``"errors"`` when exporting. If ``None``, errors are implicitly
            treated as zeros and omitted from serialization.

    Raises:
        ValueError: If ``contents`` and ``errors`` are both provided and have different lengths.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    contents: list[float]
    v_errors: list[float] | None = Field(
        default=None, alias="errors", repr=False, exclude_if=lambda v: v is None
    )
    _errors: list[float] = PrivateAttr()

    @model_validator(mode="after")
    def set_default_and_validate(self) -> SampleData:
        """Ensure contents and errors have same length."""
        if self.v_errors is None:
            self._errors = [0.0] * len(self.contents)
        else:
            self._errors = self.v_errors

            if len(self.contents) != len(self.v_errors):
                msg = f"Sample data contents ({len(self.contents)}) and errors ({len(self.v_errors)}) must have same length"
                raise ValueError(msg)
        return self

    @property
    def errors(self) -> list[float]:
        """Return the per-bin uncertainties for the sample.

        This property always returns a list of the same length as ``contents``.

        - If uncertainties were explicitly provided, they are returned as-is.
        - If the ``errors`` field was omitted during initialization (e.g. in HS3
          JSON where missing errors imply zeros), this returns a zero-filled
          list.

        Returns:
            list[float]: The per-bin uncertainties corresponding to ``contents``.
        """
        return self._errors


__all__ = ["SampleData"]
