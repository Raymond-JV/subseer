"""Schema for what the LLM discovers.

Themes come in two kinds:

- ``enumerate``  — a finite, knowledge-driven set (e.g. "other Greek gods").
                   The model lists plausible new members directly in ``candidates``.
- ``template``   — a combinatorial structural pattern. The model does NOT list
                   the combinations; it returns a ``template`` string with
                   ``{placeholder}`` slots and a spec for each slot. Code expands it.

Slots keep the schema flat (a list, not a dict keyed by slot name) so it stays
compatible with the API's structured-output constraints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Slot(BaseModel):
    """One {placeholder} in a template."""

    name: str = Field(description="Placeholder name as it appears in the template, e.g. 'n'.")
    kind: Literal["range", "enum"]

    # enum slots
    values: list[str] = Field(
        default_factory=list,
        description="Allowed values, ordered most-likely first. Used when kind='enum'.",
    )

    # range slots (numeric)
    min: int = Field(default=0, description="Inclusive start. Used when kind='range'.")
    max: int = Field(default=0, description="Inclusive end. Used when kind='range'.")
    pad: int = Field(default=0, description="Zero-pad width, e.g. 2 -> 01. 0 = no padding.")
    step: int = Field(default=1, description="Increment between values.")


class Theme(BaseModel):
    """One naming convention the model found."""

    name: str = Field(description="Short label for the convention.")
    description: str = Field(description="The rule, in one or two sentences.")
    evidence: list[str] = Field(
        description="Actual subdomains from the input that exhibit this pattern. Required."
    )
    kind: Literal["enumerate", "template"]

    # kind == "enumerate"
    candidates: list[str] = Field(
        default_factory=list,
        description="Plausible NEW members not already in the input. Used when kind='enumerate'.",
    )

    # kind == "template"
    template: str = Field(
        default="",
        description="String with {slot} placeholders, e.g. '{name}{n}.example.com'. kind='template'.",
    )
    slots: list[Slot] = Field(
        default_factory=list,
        description="Slot specs matching the placeholders in template. Used when kind='template'.",
    )


class SlotSpec(BaseModel):
    """One placeholder's concrete value list, for enrichment output."""

    name: str = Field(description="Placeholder name as it appears in the template, e.g. 's1'.")
    values: list[str] = Field(
        default_factory=list,
        description="Concrete values for this slot, most-likely first.",
    )


class EnrichedTheme(BaseModel):
    """A template whose slots the model filled with knowledge-expanded values.

    Either an EXISTING mined template with more values, or a NEW template the
    model discovered (a latent/implied dimension mining could not detect).
    """

    template: str = Field(
        description="Template string with {sN} placeholders, ending in the apex, "
        "e.g. '{s1}.dleague.example.com'."
    )
    slots: list[SlotSpec] = Field(
        default_factory=list, description="One entry per {placeholder} in template."
    )
    label: str = Field(default="", description="Short human label for the pattern (optional).")
    novel: bool = Field(
        default=False,
        description="True if this template/dimension is newly discovered (not in the mined set).",
    )


class Enrichment(BaseModel):
    """Top-level structured response from an enrich call."""

    themes: list[EnrichedTheme]
