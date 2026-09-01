"""Strict but region-local parsing for global and local VLM responses."""

from __future__ import annotations

import json
import re
from typing import cast

from image_context.region_models import GlobalSceneContext, RegionInterpretation

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_global_context(raw_response: str) -> GlobalSceneContext:
    """Validate one global scene response."""
    payload = _payload(raw_response, "global context")
    return GlobalSceneContext(
        scene_type=_string(payload, "scene_type"),
        scene_description=_string(payload, "scene_description"),
        global_attributes=_strings(payload, "global_attributes"),
        potential_hazards=_strings(payload, "potential_hazards"),
        raw_response=raw_response,
    )


def parse_region_interpretation(
    raw_response: str, expected_region_id: str
) -> RegionInterpretation:
    """Validate local context and enforce propagation of the observed region ID."""
    payload = _payload(raw_response, "region context")
    region_id = _string(payload, "region_id")
    if region_id != expected_region_id:
        raise ValueError(
            f"Region response ID '{region_id}' does not match '{expected_region_id}'."
        )
    confidence = payload.get("confidence")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("region confidence must be numeric.")
    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("region confidence must be between 0.0 and 1.0.")
    return RegionInterpretation(
        region_id=region_id,
        label=_string(payload, "label"),
        description=_string(payload, "description"),
        attributes=_strings(payload, "attributes"),
        condition=_string(payload, "condition"),
        confidence=confidence_value,
        candidate_relations=_strings(payload, "candidate_relations"),
        raw_response=raw_response,
    )


def _payload(raw_response: str, name: str) -> dict[str, object]:
    try:
        value = json.loads(_CODE_FENCE.sub("", raw_response.strip()))
    except json.JSONDecodeError as error:
        raise ValueError(f"VLM {name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"VLM {name} must be a JSON object.")
    return cast(dict[str, object], value)


def _string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _strings(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings.")
    return tuple(value)
