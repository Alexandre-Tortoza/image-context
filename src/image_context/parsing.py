"""Validation of the JSON contracts returned by the three VLM passes."""

from __future__ import annotations

import json
import re
from typing import cast

from image_context.models import (
    PassName,
    RegionKind,
    RiskFinding,
    RiskSeverity,
    VisualConcept,
    VlmPassResult,
)

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_REGION_KINDS = {"bounded_object", "environmental_region"}
_SEVERITIES = {"low", "medium", "high", "critical"}


def parse_vlm_response(raw_response: str, pass_name: PassName) -> VlmPassResult:
    """Parse and strictly validate one pass response while preserving raw text."""
    try:
        payload = json.loads(_CODE_FENCE.sub("", raw_response.strip()))
    except json.JSONDecodeError as error:
        raise ValueError(f"VLM {pass_name} response is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"VLM {pass_name} response must be a JSON object.")

    rejected_entries: list[str] = []
    concepts: list[VisualConcept] = []
    for index, item in enumerate(_list_field(payload, "concepts")):
        try:
            concepts.append(_parse_concept(item, pass_name))
        except ValueError as error:
            rejected_entries.append(f"concept[{index}]: {error}")
    raw_attributes = payload.get("scene_attributes", {})
    scene_attributes = _string_mapping(raw_attributes, "scene_attributes")
    risks: list[RiskFinding] = []
    for index, item in enumerate(_list_field(payload, "risks")):
        try:
            risks.append(_parse_risk(item))
        except ValueError as error:
            rejected_entries.append(f"risk[{index}]: {error}")
    return VlmPassResult(
        pass_name,
        tuple(concepts),
        scene_attributes,
        tuple(risks),
        raw_response,
        tuple(rejected_entries),
    )


def _parse_concept(value: object, pass_name: PassName) -> VisualConcept:
    item = _object(value, "concept")
    label = _non_empty_string(item.get("label"), "concept.label")
    query = _non_empty_string(item.get("grounding_query"), "concept.grounding_query")
    confidence = _confidence(item.get("confidence"), "concept.confidence")
    region_kind = _non_empty_string(item.get("region_kind"), "concept.region_kind")
    if region_kind not in _REGION_KINDS:
        raise ValueError(f"Unsupported concept.region_kind: {region_kind!r}.")
    attributes = _string_mapping(item.get("attributes", {}), "concept.attributes")
    return VisualConcept(
        label=label.strip(),
        grounding_query=query.strip().lower(),
        confidence=confidence,
        region_kind=cast(RegionKind, region_kind),
        attributes=attributes,
        source_pass=pass_name,
    )


def _parse_risk(value: object) -> RiskFinding:
    item = _object(value, "risk")
    severity = _non_empty_string(item.get("severity"), "risk.severity")
    if severity not in _SEVERITIES:
        raise ValueError(f"Unsupported risk.severity: {severity!r}.")
    raw_queries = _list_value(item.get("evidence_queries"), "risk.evidence_queries")
    queries = tuple(_non_empty_string(query, "risk.evidence_queries[]") for query in raw_queries)
    return RiskFinding(
        label=_non_empty_string(item.get("label"), "risk.label"),
        severity=cast(RiskSeverity, severity),
        confidence=_confidence(item.get("confidence"), "risk.confidence"),
        evidence_queries=queries,
        rationale=_non_empty_string(item.get("rationale"), "risk.rationale"),
    )


def _list_field(payload: dict[str, object], name: str) -> list[object]:
    return _list_value(payload.get(name, []), name)


def _list_value(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return cast(dict[str, object], value)


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _confidence(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return result


def _string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{name} must be an object containing only string values.")
    return cast(dict[str, str], dict(value))
