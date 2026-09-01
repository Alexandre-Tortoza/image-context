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
_HORIZONTAL_POSITIONS = {"left", "center", "right"}
_DEPTH_POSITIONS = {"foreground", "middle_ground", "background"}
_BOUNDED_ALIASES = {"structural_element", "sign", "object", "foreground_object"}
_DIFFUSE_TERMS = {
    "dust",
    "fire",
    "flood",
    "fog",
    "haze",
    "smoke",
    "steam",
    "water",
}


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
    raw_region_kind = _non_empty_string(item.get("region_kind"), "concept.region_kind")
    horizontal, depth = _parse_spatial(item.get("spatial"))
    region_kind, horizontal, depth, parser_notes = _normalize_region_kind(
        raw_region_kind,
        label,
        query,
        horizontal,
        depth,
    )
    attributes = _string_mapping(item.get("attributes", {}), "concept.attributes")
    return VisualConcept(
        label=label.strip(),
        grounding_query=query.strip().lower(),
        confidence=confidence,
        region_kind=region_kind,
        attributes=attributes,
        source_pass=pass_name,
        detector_query=_detector_query(label),
        horizontal_position=horizontal,
        depth_position=depth,
        parser_notes=parser_notes,
    )


def _parse_spatial(value: object) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    spatial = _object(value, "concept.spatial")
    horizontal = spatial.get("horizontal")
    depth = spatial.get("depth")
    parsed_horizontal = horizontal if isinstance(horizontal, str) else None
    parsed_depth = depth if isinstance(depth, str) else None
    if parsed_horizontal not in _HORIZONTAL_POSITIONS:
        parsed_horizontal = None
    if parsed_depth not in _DEPTH_POSITIONS:
        parsed_depth = None
    return parsed_horizontal, parsed_depth


def _normalize_region_kind(
    raw: str,
    label: str,
    query: str,
    horizontal: str | None,
    depth: str | None,
) -> tuple[RegionKind, str | None, str | None, tuple[str, ...]]:
    normalized = raw.strip().lower()
    notes: list[str] = []
    if normalized in _HORIZONTAL_POSITIONS:
        horizontal = normalized
        normalized = "bounded_object"
        notes.append(f"moved region_kind '{raw}' to spatial.horizontal")
    elif normalized in _DEPTH_POSITIONS:
        depth = normalized
        normalized = "bounded_object"
        notes.append(f"moved region_kind '{raw}' to spatial.depth")
    elif normalized in _BOUNDED_ALIASES:
        normalized = "bounded_object"
        notes.append(f"normalized region_kind '{raw}' to bounded_object")
    elif normalized not in _REGION_KINDS:
        raise ValueError(f"Unsupported concept.region_kind: {raw!r}.")

    text = f"{label} {query}".lower()
    if normalized == "environmental_region" and not any(term in text for term in _DIFFUSE_TERMS):
        normalized = "bounded_object"
        notes.append("normalized non-diffuse environmental_region to bounded_object")
    return cast(RegionKind, normalized), horizontal, depth, tuple(notes)


def _detector_query(label: str) -> str:
    return " ".join(label.strip().lower().split()[:3])


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
