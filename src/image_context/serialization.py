"""JSON serialization for resumable pipeline artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from image_context.models import (
    BoundingBox,
    Detection,
    ImageSample,
    PassName,
    RiskFinding,
    VisualConcept,
    VlmPassResult,
)


def sample_to_dict(sample: ImageSample) -> dict[str, Any]:
    """Serialize an extracted sample."""
    return {
        "frame_id": sample.frame_id,
        "source_index": sample.source_index,
        "timestamp_ns": sample.timestamp_ns,
        "width": sample.width,
        "height": sample.height,
        "image_path": str(sample.image_path),
    }


def concept_to_dict(concept: VisualConcept) -> dict[str, Any]:
    """Serialize one visual concept."""
    return {
        "label": concept.label,
        "grounding_query": concept.grounding_query,
        "confidence": concept.confidence,
        "region_kind": concept.region_kind,
        "attributes": concept.attributes,
        "source_pass": concept.source_pass,
        "detector_query": concept.detector_query,
        "spatial": {
            "horizontal": concept.horizontal_position,
            "depth": concept.depth_position,
        },
        "parser_notes": list(concept.parser_notes),
    }


def concept_from_dict(payload: dict[str, Any]) -> VisualConcept:
    """Deserialize one visual concept from a trusted artifact."""
    return VisualConcept(
        label=payload["label"],
        grounding_query=payload["grounding_query"],
        confidence=float(payload["confidence"]),
        region_kind=payload["region_kind"],
        attributes=dict(payload["attributes"]),
        source_pass=payload["source_pass"],
        detector_query=payload.get("detector_query"),
        horizontal_position=payload.get("spatial", {}).get("horizontal"),
        depth_position=payload.get("spatial", {}).get("depth"),
        parser_notes=tuple(payload.get("parser_notes", [])),
    )


def vlm_result_to_dict(result: VlmPassResult, *, include_raw: bool = True) -> dict[str, Any]:
    """Serialize one validated VLM pass result."""
    payload: dict[str, Any] = {
        "pass_name": result.pass_name,
        "concepts": [concept_to_dict(concept) for concept in result.concepts],
        "scene_attributes": result.scene_attributes,
        "risks": [
            {
                "label": risk.label,
                "severity": risk.severity,
                "confidence": risk.confidence,
                "evidence_queries": list(risk.evidence_queries),
                "rationale": risk.rationale,
            }
            for risk in result.risks
        ],
        "rejected_entries": list(result.rejected_entries),
    }
    if include_raw:
        payload["raw_response"] = result.raw_response
    return payload


def vlm_result_from_dict(payload: dict[str, Any]) -> VlmPassResult:
    """Deserialize one previously validated VLM pass result."""
    return VlmPassResult(
        pass_name=cast(PassName, payload["pass_name"]),
        concepts=tuple(concept_from_dict(item) for item in payload["concepts"]),
        scene_attributes=dict(payload["scene_attributes"]),
        risks=tuple(
            RiskFinding(
                label=item["label"],
                severity=item["severity"],
                confidence=float(item["confidence"]),
                evidence_queries=tuple(item["evidence_queries"]),
                rationale=item["rationale"],
            )
            for item in payload["risks"]
        ),
        raw_response=str(payload.get("raw_response", "")),
        rejected_entries=tuple(payload.get("rejected_entries", [])),
    )


def detection_to_dict(detection: Detection) -> dict[str, Any]:
    """Serialize one grounded detection."""
    box = detection.bounding_box
    return {
        "detection_id": detection.detection_id,
        "concept": concept_to_dict(detection.concept),
        "confidence": detection.confidence,
        "bounding_box": [box.x_min, box.y_min, box.x_max, box.y_max],
    }


def detection_from_dict(payload: dict[str, Any]) -> Detection:
    """Deserialize one grounded detection."""
    box = payload["bounding_box"]
    return Detection(
        detection_id=payload["detection_id"],
        concept=concept_from_dict(payload["concept"]),
        confidence=float(payload["confidence"]),
        bounding_box=BoundingBox(*(float(value) for value in box)),
    )


def path_strings(paths: tuple[Path, ...]) -> list[str]:
    """Convert paths for JSON output."""
    return [str(path) for path in paths]
