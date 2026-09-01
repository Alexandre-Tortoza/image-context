from __future__ import annotations

import json

from image_context.parsing import parse_vlm_response


def test_parse_complete_risk_response() -> None:
    raw = json.dumps(
        {
            "concepts": [
                {
                    "label": "cable",
                    "grounding_query": "loose cable",
                    "confidence": 0.82,
                    "region_kind": "bounded_object",
                    "attributes": {"position": "floor"},
                }
            ],
            "scene_attributes": {"visibility": "low"},
            "risks": [
                {
                    "label": "trip hazard",
                    "severity": "high",
                    "confidence": 0.8,
                    "evidence_queries": ["loose cable"],
                    "rationale": "A cable crosses the visible floor.",
                }
            ],
        }
    )

    result = parse_vlm_response(raw, "risks")

    assert result.concepts[0].grounding_query == "loose cable"
    assert result.risks[0].severity == "high"
    assert result.raw_response == raw


def test_parse_audits_invalid_entry_without_discarding_frame() -> None:
    raw = json.dumps(
        {
            "concepts": [
                {
                    "label": "danger",
                    "grounding_query": "danger",
                    "confidence": 0.9,
                    "region_kind": "abstract_risk",
                    "attributes": {},
                }
            ],
            "scene_attributes": {},
            "risks": [],
        }
    )

    result = parse_vlm_response(raw, "risks")

    assert not result.concepts
    assert "region_kind" in result.rejected_entries[0]
