from __future__ import annotations

from typing import cast

from image_context.models import VlmPassResult
from image_context.pipeline import ImageContextPipeline


def test_empty_concepts_receive_configured_fallback_queries() -> None:
    pipeline = ImageContextPipeline(
        sampler=cast(object, None),
        model_factory=cast(object, None),
        artifacts=cast(object, None),
        progress=cast(object, None),
        minimum_concept_confidence=0.6,
        indoor_fallback_queries=("door",),
        outdoor_fallback_queries=("tree",),
    )

    environment = VlmPassResult(
        "environment", (), {"environment": "outdoor open field"}, (), "{}"
    )
    concepts = pipeline._with_fallback((), (environment,))

    assert [concept.detector_query for concept in concepts] == ["tree"]
    assert all("fallback" in concept.parser_notes[0] for concept in concepts)
