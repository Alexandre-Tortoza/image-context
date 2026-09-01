"""Deterministic consolidation of concepts proposed by multiple VLM passes."""

from __future__ import annotations

from image_context.models import VisualConcept, VlmPassResult


def consolidate_concepts(
    pass_results: tuple[VlmPassResult, ...], minimum_confidence: float
) -> tuple[VisualConcept, ...]:
    """Deduplicate concepts by grounding query and keep the strongest evidence."""
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("Minimum confidence must be between 0.0 and 1.0.")
    by_query: dict[str, VisualConcept] = {}
    for result in pass_results:
        for concept in result.concepts:
            if concept.confidence < minimum_confidence:
                continue
            key = " ".join(concept.grounding_query.lower().split())
            current = by_query.get(key)
            if current is None or concept.confidence > current.confidence:
                by_query[key] = concept
    return tuple(
        sorted(by_query.values(), key=lambda item: (-item.confidence, item.grounding_query))
    )
