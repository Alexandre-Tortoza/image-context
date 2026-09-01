from __future__ import annotations

from image_context.consolidation import consolidate_concepts
from image_context.models import VisualConcept, VlmPassResult


def test_consolidation_keeps_strongest_query_and_filters_low_confidence() -> None:
    weak = VisualConcept("door", "metal door", 0.7, "bounded_object")
    strong = VisualConcept("exit door", "metal door", 0.9, "bounded_object", source_pass="risks")
    low = VisualConcept("sign", "exit sign", 0.4, "bounded_object")
    objects = VlmPassResult("objects", (weak, low), {}, (), "{}")
    risks = VlmPassResult("risks", (strong,), {}, (), "{}")

    result = consolidate_concepts((objects, risks), minimum_confidence=0.6)

    assert result == (strong,)
