from __future__ import annotations

from image_context.models import VisualConcept
from image_context.pipeline import _unique_detector_queries


def test_unique_detector_queries_keeps_strongest_concept() -> None:
    left = VisualConcept(
        "door",
        "red door on left",
        0.8,
        "bounded_object",
        detector_query="door",
    )
    right = VisualConcept(
        "door",
        "red door on right",
        0.9,
        "bounded_object",
        detector_query="door",
    )

    assert _unique_detector_queries((left, right)) == (right,)
