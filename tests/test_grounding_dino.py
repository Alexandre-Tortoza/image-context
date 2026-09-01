from __future__ import annotations

from image_context.grounding import non_max_suppression
from image_context.models import BoundingBox


def test_non_max_suppression_keeps_best_overlap_and_distinct_box() -> None:
    candidates = (
        (BoundingBox(0, 0, 10, 10), 0.9),
        (BoundingBox(1, 1, 11, 11), 0.8),
        (BoundingBox(20, 20, 30, 30), 0.7),
    )

    result = non_max_suppression(candidates, iou_threshold=0.5)

    assert result == (candidates[0], candidates[2])
