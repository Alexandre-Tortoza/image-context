"""Model-independent filtering for grounded bounding boxes."""

from __future__ import annotations

from image_context.models import BoundingBox


def non_max_suppression(
    candidates: tuple[tuple[BoundingBox, float], ...], iou_threshold: float
) -> tuple[tuple[BoundingBox, float], ...]:
    """Suppress lower-confidence boxes that overlap an already-kept box."""
    kept: list[tuple[BoundingBox, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
        overlaps = (
            intersection_over_union(candidate[0], existing[0]) >= iou_threshold
            for existing in kept
        )
        if not any(overlaps):
            kept.append(candidate)
    return tuple(kept)


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate intersection over union for two pixel-space boxes."""
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
    second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
    union = first_area + second_area - intersection
    return 0.0 if union == 0.0 else intersection / union
