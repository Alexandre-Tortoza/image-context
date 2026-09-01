"""Data contracts shared by the image-context pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PassName = Literal["objects", "environment", "risks"]
RegionKind = Literal["bounded_object", "environmental_region"]
RiskSeverity = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class ImageSample:
    """One image extracted from a source dataset."""

    frame_id: str
    source_index: int
    timestamp_ns: int
    width: int
    height: int
    image_path: Path


@dataclass(frozen=True)
class VisualConcept:
    """A visible concept that can be submitted to an open-vocabulary detector."""

    label: str
    grounding_query: str
    confidence: float
    region_kind: RegionKind
    attributes: dict[str, str] = field(default_factory=dict)
    source_pass: PassName = "objects"


@dataclass(frozen=True)
class RiskFinding:
    """A contextual risk claim and the visible evidence supporting it."""

    label: str
    severity: RiskSeverity
    confidence: float
    evidence_queries: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class VlmPassResult:
    """Validated structured output from one VLM pass."""

    pass_name: PassName
    concepts: tuple[VisualConcept, ...]
    scene_attributes: dict[str, str]
    risks: tuple[RiskFinding, ...]
    raw_response: str
    rejected_entries: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-aligned bounding box in xyxy format."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Bounding box must have positive area.")


@dataclass(frozen=True)
class Detection:
    """A grounded region for one consolidated visual concept."""

    detection_id: str
    concept: VisualConcept
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class Segmentation:
    """Metadata for one mask persisted by the segmentation stage."""

    detection_id: str
    confidence: float
    mask_path: Path


JsonObject = dict[str, Any]
