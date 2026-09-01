"""Structured contracts for class-agnostic region-first perception."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from image_context.models import BoundingBox


@dataclass(frozen=True)
class VisualRegion:
    """Observed image geometry discovered before semantic interpretation."""

    region_id: str
    mask: NDArray[np.bool_]
    bounding_box: BoundingBox
    confidence: float
    source: str


@dataclass(frozen=True)
class GlobalSceneContext:
    """Global VLM interpretation that does not decide region geometry."""

    scene_type: str
    scene_description: str
    global_attributes: tuple[str, ...]
    potential_hazards: tuple[str, ...]
    raw_response: str


@dataclass(frozen=True)
class RegionInterpretation:
    """Validated local interpretation tied to one observed region ID."""

    region_id: str
    label: str
    description: str
    attributes: tuple[str, ...]
    condition: str
    confidence: float
    candidate_relations: tuple[str, ...]
    raw_response: str


@dataclass(frozen=True)
class SemanticRegion:
    """Reusable 2D semantic region with geometry, context, embedding, and provenance."""

    region_id: str
    mask_path: Path
    bounding_box: BoundingBox
    geometry_confidence: float
    label: str
    description: str
    attributes: tuple[str, ...]
    condition: str
    confidence: float
    visual_embedding_path: Path
    embedding_dimension: int
    candidate_relations: tuple[str, ...]
    provenance: dict[str, str]


@dataclass(frozen=True)
class RegionFirstOutcome:
    """Summary and semantic regions produced for one image."""

    semantic_regions: tuple[SemanticRegion, ...]
    discovered_region_count: int
    discarded_region_count: int
    vlm_call_count: int
    output_directory: Path
