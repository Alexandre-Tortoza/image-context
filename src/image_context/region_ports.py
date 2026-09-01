"""Replaceable ports used by the region-first perception strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from image_context.models import ImageSample
from image_context.region_models import VisualRegion


class RegionDiscoverer(Protocol):
    """Discover class-agnostic visual regions without VLM-proposed labels."""

    def discover(self, sample: ImageSample) -> tuple[VisualRegion, ...]: ...

    def close(self) -> None: ...


class DenseVisualFeatureExtractor(Protocol):
    """Aggregate dense visual features into one embedding per region mask."""

    def extract(
        self, sample: ImageSample, regions: tuple[VisualRegion, ...]
    ) -> dict[str, NDArray[np.float32]]: ...

    def close(self) -> None: ...


class RegionVlmBackend(Protocol):
    """Analyze global images and local region composites with structured prompts."""

    def generate(self, image_path: Path, prompt: str) -> str: ...

    def close(self) -> None: ...


class RegionFirstModelFactory(Protocol):
    """Construct each heavy backend only when its sequential stage starts."""

    def create_discoverer(self) -> RegionDiscoverer: ...

    def create_feature_extractor(self) -> DenseVisualFeatureExtractor: ...

    def create_vlm(self) -> RegionVlmBackend: ...


class ImageAnalysisStrategy(Protocol):
    """Common strategy interface for independently persisted image analysis pipelines."""

    @property
    def strategy_name(self) -> str: ...

    def analyze(self, sample: ImageSample) -> object: ...
