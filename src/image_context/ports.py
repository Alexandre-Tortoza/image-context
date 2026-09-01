"""Small dependency-inversion seams for datasets and perception models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from image_context.models import Detection, ImageSample, VisualConcept


class ImageSampler(Protocol):
    """Selects and materializes images from a dataset source."""

    def sample(self, count: int, seed: int, destination: Path) -> tuple[ImageSample, ...]: ...


class VlmBackend(Protocol):
    """Generates a text response from one image and prompt."""

    def generate(self, image_path: Path, prompt: str) -> str: ...

    def close(self) -> None: ...


class GroundingBackend(Protocol):
    """Grounds one visual concept into zero or more image regions."""

    def detect(self, sample: ImageSample, concept: VisualConcept) -> tuple[Detection, ...]: ...

    def close(self) -> None: ...


class SegmentationBackend(Protocol):
    """Produces a boolean mask from a detected box."""

    def segment(
        self, sample: ImageSample, detection: Detection
    ) -> tuple[NDArray[np.bool_], float]: ...

    def close(self) -> None: ...


class ModelFactory(Protocol):
    """Builds heavy models only when their full-dataset pass starts."""

    def create_vlm(self) -> VlmBackend: ...

    def create_grounder(self) -> GroundingBackend: ...

    def create_segmenter(self) -> SegmentationBackend: ...


class ProgressReporter(Protocol):
    """Receives concise stage progress updates."""

    def stage(self, name: str, total: int) -> None: ...

    def advance(self, detail: str) -> None: ...

    def failures(self, stage: str, frame_ids: Sequence[str]) -> None: ...
