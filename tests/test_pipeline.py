from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from image_context.artifacts import ArtifactRepository
from image_context.models import BoundingBox, Detection, ImageSample, VisualConcept
from image_context.pipeline import ImageContextPipeline


class FakeSampler:
    def sample(self, count: int, seed: int, destination: Path) -> tuple[ImageSample, ...]:
        assert count == 2
        samples = []
        for index in range(count):
            frame_id = f"frame-{index:06d}"
            image_path = destination / frame_id / "image.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))
            samples.append(ImageSample(frame_id, index, seed + index, 10, 8, image_path))
        return tuple(samples)


class FakeVlm:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def generate(self, image_path: Path, prompt: str) -> str:
        self.events.append(f"vlm:{image_path.parent.name}")
        if "physical objects" in prompt:
            concepts = [_concept("door", "metal door")]
            risks = []
        elif "environmental context" in prompt:
            concepts = [_concept("smoke", "visible smoke", region="environmental_region")]
            risks = []
        else:
            concepts = [_concept("cable", "loose cable")]
            risks = [
                {
                    "label": "trip hazard",
                    "severity": "high",
                    "confidence": 0.8,
                    "evidence_queries": ["loose cable"],
                    "rationale": "Visible cable on floor.",
                }
            ]
        return json.dumps(
            {"concepts": concepts, "scene_attributes": {"test": "yes"}, "risks": risks}
        )

    def close(self) -> None:
        self.events.append("close:vlm")


class FakeGrounder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def detect(self, sample: ImageSample, concept: VisualConcept) -> tuple[Detection, ...]:
        self.events.append(f"ground:{sample.frame_id}")
        identifier = concept.grounding_query.replace(" ", "-")
        return (Detection(identifier, concept, 0.9, BoundingBox(1, 1, 7, 6)),)

    def close(self) -> None:
        self.events.append("close:grounder")


class FakeSegmenter:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def segment(
        self, sample: ImageSample, detection: Detection
    ) -> tuple[NDArray[np.bool_], float]:
        self.events.append(f"segment:{sample.frame_id}")
        mask = np.zeros((sample.height, sample.width), dtype=np.bool_)
        mask[1:6, 1:7] = True
        return mask, 0.95

    def close(self) -> None:
        self.events.append("close:segmenter")


class FakeFactory:
    def __init__(self) -> None:
        self.events: list[str] = []

    def create_vlm(self) -> FakeVlm:
        self.events.append("create:vlm")
        return FakeVlm(self.events)

    def create_grounder(self) -> FakeGrounder:
        self.events.append("create:grounder")
        return FakeGrounder(self.events)

    def create_segmenter(self) -> FakeSegmenter:
        self.events.append("create:segmenter")
        return FakeSegmenter(self.events)


class SilentProgress:
    def stage(self, name: str, total: int) -> None:
        pass

    def advance(self, detail: str) -> None:
        pass

    def failures(self, stage: str, frame_ids: Sequence[str]) -> None:
        assert not frame_ids


def test_pipeline_runs_models_sequentially_and_writes_review_artifacts(tmp_path: Path) -> None:
    repository = ArtifactRepository(tmp_path / "run")
    repository.initialize({"fingerprint": "test"}, overwrite=False)
    factory = FakeFactory()
    pipeline = ImageContextPipeline(
        FakeSampler(),
        factory,
        repository,
        SilentProgress(),
        minimum_concept_confidence=0.6,
    )

    outcome = pipeline.execute(sample_size=2, seed=42)

    assert len(outcome.completed_frame_ids) == 2
    assert not outcome.failed_frame_ids
    assert factory.events.index("close:vlm") < factory.events.index("create:grounder")
    assert factory.events.index("close:grounder") < factory.events.index("create:segmenter")
    frame = repository.frames_directory / "frame-000000"
    assert (frame / "vlm_objects.json").is_file()
    assert (frame / "vlm_environment.json").is_file()
    assert (frame / "vlm_risks.json").is_file()
    assert (frame / "result.json").is_file()
    assert (frame / "overlay.png").is_file()
    assert len(tuple((frame / "masks").glob("*.png"))) == 3


def _concept(label: str, query: str, *, region: str = "bounded_object") -> dict[str, object]:
    return {
        "label": label,
        "grounding_query": query,
        "confidence": 0.9,
        "region_kind": region,
        "attributes": {},
    }
