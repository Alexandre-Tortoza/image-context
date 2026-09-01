from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from image_context.models import BoundingBox, ImageSample
from image_context.region_models import VisualRegion
from image_context.region_parsing import parse_global_context, parse_region_interpretation
from image_context.region_pipeline import RegionFirstContextPipeline


class FakeDiscoverer:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def discover(self, sample: ImageSample) -> tuple[VisualRegion, ...]:
        self._events.append("discover")
        left = np.zeros((sample.height, sample.width), dtype=np.bool_)
        left[1:6, 1:5] = True
        right = np.zeros((sample.height, sample.width), dtype=np.bool_)
        right[1:6, 5:9] = True
        return (
            VisualRegion("region-000", left, BoundingBox(1, 1, 5, 6), 0.95, "fake-sam2"),
            VisualRegion("region-001", right, BoundingBox(5, 1, 9, 6), 0.90, "fake-sam2"),
        )

    def close(self) -> None:
        self._events.append("close:discoverer")


class FakeFeatures:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def extract(
        self, sample: ImageSample, regions: tuple[VisualRegion, ...]
    ) -> dict[str, NDArray[np.float32]]:
        del sample
        self._events.append("features")
        return {
            region.region_id: np.asarray([index, index + 1, index + 2], dtype=np.float32)
            for index, region in enumerate(regions)
        }

    def close(self) -> None:
        self._events.append("close:features")


class FakeRegionVlm:
    def __init__(self, events: list[str], *, fail_second: bool) -> None:
        self._events = events
        self._fail_second = fail_second

    def generate(self, image_path: Path, prompt: str) -> str:
        if "complete image as scene context" in prompt:
            self._events.append("vlm:global")
            return json.dumps(
                {
                    "scene_type": "corridor",
                    "scene_description": "An indoor corridor.",
                    "global_attributes": ["artificial lighting"],
                    "potential_hazards": [],
                }
            )
        region_id = "region-001" if "region-001" in prompt else "region-000"
        self._events.append(f"vlm:{region_id}")
        response_id = (
            "wrong-region" if self._fail_second and region_id == "region-001" else region_id
        )
        return json.dumps(
            {
                "region_id": response_id,
                "label": "door",
                "description": "A visible door panel.",
                "attributes": ["metal"],
                "condition": "closed",
                "confidence": 0.88,
                "candidate_relations": ["part of corridor"],
            }
        )

    def close(self) -> None:
        self._events.append("close:vlm")


class FakeRegionFactory:
    def __init__(self, *, fail_second: bool = False) -> None:
        self.events: list[str] = []
        self._fail_second = fail_second

    def create_discoverer(self) -> FakeDiscoverer:
        self.events.append("create:discoverer")
        return FakeDiscoverer(self.events)

    def create_feature_extractor(self) -> FakeFeatures:
        self.events.append("create:features")
        return FakeFeatures(self.events)

    def create_vlm(self) -> FakeRegionVlm:
        self.events.append("create:vlm")
        return FakeRegionVlm(self.events, fail_second=self._fail_second)


def test_region_first_preserves_region_embedding_association_and_artifacts(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    factory = FakeRegionFactory()
    output = tmp_path / "run" / "region-first"
    pipeline = RegionFirstContextPipeline(factory, output)

    outcome = pipeline.analyze(sample)

    assert outcome.discovered_region_count == 2
    assert len(outcome.semantic_regions) == 2
    assert outcome.vlm_call_count == 3
    first = outcome.semantic_regions[0]
    assert first.region_id == "region-000"
    assert np.array_equal(np.load(first.visual_embedding_path), np.asarray([0, 1, 2]))
    assert first.provenance["geometry"] == "fake-sam2"
    assert (output / "semantic_regions.json").is_file()
    assert (output / "global_context.json").is_file()
    assert (output / "region_overlay.png").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "complete.json").is_file()
    assert (output / "regions" / "region-000" / "mask.png").is_file()
    assert (output / "regions" / "region-000" / "context.png").is_file()


def test_region_first_loads_models_sequentially(tmp_path: Path) -> None:
    factory = FakeRegionFactory()

    RegionFirstContextPipeline(factory, tmp_path / "output").analyze(_sample(tmp_path))

    assert factory.events.index("close:discoverer") < factory.events.index("create:features")
    assert factory.events.index("close:features") < factory.events.index("create:vlm")


def test_region_first_resumes_completed_stages_without_loading_models(tmp_path: Path) -> None:
    output = tmp_path / "output"
    stage_fingerprints = {
        "region-discovery": "discovery-v1",
        "dense-features": "features-v1",
        "vlm-global-local": "vlm-v1",
    }
    RegionFirstContextPipeline(
        FakeRegionFactory(), output, stage_fingerprints
    ).analyze(_sample(tmp_path))
    resumed_factory = FakeRegionFactory()

    outcome = RegionFirstContextPipeline(
        resumed_factory, output, stage_fingerprints
    ).analyze(_sample(tmp_path))

    assert resumed_factory.events == []
    assert outcome.vlm_call_count == 0
    assert len(outcome.semantic_regions) == 2


def test_malformed_local_response_only_discards_its_region(tmp_path: Path) -> None:
    output = tmp_path / "output"

    outcome = RegionFirstContextPipeline(
        FakeRegionFactory(fail_second=True), output
    ).analyze(_sample(tmp_path))

    assert [region.region_id for region in outcome.semantic_regions] == ["region-000"]
    assert outcome.discarded_region_count == 1
    payload = json.loads((output / "semantic_regions.json").read_text(encoding="utf-8"))
    assert payload["failed_regions"][0]["region_id"] == "region-001"
    assert (output / "regions" / "region-001" / "vlm_raw.txt").is_file()


def test_strategy_output_does_not_modify_baseline_directory(tmp_path: Path) -> None:
    baseline_marker = tmp_path / "run" / "baseline" / "result.json"
    baseline_marker.parent.mkdir(parents=True)
    baseline_marker.write_text("baseline", encoding="utf-8")

    RegionFirstContextPipeline(
        FakeRegionFactory(), tmp_path / "run" / "region-first"
    ).analyze(_sample(tmp_path))

    assert baseline_marker.read_text(encoding="utf-8") == "baseline"


def test_region_parser_rejects_changed_region_id() -> None:
    raw = json.dumps(
        {
            "region_id": "region-999",
            "label": "door",
            "description": "door",
            "attributes": [],
            "condition": "unknown",
            "confidence": 0.5,
            "candidate_relations": [],
        }
    )

    try:
        parse_region_interpretation(raw, "region-000")
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Expected a mismatched region ID to fail.")


def test_global_parser_accepts_fenced_json() -> None:
    context = parse_global_context(
        '```json\n{"scene_type":"room","scene_description":"A room",'
        '"global_attributes":[],"potential_hazards":[]}\n```'
    )

    assert context.scene_type == "room"


def _sample(tmp_path: Path) -> ImageSample:
    image_path = tmp_path / "source.png"
    assert cv2.imwrite(str(image_path), np.full((8, 10, 3), 80, dtype=np.uint8))
    return ImageSample("image", 0, 0, 10, 8, image_path)
