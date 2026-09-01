"""Stage-oriented orchestration with one heavy model resident at a time."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2

from image_context.artifacts import ArtifactRepository
from image_context.consolidation import consolidate_concepts
from image_context.models import Detection, ImageSample, PassName, VisualConcept, VlmPassResult
from image_context.parsing import parse_vlm_response
from image_context.ports import ImageSampler, ModelFactory, ProgressReporter
from image_context.prompts import build_prompt

_PASS_NAMES: tuple[PassName, ...] = ("objects", "environment", "risks")


@dataclass(frozen=True)
class PipelineOutcome:
    """Summary of one pipeline execution."""

    samples: tuple[ImageSample, ...]
    completed_frame_ids: tuple[str, ...]
    failed_frame_ids: tuple[str, ...]


class ImageContextPipeline:
    """Run extraction, contextual analysis, grounding, and segmentation in full passes."""

    strategy_name = "baseline"

    def __init__(
        self,
        sampler: ImageSampler,
        model_factory: ModelFactory,
        artifacts: ArtifactRepository,
        progress: ProgressReporter,
        *,
        minimum_concept_confidence: float,
        indoor_fallback_queries: tuple[str, ...] = (),
        outdoor_fallback_queries: tuple[str, ...] = (),
    ) -> None:
        self._sampler = sampler
        self._model_factory = model_factory
        self._artifacts = artifacts
        self._progress = progress
        self._minimum_concept_confidence = minimum_concept_confidence
        self._indoor_fallback_queries = indoor_fallback_queries
        self._outdoor_fallback_queries = outdoor_fallback_queries
        self._stage_seconds: dict[str, float] = {}

    @property
    def stage_seconds(self) -> dict[str, float]:
        """Return timings from the latest execution for experimental comparison."""
        return dict(self._stage_seconds)

    def execute(self, sample_size: int, seed: int) -> PipelineOutcome:
        """Execute all stages and isolate model errors to their source frame."""
        self._stage_seconds = {}
        self._progress.stage("sample-images", sample_size)
        started = time.perf_counter()
        samples = self._sampler.sample(sample_size, seed, self._artifacts.frames_directory)
        self._artifacts.write_selection(samples)
        self._stage_seconds["sample_images"] = time.perf_counter() - started
        for sample in samples:
            self._progress.advance(sample.frame_id)

        started = time.perf_counter()
        pass_results = self._run_vlm_pass(samples)
        self._stage_seconds["vlm"] = time.perf_counter() - started
        return self.execute_from_results(samples, pass_results)

    def analyze(self, sample: ImageSample) -> PipelineOutcome:
        """Analyze one prepared image through the unchanged baseline model sequence."""
        self._stage_seconds = {}
        samples = (sample,)
        self._artifacts.write_selection(samples)
        started = time.perf_counter()
        pass_results = self._run_vlm_pass(samples)
        self._stage_seconds["vlm"] = time.perf_counter() - started
        return self.execute_from_results(samples, pass_results)

    def execute_from_results(
        self,
        samples: tuple[ImageSample, ...],
        pass_results: dict[str, tuple[VlmPassResult, ...]],
    ) -> PipelineOutcome:
        """Run consolidation, grounding, and segmentation from persisted VLM responses."""
        started = time.perf_counter()
        concepts = {
            frame_id: self._with_fallback(
                consolidate_concepts(results, self._minimum_concept_confidence), results
            )
            for frame_id, results in pass_results.items()
        }
        for sample in samples:
            if sample.frame_id in concepts:
                self._artifacts.write_concepts(sample, concepts[sample.frame_id])
        self._stage_seconds["consolidation"] = time.perf_counter() - started

        started = time.perf_counter()
        detections = self._run_grounding_pass(samples, concepts)
        self._stage_seconds["grounding"] = time.perf_counter() - started
        started = time.perf_counter()
        completed = self._run_segmentation_pass(samples, pass_results, concepts, detections)
        self._stage_seconds["segmentation"] = time.perf_counter() - started
        failed = tuple(sample.frame_id for sample in samples if sample.frame_id not in completed)
        return PipelineOutcome(samples, tuple(completed), failed)

    def _with_fallback(
        self,
        concepts: tuple[VisualConcept, ...],
        pass_results: tuple[VlmPassResult, ...],
    ) -> tuple[VisualConcept, ...]:
        if concepts:
            return concepts
        environment_text = " ".join(
            value
            for result in pass_results
            for key, value in result.scene_attributes.items()
            if key in {"environment", "lighting"}
        ).lower()
        is_outdoor = "outdoor" in environment_text or "open field" in environment_text
        queries = (
            self._outdoor_fallback_queries if is_outdoor else self._indoor_fallback_queries
        )
        return tuple(
            VisualConcept(
                label=query,
                grounding_query=query,
                detector_query=query,
                confidence=self._minimum_concept_confidence,
                region_kind="bounded_object",
                parser_notes=(
                    f"configured {'outdoor' if is_outdoor else 'indoor'} fallback after "
                    "empty VLM result",
                ),
            )
            for query in queries
        )

    def _run_vlm_pass(
        self, samples: tuple[ImageSample, ...]
    ) -> dict[str, tuple[VlmPassResult, ...]]:
        self._progress.stage("vlm-three-pass-context", len(samples) * len(_PASS_NAMES))
        backend = self._model_factory.create_vlm()
        results_by_frame: dict[str, tuple[VlmPassResult, ...]] = {}
        failures: list[str] = []
        try:
            for sample in samples:
                frame_results: list[VlmPassResult] = []
                for pass_name in _PASS_NAMES:
                    try:
                        result = self._artifacts.read_vlm_result(sample, pass_name)
                        if result is None:
                            prompt = build_prompt(pass_name, tuple(frame_results))
                            raw_response = backend.generate(sample.image_path, prompt)
                            self._artifacts.write_vlm_raw_response(
                                sample, pass_name, raw_response
                            )
                            result = parse_vlm_response(raw_response, pass_name)
                            self._artifacts.write_vlm_result(sample, result)
                        frame_results.append(result)
                    except Exception as error:  # noqa: BLE001 - per-pass isolation
                        failures.append(f"{sample.frame_id}:{pass_name}")
                        self._artifacts.write_failure(sample, f"vlm_{pass_name}", error)
                    self._progress.advance(f"{sample.frame_id}:{pass_name}")
                if frame_results:
                    results_by_frame[sample.frame_id] = tuple(frame_results)
        finally:
            backend.close()
        self._progress.failures("vlm", failures)
        return results_by_frame

    def _run_grounding_pass(
        self,
        samples: tuple[ImageSample, ...],
        concepts_by_frame: dict[str, tuple[VisualConcept, ...]],
    ) -> dict[str, tuple[Detection, ...]]:
        grounding_concepts_by_frame = {
            frame_id: _unique_detector_queries(concepts)
            for frame_id, concepts in concepts_by_frame.items()
        }
        total = sum(len(concepts) for concepts in grounding_concepts_by_frame.values())
        self._progress.stage("grounding-dino", total)
        if total == 0:
            empty_results: dict[str, tuple[Detection, ...]] = {
                sample.frame_id: ()
                for sample in samples
                if sample.frame_id in concepts_by_frame
            }
            for sample in samples:
                if sample.frame_id in empty_results:
                    self._artifacts.write_detections(sample, (), ())
            return empty_results
        backend = self._model_factory.create_grounder()
        detections_by_frame: dict[str, tuple[Detection, ...]] = {}
        failures: list[str] = []
        try:
            for sample in samples:
                concepts = grounding_concepts_by_frame.get(sample.frame_id)
                if concepts is None:
                    continue
                resumed = self._artifacts.read_detections(sample)
                if resumed is not None:
                    detections_by_frame[sample.frame_id] = resumed
                    for concept in concepts:
                        query = concept.detector_query or concept.label.lower()
                        self._progress.advance(f"{sample.frame_id}:{query}:resumed")
                    continue
                frame_detections: list[Detection] = []
                try:
                    for concept in concepts:
                        frame_detections.extend(backend.detect(sample, concept))
                        query = concept.detector_query or concept.label.lower()
                        self._progress.advance(f"{sample.frame_id}:{query}")
                except Exception as error:  # noqa: BLE001 - intentional per-frame isolation
                    failures.append(sample.frame_id)
                    self._artifacts.write_failure(sample, "grounding", error)
                    continue
                detections = tuple(frame_detections)
                self._artifacts.write_detections(sample, concepts, detections)
                detections_by_frame[sample.frame_id] = detections
        finally:
            backend.close()
        self._progress.failures("grounding", failures)
        return detections_by_frame

    def _run_segmentation_pass(
        self,
        samples: tuple[ImageSample, ...],
        pass_results: dict[str, tuple[VlmPassResult, ...]],
        concepts_by_frame: dict[str, tuple[VisualConcept, ...]],
        detections_by_frame: dict[str, tuple[Detection, ...]],
    ) -> list[str]:
        total = sum(len(detections) for detections in detections_by_frame.values())
        self._progress.stage("sam2", total)
        if total == 0:
            completed_without_masks: list[str] = []
            for sample in samples:
                detections = detections_by_frame.get(sample.frame_id)
                if detections is None:
                    continue
                self._artifacts.write_overlay(sample, detections, {})
                self._artifacts.write_result(
                    sample,
                    pass_results[sample.frame_id],
                    concepts_by_frame[sample.frame_id],
                    detections,
                )
                completed_without_masks.append(sample.frame_id)
            return completed_without_masks
        backend = self._model_factory.create_segmenter()
        completed: list[str] = []
        failures: list[str] = []
        try:
            for sample in samples:
                detections = detections_by_frame.get(sample.frame_id)
                if detections is None:
                    continue
                masks = {}
                try:
                    for detection in detections:
                        if self._artifacts.mask_is_complete(sample, detection):
                            self._progress.advance(f"{detection.detection_id}:resumed")
                            continue
                        mask, confidence = backend.segment(sample, detection)
                        if mask.shape != (sample.height, sample.width):
                            raise ValueError(
                                f"SAM2 returned mask shape {mask.shape}; expected "
                                f"{(sample.height, sample.width)}."
                            )
                        self._artifacts.write_mask(sample, detection, mask, confidence)
                        masks[detection.detection_id] = mask
                        self._progress.advance(detection.detection_id)
                    self._artifacts.write_overlay(sample, detections, masks)
                    self._artifacts.write_result(
                        sample,
                        pass_results[sample.frame_id],
                        concepts_by_frame[sample.frame_id],
                        detections,
                    )
                except (RuntimeError, ValueError, OSError, cv2.error) as error:
                    failures.append(sample.frame_id)
                    self._artifacts.write_failure(sample, "sam2", error)
                    continue
                completed.append(sample.frame_id)
        finally:
            backend.close()
        self._progress.failures("sam2", failures)
        return completed


class ConsoleProgressReporter:
    """Minimal terminal progress reporter without a UI dependency."""

    def __init__(self) -> None:
        self._stage = ""
        self._total = 0
        self._current = 0

    def stage(self, name: str, total: int) -> None:
        self._stage = name
        self._total = total
        self._current = 0
        print(f"[{name}] starting ({total} items)")

    def advance(self, detail: str) -> None:
        self._current += 1
        print(f"[{self._stage}] {self._current}/{self._total}: {detail}")

    def failures(self, stage: str, frame_ids: Sequence[str]) -> None:
        if frame_ids:
            print(f"[{stage}] failed frames: {', '.join(frame_ids)}")


def _unique_detector_queries(
    concepts: tuple[VisualConcept, ...],
) -> tuple[VisualConcept, ...]:
    """Keep the strongest concept for each query submitted to Grounding DINO."""
    by_query: dict[str, VisualConcept] = {}
    for concept in concepts:
        query = concept.detector_query or concept.label.lower()
        current = by_query.get(query)
        if current is None or concept.confidence > current.confidence:
            by_query[query] = concept
    return tuple(by_query.values())
