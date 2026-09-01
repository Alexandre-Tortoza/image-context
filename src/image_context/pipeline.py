"""Stage-oriented orchestration with one heavy model resident at a time."""

from __future__ import annotations

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

    def __init__(
        self,
        sampler: ImageSampler,
        model_factory: ModelFactory,
        artifacts: ArtifactRepository,
        progress: ProgressReporter,
        *,
        minimum_concept_confidence: float,
    ) -> None:
        self._sampler = sampler
        self._model_factory = model_factory
        self._artifacts = artifacts
        self._progress = progress
        self._minimum_concept_confidence = minimum_concept_confidence

    def execute(self, sample_size: int, seed: int) -> PipelineOutcome:
        """Execute all stages and isolate model errors to their source frame."""
        self._progress.stage("sample-images", sample_size)
        samples = self._sampler.sample(sample_size, seed, self._artifacts.frames_directory)
        self._artifacts.write_selection(samples)
        for sample in samples:
            self._progress.advance(sample.frame_id)

        pass_results = self._run_vlm_pass(samples)
        concepts = {
            frame_id: consolidate_concepts(results, self._minimum_concept_confidence)
            for frame_id, results in pass_results.items()
        }
        for sample in samples:
            if sample.frame_id in concepts:
                self._artifacts.write_concepts(sample, concepts[sample.frame_id])

        detections = self._run_grounding_pass(samples, concepts)
        completed = self._run_segmentation_pass(samples, pass_results, concepts, detections)
        failed = tuple(sample.frame_id for sample in samples if sample.frame_id not in completed)
        return PipelineOutcome(samples, tuple(completed), failed)

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
                try:
                    for pass_name in _PASS_NAMES:
                        result = self._artifacts.read_vlm_result(sample, pass_name)
                        if result is None:
                            prompt = build_prompt(pass_name, tuple(frame_results))
                            result = parse_vlm_response(
                                backend.generate(sample.image_path, prompt), pass_name
                            )
                            self._artifacts.write_vlm_result(sample, result)
                        frame_results.append(result)
                        self._progress.advance(f"{sample.frame_id}:{pass_name}")
                except Exception as error:  # noqa: BLE001 - intentional per-frame isolation
                    failures.append(sample.frame_id)
                    self._artifacts.write_failure(sample, "vlm", error)
                    continue
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
        total = sum(len(concepts) for concepts in concepts_by_frame.values())
        self._progress.stage("grounding-dino", total)
        if total == 0:
            empty_results: dict[str, tuple[Detection, ...]] = {
                sample.frame_id: ()
                for sample in samples
                if sample.frame_id in concepts_by_frame
            }
            for sample in samples:
                if sample.frame_id in empty_results:
                    self._artifacts.write_detections(sample, ())
            return empty_results
        backend = self._model_factory.create_grounder()
        detections_by_frame: dict[str, tuple[Detection, ...]] = {}
        failures: list[str] = []
        try:
            for sample in samples:
                concepts = concepts_by_frame.get(sample.frame_id)
                if concepts is None:
                    continue
                resumed = self._artifacts.read_detections(sample)
                if resumed is not None:
                    detections_by_frame[sample.frame_id] = resumed
                    for concept in concepts:
                        self._progress.advance(f"{sample.frame_id}:{concept.grounding_query}:resumed")
                    continue
                frame_detections: list[Detection] = []
                try:
                    for concept in concepts:
                        frame_detections.extend(backend.detect(sample, concept))
                        self._progress.advance(f"{sample.frame_id}:{concept.grounding_query}")
                except Exception as error:  # noqa: BLE001 - intentional per-frame isolation
                    failures.append(sample.frame_id)
                    self._artifacts.write_failure(sample, "grounding", error)
                    continue
                detections = tuple(frame_detections)
                self._artifacts.write_detections(sample, detections)
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
                except (ValueError, OSError, cv2.error) as error:
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
