"""Durable JSON, mask, and visualization artifacts for one run."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from image_context.models import Detection, ImageSample, PassName, VisualConcept, VlmPassResult
from image_context.serialization import (
    concept_to_dict,
    detection_from_dict,
    detection_to_dict,
    sample_to_dict,
    vlm_result_from_dict,
    vlm_result_to_dict,
)


class ArtifactRepository:
    """Owns the stable on-disk contract and resumable stage records."""

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory

    def initialize(self, manifest: dict[str, Any], *, overwrite: bool) -> None:
        """Initialize a run or validate that an existing run is compatible."""
        manifest_path = self.run_directory / "manifest.json"
        if self.run_directory.exists() and overwrite:
            shutil.rmtree(self.run_directory)
        if manifest_path.exists():
            existing = self._read_json(manifest_path)
            if existing.get("fingerprint") != manifest.get("fingerprint"):
                raise ValueError(
                    f"Run directory '{self.run_directory}' contains a different configuration. "
                    "Use --overwrite or another --run-id."
                )
            return
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._write_json(manifest_path, manifest)

    @property
    def frames_directory(self) -> Path:
        """Directory where extracted frame folders are stored."""
        return self.run_directory / "frames"

    def frame_directory(self, sample: ImageSample) -> Path:
        """Return one sample's artifact directory."""
        return self.frames_directory / sample.frame_id

    def write_selection(self, samples: tuple[ImageSample, ...]) -> None:
        """Persist selected source indices and extracted image metadata."""
        self._write_json(
            self.run_directory / "selected_frames.json",
            {"frames": [sample_to_dict(sample) for sample in samples]},
        )

    def write_vlm_result(self, sample: ImageSample, result: VlmPassResult) -> None:
        """Persist a validated VLM pass response."""
        self._write_json(
            self.frame_directory(sample) / f"vlm_{result.pass_name}.json",
            vlm_result_to_dict(result),
        )

    def read_vlm_result(self, sample: ImageSample, pass_name: PassName) -> VlmPassResult | None:
        """Load a completed VLM pass when resuming."""
        path = self.frame_directory(sample) / f"vlm_{pass_name}.json"
        return None if not path.exists() else vlm_result_from_dict(self._read_json(path))

    def write_concepts(self, sample: ImageSample, concepts: tuple[VisualConcept, ...]) -> None:
        """Persist concepts after cross-pass consolidation."""
        self._write_json(
            self.frame_directory(sample) / "consolidated_context.json",
            {"concepts": [concept_to_dict(concept) for concept in concepts]},
        )

    def write_detections(self, sample: ImageSample, detections: tuple[Detection, ...]) -> None:
        """Persist Grounding DINO results."""
        self._write_json(
            self.frame_directory(sample) / "detections.json",
            {"detections": [detection_to_dict(item) for item in detections]},
        )

    def read_detections(self, sample: ImageSample) -> tuple[Detection, ...] | None:
        """Load completed Grounding DINO results when resuming."""
        path = self.frame_directory(sample) / "detections.json"
        if not path.exists():
            return None
        return tuple(detection_from_dict(item) for item in self._read_json(path)["detections"])

    def write_mask(
        self,
        sample: ImageSample,
        detection: Detection,
        mask: NDArray[np.bool_],
        confidence: float,
    ) -> Path:
        """Persist one lossless binary mask and its metadata."""
        mask_directory = self.frame_directory(sample) / "masks"
        mask_directory.mkdir(parents=True, exist_ok=True)
        mask_path = mask_directory / f"{detection.detection_id}.png"
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise OSError(f"Could not write segmentation mask to '{mask_path}'.")
        self._write_json(
            mask_directory / f"{detection.detection_id}.json",
            {
                "detection_id": detection.detection_id,
                "confidence": confidence,
                "mask_path": str(mask_path),
            },
        )
        return mask_path

    def mask_is_complete(self, sample: ImageSample, detection: Detection) -> bool:
        """Return whether both mask and metadata exist for a detection."""
        directory = self.frame_directory(sample) / "masks"
        return (directory / f"{detection.detection_id}.png").exists() and (
            directory / f"{detection.detection_id}.json"
        ).exists()

    def write_overlay(
        self,
        sample: ImageSample,
        detections: tuple[Detection, ...],
        masks: dict[str, NDArray[np.bool_]],
    ) -> Path:
        """Render boxes, labels, and translucent masks for human inspection."""
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read extracted image '{sample.image_path}'.")
        palette = ((40, 220, 255), (255, 120, 40), (90, 230, 90), (220, 80, 220))
        for index, detection in enumerate(detections):
            color = palette[index % len(palette)]
            mask = masks.get(detection.detection_id)
            if mask is None:
                mask_path = self.frame_directory(sample) / "masks" / (
                    f"{detection.detection_id}.png"
                )
                loaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = None if loaded is None else loaded > 0
            if mask is not None:
                image[mask] = (image[mask] * 0.55 + np.asarray(color) * 0.45).astype(np.uint8)
            box = detection.bounding_box
            start = (round(box.x_min), round(box.y_min))
            end = (round(box.x_max), round(box.y_max))
            cv2.rectangle(image, start, end, color, 2)
            cv2.putText(
                image,
                f"{detection.concept.label} {detection.confidence:.2f}",
                (start[0], max(18, start[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        path = self.frame_directory(sample) / "overlay.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write overlay to '{path}'.")
        return path

    def write_result(
        self,
        sample: ImageSample,
        pass_results: tuple[VlmPassResult, ...],
        concepts: tuple[VisualConcept, ...],
        detections: tuple[Detection, ...],
    ) -> None:
        """Persist the complete human- and machine-readable result for one image."""
        self._write_json(
            self.frame_directory(sample) / "result.json",
            {
                "sample": sample_to_dict(sample),
                "scene_attributes_by_pass": {
                    result.pass_name: result.scene_attributes for result in pass_results
                },
                "risks": [
                    risk
                    for result in pass_results
                    for risk in vlm_result_to_dict(result, include_raw=False)["risks"]
                ],
                "concepts": [concept_to_dict(concept) for concept in concepts],
                "detections": [detection_to_dict(detection) for detection in detections],
            },
        )

    def write_failure(self, sample: ImageSample, stage: str, error: Exception) -> None:
        """Persist a frame-local failure without discarding other samples."""
        self._write_json(
            self.frame_directory(sample) / f"failure_{stage}.json",
            {"stage": stage, "error_type": type(error).__name__, "error": str(error)},
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as json_file:
            result: dict[str, Any] = json.load(json_file)
        return result

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2, sort_keys=True)
            json_file.flush()
            os.fsync(json_file.fileno())
        os.replace(temporary, path)
