"""Grounding DINO backend using Hugging Face Transformers."""

from __future__ import annotations

import gc
import hashlib
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from image_context.config import GroundingConfig
from image_context.grounding import non_max_suppression
from image_context.models import BoundingBox, Detection, ImageSample, VisualConcept


class GroundingDinoBackend:
    """Ground each concept independently to preserve concept-region provenance."""

    def __init__(self, config: GroundingConfig) -> None:
        self._config = config
        self._processor: Any = AutoProcessor.from_pretrained(config.checkpoint)
        self._model: Any = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.checkpoint
        ).to(config.device)
        self._model.eval()

    def detect(self, sample: ImageSample, concept: VisualConcept) -> tuple[Detection, ...]:
        """Return the strongest pixel-space regions for one concept query."""
        detector_query = concept.detector_query or concept.label.lower()
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
            inputs = self._processor(
                images=image,
                text=f"{detector_query.strip().lower()}.",
                return_tensors="pt",
            ).to(self._model.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=self._config.box_threshold,
            text_threshold=self._config.text_threshold,
            target_sizes=[(sample.height, sample.width)],
        )[0]
        candidates = tuple(
            (
                BoundingBox(*(float(coordinate) for coordinate in box)),
                float(score),
            )
            for box, score in zip(result["boxes"], result["scores"], strict=True)
            if float(score) >= self._minimum_score(concept)
        )
        kept = non_max_suppression(candidates, self._config.nms_iou_threshold)[
            : self._config.maximum_detections_per_concept
        ]
        detections: list[Detection] = []
        for box, score in kept:
            identity = (
                f"{sample.frame_id}|{detector_query}|{box.x_min:.2f}|{box.y_min:.2f}|"
                f"{box.x_max:.2f}|{box.y_max:.2f}"
            )
            detections.append(
                Detection(
                    detection_id=hashlib.sha256(identity.encode()).hexdigest()[:12],
                    concept=concept,
                    confidence=score,
                    bounding_box=box,
                )
            )
        return tuple(detections)

    def _minimum_score(self, concept: VisualConcept) -> float:
        is_fallback = any("fallback" in note for note in concept.parser_notes)
        return (
            self._config.fallback_box_threshold
            if is_fallback
            else self._config.box_threshold
        )

    def close(self) -> None:
        """Release model references and CUDA cache before the SAM2 pass."""
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
