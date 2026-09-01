"""Grounding DINO backend using Hugging Face Transformers."""

from __future__ import annotations

import gc
import hashlib
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from image_context.config import GroundingConfig
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
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
            inputs = self._processor(
                images=image,
                text=f"{concept.grounding_query.strip().lower()}.",
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
        candidates = sorted(
            zip(result["boxes"], result["scores"], strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[: self._config.maximum_detections_per_concept]
        query_hash = hashlib.sha256(concept.grounding_query.encode()).hexdigest()[:10]
        return tuple(
            Detection(
                detection_id=f"{query_hash}-{index}",
                concept=concept,
                confidence=float(score),
                bounding_box=BoundingBox(*(float(coordinate) for coordinate in box)),
            )
            for index, (box, score) in enumerate(candidates)
        )

    def close(self) -> None:
        """Release model references and CUDA cache before the SAM2 pass."""
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
