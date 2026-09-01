"""SAM2 backend using a Grounding DINO box as the mask prompt."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from image_context.config import Sam2Config
from image_context.models import Detection, ImageSample


class Sam2Backend:
    """Generate one segmentation mask per grounded detection."""

    def __init__(self, config: Sam2Config) -> None:
        self._config = config
        self._processor: Any = Sam2Processor.from_pretrained(config.checkpoint)
        self._model: Any = Sam2Model.from_pretrained(config.checkpoint).to(config.device)
        self._model.eval()

    def segment(
        self, sample: ImageSample, detection: Detection
    ) -> tuple[NDArray[np.bool_], float]:
        """Segment one detection using its pixel-space box."""
        box = detection.bounding_box
        input_boxes = [[[box.x_min, box.y_min, box.x_max, box.y_max]]]
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
            inputs = self._processor(
                images=image, input_boxes=input_boxes, return_tensors="pt"
            ).to(self._model.device)
        with torch.inference_mode():
            outputs = self._model(**inputs, multimask_output=False)
        masks = self._processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])
        mask: NDArray[np.bool_] = masks[0][0, 0].cpu().numpy().astype(np.bool_)
        return mask, float(outputs.iou_scores[0, 0, 0])

    def close(self) -> None:
        """Release the final model and CUDA cache."""
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
