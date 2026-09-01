"""SAM2 automatic region discovery and DINOv2 dense region features."""

from __future__ import annotations

import gc
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, Sam2Model, Sam2Processor

from image_context.adapters.qwen import QwenVlmBackend
from image_context.config import QwenConfig, RegionFirstConfig, Sam2Config
from image_context.models import ImageSample
from image_context.region_models import VisualRegion
from image_context.region_pipeline import bounding_box_from_mask


class TransformersRegionFirstModelFactory:
    """Lazily construct region-first models so only one occupies VRAM at a time."""

    def __init__(
        self, sam2: Sam2Config, region_first: RegionFirstConfig, qwen: QwenConfig
    ) -> None:
        self._sam2 = sam2
        self._region_first = region_first
        self._qwen = qwen

    def create_discoverer(self) -> Sam2AutomaticRegionDiscoverer:
        return Sam2AutomaticRegionDiscoverer(self._sam2, self._region_first)

    def create_feature_extractor(self) -> DinoV2DenseFeatureExtractor:
        return DinoV2DenseFeatureExtractor(self._region_first)

    def create_vlm(self) -> QwenVlmBackend:
        return QwenVlmBackend(self._qwen)


class Sam2AutomaticRegionDiscoverer:
    """Discover class-agnostic masks from a deterministic grid of positive point prompts."""

    def __init__(self, sam2: Sam2Config, config: RegionFirstConfig) -> None:
        self._config = config
        self._processor: Any = Sam2Processor.from_pretrained(sam2.checkpoint)
        self._model: Any = Sam2Model.from_pretrained(sam2.checkpoint).to(sam2.device)
        self._model.eval()

    def discover(self, sample: ImageSample) -> tuple[VisualRegion, ...]:
        """Generate, filter, and mask-IoU-deduplicate automatic proposals."""
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
        candidates: list[tuple[NDArray[np.bool_], float]] = []
        size = self._config.discovery_grid_size
        points = [
            [
                (column + 0.5) * sample.width / size,
                (row + 0.5) * sample.height / size,
            ]
            for row in range(size)
            for column in range(size)
        ]
        inputs = self._processor(
            images=image,
            input_points=[[[point] for point in points]],
            input_labels=[[[1] for _ in points]],
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            outputs = self._model(**inputs, multimask_output=True)
        processed = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], binarize=False
        )[0]
        mask_logits = processed.reshape(-1, sample.height, sample.width)
        scores = outputs.iou_scores.reshape(-1)
        object_logits = outputs.object_score_logits.reshape(-1)
        masks_per_object = max(1, mask_logits.shape[0] // object_logits.numel())
        presence = object_logits.repeat_interleave(masks_per_object) > 0
        for logits_tensor, score_tensor, is_present in zip(
            mask_logits, scores, presence, strict=False
        ):
            logits = logits_tensor.detach().cpu().numpy()
            score = float(score_tensor.detach().cpu())
            present = bool(is_present.detach().cpu())
            mask = logits > 0
            area_fraction = float(mask.mean())
            stability_union = int((logits > -1.0).sum())
            stability = (
                0.0 if stability_union == 0 else float((logits > 1.0).sum() / stability_union)
            )
            if (
                present
                and score >= self._config.minimum_region_confidence
                and stability >= self._config.minimum_region_stability
                and self._config.minimum_region_area_fraction
                <= area_fraction
                <= self._config.maximum_region_area_fraction
            ):
                candidates.append((mask, score))
        selected: list[tuple[NDArray[np.bool_], float]] = []
        for mask, score in sorted(candidates, key=lambda item: item[1], reverse=True):
            if all(_mask_iou(mask, kept) < self._config.mask_iou_threshold for kept, _ in selected):
                selected.append((mask, score))
            if len(selected) == self._config.maximum_regions:
                break
        return tuple(
            VisualRegion(
                region_id=f"region-{index:03d}",
                mask=mask,
                bounding_box=bounding_box_from_mask(mask),
                confidence=score,
                source="sam2-automatic-grid-prompt",
            )
            for index, (mask, score) in enumerate(selected)
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        _release_cuda()


class DinoV2DenseFeatureExtractor:
    """Pool DINOv2 patch features through each SAM2 region mask."""

    def __init__(self, config: RegionFirstConfig) -> None:
        self._device = config.feature_device
        self._processor: Any = AutoImageProcessor.from_pretrained(config.feature_checkpoint)
        self._model: Any = AutoModel.from_pretrained(config.feature_checkpoint).to(self._device)
        self._model.eval()

    def extract(
        self, sample: ImageSample, regions: tuple[VisualRegion, ...]
    ) -> dict[str, NDArray[np.float32]]:
        """Average normalized patch tokens selected by each resized binary mask."""
        if not regions:
            return {}
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
            inputs = self._processor(
                images=image, do_center_crop=False, return_tensors="pt"
            ).to(self._device)
        with torch.inference_mode():
            hidden = self._model(**inputs).last_hidden_state[0]
        pixel_height, pixel_width = inputs["pixel_values"].shape[-2:]
        patch_size = int(self._model.config.patch_size)
        grid_height, grid_width = pixel_height // patch_size, pixel_width // patch_size
        patch_count = grid_height * grid_width
        patch_features = hidden[-patch_count:].reshape(grid_height, grid_width, -1)
        result: dict[str, NDArray[np.float32]] = {}
        for region in regions:
            resized = cv2.resize(
                region.mask.astype(np.uint8),
                (grid_width, grid_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if not resized.any():
                box = region.bounding_box
                center_x = min(
                    grid_width - 1,
                    round((box.x_min + box.x_max) / 2 / sample.width * grid_width),
                )
                center_y = min(
                    grid_height - 1,
                    round((box.y_min + box.y_max) / 2 / sample.height * grid_height),
                )
                resized[center_y, center_x] = True
            embedding = patch_features[torch.from_numpy(resized).to(self._device)].mean(dim=0)
            embedding = torch.nn.functional.normalize(embedding, dim=0)
            result[region.region_id] = embedding.detach().cpu().numpy().astype(np.float32)
        return result

    def close(self) -> None:
        self._model = None
        self._processor = None
        _release_cuda()


def _mask_iou(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return 0.0 if union == 0 else float(intersection / union)


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
