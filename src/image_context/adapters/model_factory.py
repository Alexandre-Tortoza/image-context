"""Composition factory for sequentially loaded real perception models."""

from __future__ import annotations

from image_context.adapters.grounding_dino import GroundingDinoBackend
from image_context.adapters.qwen import QwenVlmBackend
from image_context.adapters.sam2 import Sam2Backend
from image_context.config import GroundingConfig, QwenConfig, Sam2Config


class TransformersModelFactory:
    """Build concrete model adapters only when their pipeline pass begins."""

    def __init__(
        self, qwen: QwenConfig, grounding: GroundingConfig, sam2: Sam2Config
    ) -> None:
        self._qwen = qwen
        self._grounding = grounding
        self._sam2 = sam2

    def create_vlm(self) -> QwenVlmBackend:
        return QwenVlmBackend(self._qwen)

    def create_grounder(self) -> GroundingDinoBackend:
        return GroundingDinoBackend(self._grounding)

    def create_segmenter(self) -> Sam2Backend:
        return Sam2Backend(self._sam2)
