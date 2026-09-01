"""Qwen3-VL backend using Hugging Face Transformers."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from image_context.config import QwenConfig


class QwenVlmBackend:
    """Keep one Qwen checkpoint resident for the complete three-prompt pass."""

    def __init__(self, config: QwenConfig) -> None:
        self._config = config
        model_options: dict[str, Any] = {}
        if config.quantization == "int4":
            from transformers import BitsAndBytesConfig

            model_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_options["device_map"] = config.device
        else:
            model_options["dtype"] = torch.bfloat16
        self._processor: Any = AutoProcessor.from_pretrained(config.checkpoint)
        self._model: Any = AutoModelForImageTextToText.from_pretrained(
            config.checkpoint, **model_options
        )
        if config.quantization == "none":
            self._model = self._model.to(config.device)
        self._model.eval()

    def generate(self, image_path: Path, prompt: str) -> str:
        """Generate deterministic structured text for one image and prompt."""
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        response: str = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return response.strip()

    def close(self) -> None:
        """Release model references and CUDA cache before the DINO pass."""
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
