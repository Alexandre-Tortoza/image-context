from __future__ import annotations

from image_context.models import VlmPassResult
from image_context.prompts import build_prompt


def test_later_prompt_embeds_prior_validated_result() -> None:
    objects = VlmPassResult(
        pass_name="objects",
        concepts=(),
        scene_attributes={"location": "corridor"},
        risks=(),
        raw_response="not embedded",
    )

    prompt = build_prompt("environment", (objects,))

    assert "corridor" in prompt
    assert "not embedded" not in prompt
    assert "hypotheses" in prompt
    assert prompt.index("corridor") < prompt.index("Inspect the image")
