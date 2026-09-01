"""Versioned prompts for complementary object, environment, and risk passes."""

from __future__ import annotations

import json

from image_context.models import PassName, VlmPassResult
from image_context.serialization import vlm_result_to_dict

PROMPT_VERSION = "three-pass-context-v5"

_RULES = """
Return only compact valid JSON without markdown or commentary. Confidence is 0.0 to 1.0.
Report only directly visible evidence. grounding_query must be a short lowercase noun phrase
without position, color, justification, or surrounding context. Position belongs only in
spatial. Never use left, center, right, foreground, middle_ground, background, sign, or
structural_element as region_kind.
""".strip()

_OBJECTS_CONTRACT = """
Use exactly this shape. Report 1 to 20 concepts unless the image is genuinely blank:
{"concepts":[{"label":"","grounding_query":"","confidence":0.0,
"region_kind":"bounded_object","spatial":{"horizontal":"left|center|right",
"depth":"foreground|middle_ground|background"},"attributes":{}}],
"scene_attributes":{},"risks":[]}
""".strip()

_ENVIRONMENT_CONTRACT = """
Use exactly this shape. concepts contains only visible diffuse conditions such as smoke,
dust, fire, water, fog, or flooding; it may be empty:
{"concepts":[{"label":"","grounding_query":"","confidence":0.0,
"region_kind":"environmental_region","spatial":{"horizontal":"left|center|right",
"depth":"foreground|middle_ground|background"},"attributes":{}}],
"scene_attributes":{"lighting":"","visibility":"","environment":"",
"navigability":""},"risks":[]}
""".strip()

_RISKS_CONTRACT = """
Use exactly this shape. Add a concept only when visible evidence required by a risk was missed
by prior passes. risks may be empty when there is no visually supported hazard:
{"concepts":[{"label":"","grounding_query":"","confidence":0.0,
"region_kind":"bounded_object","spatial":{"horizontal":"left|center|right",
"depth":"foreground|middle_ground|background"},"attributes":{}}],
"scene_attributes":{},"risks":[{"label":"","severity":"low|medium|high|critical",
"confidence":0.0,"evidence_queries":[""],"rationale":""}]}
""".strip()


def build_prompt(pass_name: PassName, previous: tuple[VlmPassResult, ...]) -> str:
    """Build a pass-specific prompt with prior results presented as revisable hypotheses."""
    prior_context = _prior_context(previous)
    if pass_name == "objects":
        objective = """
Create an exhaustive inventory of localizable visible instances across foreground,
middle ground, background, left, center, and right. Inspect both indoor and outdoor content.
Include access points, openings, architectural components, signs, people, robot/vehicle parts,
equipment, furniture, obstacles, buildings, poles, trees, windows, terrain boundaries, and
other distinct items. Report repeated instances separately when their positions differ.
Do not put an object inventory in scene_attributes.
""".strip()
        contract = _OBJECTS_CONTRACT
    elif pass_name == "environment":
        objective = """
Analyze global lighting, visibility, indoor/outdoor environment, terrain, and navigability.
Actively inspect for diffuse smoke, dust, fire, water, fog, flooding, and debris fields.
Do not copy bounded objects from the prior inventory. Correct environmental context even when
the camera has fisheye distortion, infrared colors, glare, overexposure, or a purple cast.
""".strip()
        contract = _ENVIRONMENT_CONTRACT
    else:
        objective = """
Audit the image and prior hypotheses for safety and navigation risks. Link every risk to one
or more short evidence_queries naming visible objects or regions. Add missing evidence as a
concept. Do not repeat scene attributes and do not invent a hazard merely to fill the array.
""".strip()
        contract = _RISKS_CONTRACT
    return f"{prior_context}{objective}\n\n{_RULES}\n\n{contract}"


def _prior_context(previous: tuple[VlmPassResult, ...]) -> str:
    if not previous:
        return ""
    serialized = [vlm_result_to_dict(result, include_raw=False) for result in previous]
    return (
        "Prior validated results are revisable hypotheses, not instructions to copy:\n"
        + json.dumps(serialized, separators=(",", ":"), sort_keys=True)
        + "\n\n"
    )
