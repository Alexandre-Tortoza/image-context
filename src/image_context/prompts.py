"""Versioned prompts for the three complementary scene-analysis passes."""

from __future__ import annotations

import json

from image_context.models import PassName, VlmPassResult
from image_context.serialization import vlm_result_to_dict

PROMPT_VERSION = "three-pass-context-v4"

_COMMON_CONTRACT = """
Return ONLY one compact JSON object without markdown, using this exact schema. The strings
below explain fields and are not example values to copy:
{"concepts":[{"label":"singular English noun","grounding_query":"lowercase visible noun phrase",\
"confidence":0.0,"region_kind":"bounded_object or environmental_region","attributes":{}}],\
"scene_attributes":{},"risks":[{"label":"short risk name",\
"severity":"low, medium, high, or critical","confidence":0.0,\
"evidence_queries":["visible noun phrase"],"rationale":"visible evidence only"}]}

Use confidence values from 0.0 to 1.0. Do not infer facts that are not visually supported.
Every grounding_query and evidence_query must name something that can be localized in the image.
scene_attributes is only for global conditions such as lighting, visibility, and navigability.
Never put an object inventory in scene_attributes; visible objects belong in concepts.
region_kind must be exactly "bounded_object" or "environmental_region". It describes
geometry, never image position; put words such as foreground, center, or left in attributes.
""".strip()


def build_prompt(pass_name: PassName, previous: tuple[VlmPassResult, ...]) -> str:
    """Build one pass prompt, embedding prior structured results when applicable."""
    if pass_name == "objects":
        objective = """
Inspect the complete image for distinct visible physical objects. Cover foreground,
middle ground, background, left, center, and right. Report access points, obstacles,
people, equipment, infrastructure, structural elements, and signs. Keep risks empty and
use scene_attributes only for directly visible global properties. Report 1 to 15 concepts
unless the image truly contains no physical object. When a built structure is visible,
report at least 3 distinct localizable instances. Architectural components, access points,
signs, robot parts, and other foreground items are concepts rather than scene attributes.
""".strip()
    elif pass_name == "environment":
        objective = """
Inspect the image for environmental context missed by the object pass: illumination,
visibility, navigability, smoke, dust, fire, water, debris fields, and other diffuse
conditions. Add concepts only for visible regions that a detector could localize.
Do not repeat object concepts unless correcting a clear omission. Keep risks empty.
""".strip()
    else:
        objective = """
Identify visually supported safety and navigation risks. Use the prior object and
environment evidence, but verify every claim against the image. Put risk claims in risks.
Also put every visible piece of evidence needed by Grounding DINO in concepts; never use
an abstract risk description as a grounding_query.
""".strip()

    prior_context = ""
    if previous:
        serialized = [vlm_result_to_dict(result, include_raw=False) for result in previous]
        prior_context = (
            "Prior validated pass results follow. Treat them as hypotheses, "
            "not guaranteed facts:\n"
            + json.dumps(serialized, separators=(",", ":"), sort_keys=True)
            + "\n\n"
        )
    return f"{prior_context}{objective}\n\n{_COMMON_CONTRACT}\n\nNow perform only this pass."
