"""Structured prompts for global and region-local VLM interpretation."""

from __future__ import annotations

import json

from image_context.region_models import GlobalSceneContext, VisualRegion

REGION_PROMPT_VERSION = "region-first-v1"

GLOBAL_CONTEXT_PROMPT = """Analyze the complete image as scene context. Do not enumerate objects.
Return only one JSON object with exactly this shape:
{
  "scene_type": "short scene category",
  "scene_description": "concise global description",
  "global_attributes": ["lighting, layout, weather, or activity"],
  "potential_hazards": ["hazards supported by visible evidence"]
}
Use empty arrays when no attribute or hazard is visible."""


def build_region_prompt(region: VisualRegion, context: GlobalSceneContext) -> str:
    """Build a local prompt grounded by immutable observed geometry and global context."""
    global_payload = {
        "scene_type": context.scene_type,
        "scene_description": context.scene_description,
        "global_attributes": list(context.global_attributes),
        "potential_hazards": list(context.potential_hazards),
    }
    return f"""Analyze only the highlighted/cropped visual region {region.region_id}.
The region was discovered class-agnostically before this semantic analysis. Do not change its ID
or geometry. Use the global scene context only to disambiguate visible local evidence.
Global context: {json.dumps(global_payload, sort_keys=True)}
Return only one JSON object with exactly this shape:
{{
  "region_id": "{region.region_id}",
  "label": "concise visible class or region name",
  "description": "what is visibly present in this region",
  "attributes": ["visible local properties"],
  "condition": "visible state, or unknown",
  "confidence": 0.0,
  "candidate_relations": ["possible relation to another region or scene"]
}}
Do not claim hidden identity, intent, or relations as facts."""
