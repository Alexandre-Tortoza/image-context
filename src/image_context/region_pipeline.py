"""Region-first orchestration with geometry/semantics separation and durable artifacts."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_context.models import BoundingBox, ImageSample
from image_context.region_models import (
    GlobalSceneContext,
    RegionFirstOutcome,
    SemanticRegion,
    VisualRegion,
)
from image_context.region_parsing import parse_global_context, parse_region_interpretation
from image_context.region_ports import (
    RegionFirstModelFactory,
)
from image_context.region_prompts import GLOBAL_CONTEXT_PROMPT, build_region_prompt


class RegionFirstContextPipeline:
    """Discover regions first, then attach dense and semantic context to their stable IDs."""

    strategy_name = "region-first"

    def __init__(
        self,
        model_factory: RegionFirstModelFactory,
        output_directory: Path,
        stage_fingerprints: dict[str, str] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._output_directory = output_directory
        self._stage_fingerprints = stage_fingerprints or {}

    def analyze(self, sample: ImageSample) -> RegionFirstOutcome:
        """Analyze one image while isolating malformed local VLM responses by region."""
        started = time.perf_counter()
        self._output_directory.mkdir(parents=True, exist_ok=True)
        stage_seconds: dict[str, float] = {}

        stage_started = time.perf_counter()
        regions = self._read_regions()
        if regions is None:
            region_directory = self._output_directory / "regions"
            if region_directory.exists():
                shutil.rmtree(region_directory)
            discoverer = self._model_factory.create_discoverer()
            try:
                regions = discoverer.discover(sample)
            finally:
                discoverer.close()
        self._validate_regions(sample, regions)
        stage_seconds["region_discovery"] = time.perf_counter() - stage_started

        masks = self._persist_regions(regions)
        stage_started = time.perf_counter()
        embeddings = self._read_embeddings(regions)
        if embeddings is None:
            feature_extractor = self._model_factory.create_feature_extractor()
            try:
                embeddings = feature_extractor.extract(sample, regions)
            finally:
                feature_extractor.close()
        self._validate_embeddings(regions, embeddings)
        embedding_paths = self._persist_embeddings(embeddings)
        stage_seconds["dense_features"] = time.perf_counter() - stage_started

        vlm_calls = 0
        semantic_regions: list[SemanticRegion] = []
        failures: list[dict[str, str]] = []
        stage_started = time.perf_counter()
        raw_global = self._read_global_response()
        cached_local = {
            region.region_id: self._read_local_response(region.region_id) for region in regions
        }
        needs_vlm = raw_global is None or any(value is None for value in cached_local.values())
        vlm = self._model_factory.create_vlm() if needs_vlm else None
        try:
            if raw_global is None:
                if vlm is None:
                    raise RuntimeError("VLM backend is unavailable for global context generation.")
                raw_global = vlm.generate(sample.image_path, GLOBAL_CONTEXT_PROMPT)
                vlm_calls += 1
                (self._output_directory / "vlm_global_raw.txt").write_text(
                    raw_global, encoding="utf-8"
                )
            global_context = parse_global_context(raw_global)
            self._write_json(
                self._output_directory / "global_context.json",
                self._global_payload(global_context),
            )
            self._write_stage_marker("vlm-global-local", "global_complete.json")
            for region in regions:
                try:
                    composite_path = self._write_region_composite(sample, region)
                    region_directory = self._output_directory / "regions" / region.region_id
                    raw_local = cached_local[region.region_id]
                    if raw_local is None:
                        if vlm is None:
                            raise RuntimeError(
                                "VLM backend is unavailable for local context generation."
                            )
                        raw_local = vlm.generate(
                            composite_path, build_region_prompt(region, global_context)
                        )
                        vlm_calls += 1
                        (region_directory / "vlm_raw.txt").write_text(
                            raw_local, encoding="utf-8"
                        )
                    interpretation = parse_region_interpretation(raw_local, region.region_id)
                    self._write_stage_marker(
                        "vlm-global-local",
                        str(Path("regions") / region.region_id / "vlm_complete.json"),
                    )
                    semantic_regions.append(
                        SemanticRegion(
                            region_id=region.region_id,
                            mask_path=masks[region.region_id],
                            bounding_box=region.bounding_box,
                            geometry_confidence=region.confidence,
                            label=interpretation.label,
                            description=interpretation.description,
                            attributes=interpretation.attributes,
                            condition=interpretation.condition,
                            confidence=interpretation.confidence,
                            visual_embedding_path=embedding_paths[region.region_id],
                            embedding_dimension=int(embeddings[region.region_id].size),
                            candidate_relations=interpretation.candidate_relations,
                            provenance={
                                "geometry": region.source,
                                "semantics": "qwen-local",
                                "global_context": "qwen-global",
                                "visual_embedding": "dinov2-mask-pooling",
                            },
                        )
                    )
                except Exception as error:  # noqa: BLE001 - explicit per-region isolation
                    failures.append(
                        {
                            "region_id": region.region_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
        finally:
            if vlm is not None:
                vlm.close()
        stage_seconds["vlm_global_and_local"] = time.perf_counter() - stage_started

        self._write_overlay(sample, regions, semantic_regions)
        payload = {
            "schema_version": 1,
            "strategy": self.strategy_name,
            "sample": {
                "frame_id": sample.frame_id,
                "image_path": str(sample.image_path),
                "width": sample.width,
                "height": sample.height,
            },
            "global_context": self._global_payload(global_context),
            "semantic_regions": [self._semantic_payload(item) for item in semantic_regions],
            "failed_regions": failures,
        }
        self._write_json(self._output_directory / "semantic_regions.json", payload)
        elapsed = time.perf_counter() - started
        self._write_json(
            self._output_directory / "metrics.json",
            {
                "strategy": self.strategy_name,
                "elapsed_seconds": elapsed,
                "stage_seconds": stage_seconds,
                "discovered_regions": len(regions),
                "semantic_regions": len(semantic_regions),
                "labeled_regions": sum(bool(region.label) for region in semantic_regions),
                "average_confidence": (
                    sum(region.confidence for region in semantic_regions) / len(semantic_regions)
                    if semantic_regions
                    else 0.0
                ),
                "discarded_regions": len(regions) - len(semantic_regions),
                "vlm_calls": vlm_calls,
            },
        )
        self._write_json(
            self._output_directory / "complete.json",
            {
                "status": (
                    "partial" if failures else "complete"
                ),
                "successful": not regions or bool(semantic_regions),
            },
        )
        return RegionFirstOutcome(
            tuple(semantic_regions),
            len(regions),
            len(regions) - len(semantic_regions),
            vlm_calls,
            self._output_directory,
        )

    @staticmethod
    def _validate_regions(sample: ImageSample, regions: tuple[VisualRegion, ...]) -> None:
        identifiers: set[str] = set()
        for region in regions:
            if region.region_id in identifiers:
                raise ValueError(f"Duplicate discovered region ID: '{region.region_id}'.")
            if region.mask.shape != (sample.height, sample.width):
                raise ValueError(
                    f"Region '{region.region_id}' has mask shape {region.mask.shape}; expected "
                    f"{(sample.height, sample.width)}."
                )
            if not region.mask.any():
                raise ValueError(f"Region '{region.region_id}' has an empty mask.")
            identifiers.add(region.region_id)

    @staticmethod
    def _validate_embeddings(
        regions: tuple[VisualRegion, ...], embeddings: dict[str, np.ndarray[Any, Any]]
    ) -> None:
        expected = {region.region_id for region in regions}
        if set(embeddings) != expected:
            raise ValueError("Dense feature output must contain exactly one embedding per region.")
        for region_id, embedding in embeddings.items():
            if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
                raise ValueError(f"Region '{region_id}' has an invalid visual embedding.")

    def _persist_regions(self, regions: tuple[VisualRegion, ...]) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for region in regions:
            directory = self._output_directory / "regions" / region.region_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "mask.png"
            if not cv2.imwrite(str(path), region.mask.astype(np.uint8) * 255):
                raise OSError(f"Could not write region mask to '{path}'.")
            paths[region.region_id] = path
        self._write_json(
            self._output_directory / "discovered_regions.json",
            {
                "stage_fingerprint": self._stage_fingerprints.get("region-discovery", ""),
                "regions": [
                    {
                        "region_id": region.region_id,
                        "mask_path": self._relative(paths[region.region_id]),
                        "bounding_box": [
                            region.bounding_box.x_min,
                            region.bounding_box.y_min,
                            region.bounding_box.x_max,
                            region.bounding_box.y_max,
                        ],
                        "confidence": region.confidence,
                        "source": region.source,
                    }
                    for region in regions
                ],
            },
        )
        return paths

    def _read_regions(self) -> tuple[VisualRegion, ...] | None:
        path = self._output_directory / "discovered_regions.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage_fingerprint") != self._stage_fingerprints.get(
            "region-discovery", ""
        ):
            return None
        regions: list[VisualRegion] = []
        for item in payload.get("regions", []):
            mask_path = self._output_directory / item["mask_path"]
            loaded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if loaded is None:
                return None
            box = item["bounding_box"]
            regions.append(
                VisualRegion(
                    item["region_id"],
                    loaded > 0,
                    BoundingBox(*map(float, box)),
                    float(item["confidence"]),
                    item["source"],
                )
            )
        return tuple(regions)

    def _persist_embeddings(
        self, embeddings: dict[str, np.ndarray[Any, Any]]
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for region_id, embedding in embeddings.items():
            path = self._output_directory / "regions" / region_id / "visual_embedding.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, embedding, allow_pickle=False)
            paths[region_id] = path
        self._write_stage_marker("dense-features", "dense_features_complete.json")
        return paths

    def _read_embeddings(
        self, regions: tuple[VisualRegion, ...]
    ) -> dict[str, np.ndarray[Any, Any]] | None:
        marker = self._read_stage_marker("dense_features_complete.json")
        if marker != self._stage_fingerprints.get("dense-features", ""):
            return None
        result: dict[str, np.ndarray[Any, Any]] = {}
        for region in regions:
            path = self._output_directory / "regions" / region.region_id / "visual_embedding.npy"
            if not path.is_file():
                return None
            result[region.region_id] = np.load(path, allow_pickle=False)
        return result

    def _read_global_response(self) -> str | None:
        if self._read_stage_marker("global_complete.json") != self._stage_fingerprints.get(
            "vlm-global-local", ""
        ):
            return None
        path = self._output_directory / "vlm_global_raw.txt"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def _read_local_response(self, region_id: str) -> str | None:
        directory = self._output_directory / "regions" / region_id
        marker = directory / "vlm_complete.json"
        raw_path = directory / "vlm_raw.txt"
        if not marker.is_file() or not raw_path.is_file():
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("stage_fingerprint") != self._stage_fingerprints.get(
            "vlm-global-local", ""
        ):
            return None
        return raw_path.read_text(encoding="utf-8")

    def _write_stage_marker(self, stage: str, relative_path: str) -> None:
        self._write_json(
            self._output_directory / relative_path,
            {"stage_fingerprint": self._stage_fingerprints.get(stage, "")},
        )

    def _read_stage_marker(self, relative_path: str) -> str | None:
        path = self._output_directory / relative_path
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("stage_fingerprint")
        return value if isinstance(value, str) else None

    def _write_region_composite(self, sample: ImageSample, region: VisualRegion) -> Path:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read image '{sample.image_path}'.")
        highlighted = image.copy()
        color = np.asarray((40, 220, 255), dtype=np.uint8)
        highlighted[region.mask] = (
            highlighted[region.mask].astype(np.float32) * 0.45 + color * 0.55
        ).astype(np.uint8)
        box = region.bounding_box
        x1, y1 = max(0, int(box.x_min)), max(0, int(box.y_min))
        x2, y2 = min(sample.width, int(np.ceil(box.x_max))), min(
            sample.height, int(np.ceil(box.y_max))
        )
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"Region '{region.region_id}' has an invalid crop.")
        interpolation = cv2.INTER_AREA if crop.shape[0] > sample.height else cv2.INTER_CUBIC
        fitted = cv2.resize(crop, (sample.width, sample.height), interpolation=interpolation)
        composite = np.concatenate((highlighted, fitted), axis=1)
        path = self._output_directory / "regions" / region.region_id / "context.png"
        if not cv2.imwrite(str(path), composite):
            raise OSError(f"Could not write region composite to '{path}'.")
        return path

    def _write_overlay(
        self,
        sample: ImageSample,
        regions: tuple[VisualRegion, ...],
        semantic_regions: list[SemanticRegion],
    ) -> None:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read image '{sample.image_path}'.")
        labels = {region.region_id: region.label for region in semantic_regions}
        palette = ((40, 220, 255), (255, 120, 40), (90, 230, 90), (220, 80, 220))
        for index, region in enumerate(regions):
            color = palette[index % len(palette)]
            image[region.mask] = (
                image[region.mask].astype(np.float32) * 0.65 + np.asarray(color) * 0.35
            ).astype(np.uint8)
            box = region.bounding_box
            origin = (round(box.x_min), max(18, round(box.y_min)))
            cv2.putText(
                image,
                f"{region.region_id}: {labels.get(region.region_id, 'unresolved')}",
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        path = self._output_directory / "region_overlay.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write region overlay to '{path}'.")

    @staticmethod
    def _global_payload(context: GlobalSceneContext) -> dict[str, object]:
        return {
            "scene_type": context.scene_type,
            "scene_description": context.scene_description,
            "global_attributes": list(context.global_attributes),
            "potential_hazards": list(context.potential_hazards),
        }

    def _semantic_payload(self, region: SemanticRegion) -> dict[str, object]:
        box = region.bounding_box
        return {
            "region_id": region.region_id,
            "mask_path": self._relative(region.mask_path),
            "bounding_box": [box.x_min, box.y_min, box.x_max, box.y_max],
            "geometry_confidence": region.geometry_confidence,
            "label": region.label,
            "description": region.description,
            "attributes": list(region.attributes),
            "condition": region.condition,
            "confidence": region.confidence,
            "visual_embedding_path": self._relative(region.visual_embedding_path),
            "embedding_dimension": region.embedding_dimension,
            "candidate_relations": list(region.candidate_relations),
            "provenance": region.provenance,
        }

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self._output_directory))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)


def bounding_box_from_mask(mask: np.ndarray[Any, Any]) -> BoundingBox:
    """Return an exclusive pixel-space xyxy box around a non-empty binary mask."""
    rows, columns = np.nonzero(mask)
    if not rows.size:
        raise ValueError("Cannot build a bounding box from an empty mask.")
    return BoundingBox(
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    )
