"""Typed YAML configuration for the executable pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """ROS bag image source settings."""

    bag_path: Path
    image_topic: str
    sample_size: int = 10
    seed: int = 42


@dataclass(frozen=True)
class QwenConfig:
    """Qwen model and generation settings."""

    checkpoint: str = "Qwen/Qwen3-VL-4B-Instruct"
    quantization: str = "int4"
    device: str = "cuda"
    max_new_tokens: int = 2048


@dataclass(frozen=True)
class GroundingConfig:
    """Grounding DINO settings."""

    checkpoint: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.60
    fallback_box_threshold: float = 0.40
    maximum_detections_per_concept: int = 3


@dataclass(frozen=True)
class Sam2Config:
    """SAM2 settings."""

    checkpoint: str = "facebook/sam2.1-hiera-tiny"
    device: str = "cuda"


@dataclass(frozen=True)
class RegionFirstConfig:
    """Class-agnostic discovery, dense feature, and local analysis settings."""

    feature_checkpoint: str = "facebook/dinov2-small"
    feature_device: str = "cuda"
    discovery_grid_size: int = 4
    maximum_regions: int = 12
    minimum_region_area_fraction: float = 0.002
    maximum_region_area_fraction: float = 0.80
    minimum_region_confidence: float = 0.70
    minimum_region_stability: float = 0.90
    mask_iou_threshold: float = 0.85


@dataclass(frozen=True)
class PipelineConfig:
    """Complete run configuration."""

    dataset: DatasetConfig
    qwen: QwenConfig = QwenConfig()
    grounding: GroundingConfig = GroundingConfig()
    sam2: Sam2Config = Sam2Config()
    region_first: RegionFirstConfig = RegionFirstConfig()
    minimum_concept_confidence: float = 0.60
    indoor_fallback_queries: tuple[str, ...] = ()
    outdoor_fallback_queries: tuple[str, ...] = ()
    output_directory: Path = Path("runs")
    run_id: str = "corridor02-sample"


def load_config(path: Path) -> PipelineConfig:
    """Load configuration and resolve filesystem paths relative to the YAML file."""
    with path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be an object.")
    payload = cast(dict[str, Any], raw)
    dataset = _mapping(payload, "dataset")
    qwen = _mapping(payload, "qwen", required=False)
    grounding = _mapping(payload, "grounding", required=False)
    sam2 = _mapping(payload, "sam2", required=False)
    region_first = _mapping(payload, "region_first", required=False)
    base = path.resolve().parent
    bag_path = _resolved_path(base, _string(dataset, "bag_path"))
    output_directory = _resolved_path(base, str(payload.get("output_directory", "runs")))
    config = PipelineConfig(
        dataset=DatasetConfig(
            bag_path=bag_path,
            image_topic=_string(dataset, "image_topic"),
            sample_size=int(dataset.get("sample_size", 10)),
            seed=int(dataset.get("seed", 42)),
        ),
        qwen=QwenConfig(**qwen),
        grounding=GroundingConfig(**grounding),
        sam2=Sam2Config(**sam2),
        region_first=RegionFirstConfig(**region_first),
        minimum_concept_confidence=float(payload.get("minimum_concept_confidence", 0.60)),
        indoor_fallback_queries=tuple(payload.get("indoor_fallback_queries", [])),
        outdoor_fallback_queries=tuple(payload.get("outdoor_fallback_queries", [])),
        output_directory=output_directory,
        run_id=str(payload.get("run_id", "corridor02-sample")),
    )
    _validate(config)
    return config


def _validate(config: PipelineConfig) -> None:
    if config.dataset.sample_size <= 0:
        raise ValueError("dataset.sample_size must be positive.")
    if not config.dataset.image_topic:
        raise ValueError("dataset.image_topic must not be empty.")
    if config.qwen.quantization not in {"none", "int4"}:
        raise ValueError("qwen.quantization must be 'none' or 'int4'.")
    if not 0.0 <= config.minimum_concept_confidence <= 1.0:
        raise ValueError("minimum_concept_confidence must be between 0.0 and 1.0.")
    for name, threshold in (
        ("grounding.box_threshold", config.grounding.box_threshold),
        ("grounding.text_threshold", config.grounding.text_threshold),
        ("grounding.nms_iou_threshold", config.grounding.nms_iou_threshold),
        ("grounding.fallback_box_threshold", config.grounding.fallback_box_threshold),
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0.")
    if config.grounding.maximum_detections_per_concept <= 0:
        raise ValueError("grounding.maximum_detections_per_concept must be positive.")
    if config.region_first.discovery_grid_size <= 0:
        raise ValueError("region_first.discovery_grid_size must be positive.")
    if config.region_first.maximum_regions <= 0:
        raise ValueError("region_first.maximum_regions must be positive.")
    if not 0.0 <= config.region_first.mask_iou_threshold <= 1.0:
        raise ValueError("region_first.mask_iou_threshold must be between 0.0 and 1.0.")
    for name, threshold in (
        ("minimum_region_confidence", config.region_first.minimum_region_confidence),
        ("minimum_region_stability", config.region_first.minimum_region_stability),
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"region_first.{name} must be between 0.0 and 1.0.")
    if not (
        0.0
        < config.region_first.minimum_region_area_fraction
        < config.region_first.maximum_region_area_fraction
        <= 1.0
    ):
        raise ValueError("region_first area fractions must satisfy 0 < minimum < maximum <= 1.")
    fallback_queries = config.indoor_fallback_queries + config.outdoor_fallback_queries
    invalid_fallback = any(
        not isinstance(query, str) or not query.strip()
        for query in fallback_queries
    )
    if invalid_fallback:
        raise ValueError("fallback query lists must contain only non-empty strings.")


def _mapping(payload: dict[str, Any], name: str, *, required: bool = True) -> dict[str, Any]:
    value = payload.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field '{name}' must be an object.")
    return cast(dict[str, Any], value)


def _string(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration field '{name}' must be a non-empty string.")
    return value


def _resolved_path(base: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (base / path).resolve()
