"""Command-line entry point for extraction and full perception runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import cv2

from image_context.adapters.rosbag_sampler import RosbagImageSampler
from image_context.artifacts import ArtifactRepository
from image_context.config import PipelineConfig, load_config
from image_context.models import ImageSample, PassName, VlmPassResult
from image_context.parsing import parse_vlm_response
from image_context.pipeline import ConsoleProgressReporter, ImageContextPipeline
from image_context.prompts import PROMPT_VERSION


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        config = _apply_overrides(load_config(args.config), args)
        if args.command == "analyze":
            return _run_analyze(config, args)
        source_run = None if args.command != "reprocess" else args.source_run.resolve()
        artifacts = _initialize_artifacts(
            config, overwrite=args.overwrite, source_run=source_run
        )
        if args.command == "sample":
            sampler = RosbagImageSampler(config.dataset.bag_path, config.dataset.image_topic)
            samples = sampler.sample(
                config.dataset.sample_size, config.dataset.seed, artifacts.frames_directory
            )
            artifacts.write_selection(samples)
            print(f"Extracted {len(samples)} images to {artifacts.frames_directory}")
            return 0

        # Heavy model modules are imported only for a full run, never for sampling or --help.
        from image_context.adapters.model_factory import TransformersModelFactory

        sampler = RosbagImageSampler(config.dataset.bag_path, config.dataset.image_topic)
        pipeline = ImageContextPipeline(
            sampler=sampler,
            model_factory=TransformersModelFactory(config.qwen, config.grounding, config.sam2),
            artifacts=artifacts,
            progress=ConsoleProgressReporter(),
            minimum_concept_confidence=config.minimum_concept_confidence,
            indoor_fallback_queries=config.indoor_fallback_queries,
            outdoor_fallback_queries=config.outdoor_fallback_queries,
        )
        if args.command == "reprocess":
            samples, pass_results = _load_source_vlm_results(source_run, artifacts)
            outcome = pipeline.execute_from_results(samples, pass_results)
        else:
            outcome = pipeline.execute(config.dataset.sample_size, config.dataset.seed)
        print(
            f"Run complete: {len(outcome.completed_frame_ids)} completed, "
            f"{len(outcome.failed_frame_ids)} failed. Artifacts: {artifacts.run_directory}"
        )
        return 0 if not outcome.failed_frame_ids else 2
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-context",
        description="Sample ROS bag images and process them with Qwen, Grounding DINO, and SAM2.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("sample", "run", "reprocess"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("config.yaml"))
        command.add_argument("--sample-size", type=int)
        command.add_argument("--seed", type=int)
        command.add_argument("--output", type=Path)
        command.add_argument("--run-id")
        command.add_argument("--overwrite", action="store_true")
        if name == "reprocess":
            command.add_argument("--source-run", type=Path, required=True)
    analyze = subparsers.add_parser(
        "analyze", description="Analyze one image with baseline, region-first, or both."
    )
    analyze.add_argument("--config", type=Path, default=Path("config.yaml"))
    analyze.add_argument("--image", type=Path, required=True)
    analyze.add_argument(
        "--strategy", choices=("baseline", "region-first", "all"), default="all"
    )
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--run-id")
    analyze.add_argument("--overwrite", action="store_true")
    analyze.set_defaults(sample_size=None, seed=None)
    return parser


def _apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    dataset = replace(
        config.dataset,
        sample_size=(
            config.dataset.sample_size if args.sample_size is None else args.sample_size
        ),
        seed=config.dataset.seed if args.seed is None else args.seed,
    )
    output = config.output_directory if args.output is None else args.output.resolve()
    run_id = config.run_id if args.run_id is None else args.run_id
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be a single non-empty directory name.")
    return replace(config, dataset=dataset, output_directory=output, run_id=run_id)


def _initialize_artifacts(
    config: PipelineConfig, *, overwrite: bool, source_run: Path | None = None
) -> ArtifactRepository:
    configuration = {
        "dataset": {
            "bag_path": str(config.dataset.bag_path),
            "image_topic": config.dataset.image_topic,
            "sample_size": config.dataset.sample_size,
            "seed": config.dataset.seed,
        },
        "qwen": vars(config.qwen),
        "grounding": vars(config.grounding),
        "sam2": vars(config.sam2),
        "minimum_concept_confidence": config.minimum_concept_confidence,
        "indoor_fallback_queries": list(config.indoor_fallback_queries),
        "outdoor_fallback_queries": list(config.outdoor_fallback_queries),
        "prompt_version": PROMPT_VERSION,
    }
    if source_run is not None:
        configuration["reprocessed_from"] = str(source_run)
        source_manifest_path = source_run / "manifest.json"
        if source_manifest_path.is_file():
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            configuration["source_fingerprint"] = source_manifest.get("fingerprint")
            configuration["source_prompt_version"] = source_manifest.get(
                "configuration", {}
            ).get("prompt_version")
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    artifacts = ArtifactRepository(config.output_directory / config.run_id)
    artifacts.initialize(
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "model_loading_strategy": "one-at-a-time",
            "configuration": configuration,
        },
        overwrite=overwrite,
    )
    return artifacts


def _load_source_vlm_results(
    source_run: Path | None, artifacts: ArtifactRepository
) -> tuple[tuple[ImageSample, ...], dict[str, tuple[VlmPassResult, ...]]]:
    if source_run is None:
        raise ValueError("reprocess requires --source-run.")
    selection_path = source_run / "selected_frames.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Source selection not found: '{selection_path}'.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    samples: list[ImageSample] = []
    results_by_frame: dict[str, tuple[VlmPassResult, ...]] = {}
    pass_names: tuple[PassName, ...] = ("objects", "environment", "risks")
    for item in selection["frames"]:
        frame_id = item["frame_id"]
        source_image = source_run / "frames" / frame_id / "image.png"
        destination = artifacts.frames_directory / frame_id / "image.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination)
        sample = ImageSample(
            frame_id=frame_id,
            source_index=int(item["source_index"]),
            timestamp_ns=int(item["timestamp_ns"]),
            width=int(item["width"]),
            height=int(item["height"]),
            image_path=destination,
        )
        frame_results: list[VlmPassResult] = []
        for pass_name in pass_names:
            source_result_path = source_run / "frames" / frame_id / f"vlm_{pass_name}.json"
            if not source_result_path.is_file():
                raise FileNotFoundError(f"Source VLM result not found: '{source_result_path}'.")
            source_payload = json.loads(source_result_path.read_text(encoding="utf-8"))
            result = parse_vlm_response(source_payload["raw_response"], pass_name)
            artifacts.write_vlm_result(sample, result)
            frame_results.append(result)
        samples.append(sample)
        results_by_frame[frame_id] = tuple(frame_results)
    sample_tuple = tuple(samples)
    artifacts.write_selection(sample_tuple)
    return sample_tuple, results_by_frame


def _run_analyze(config: PipelineConfig, args: argparse.Namespace) -> int:
    """Run selected strategies into independent directories and write a comparison summary."""
    original_source = args.image.resolve()
    if not original_source.is_file():
        raise FileNotFoundError(f"Image not found: '{original_source}'.")
    root = config.output_directory / config.run_id
    root.mkdir(parents=True, exist_ok=True)
    source = root / "input.png"
    pixels = cv2.imread(str(original_source), cv2.IMREAD_COLOR)
    if pixels is None or not cv2.imwrite(str(source), pixels):
        raise OSError(f"Could not normalize input image '{original_source}'.")
    source_fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    selected = (
        ("baseline", "region-first") if args.strategy == "all" else (args.strategy,)
    )
    successful: list[bool] = []
    if "baseline" in selected:
        successful.append(
            _run_baseline_image(
                config, source, root / "baseline", source_fingerprint, args.overwrite
            )
        )
    if "region-first" in selected:
        successful.append(
            _run_region_first_image(
                config, source, root / "region-first", source_fingerprint, args.overwrite
            )
        )
    _write_comparison(root, original_source, source_fingerprint)
    print(f"Analysis complete. Artifacts: {root}")
    return 0 if all(successful) else 2


def _run_baseline_image(
    config: PipelineConfig,
    source: Path,
    output: Path,
    source_fingerprint: str,
    overwrite: bool,
) -> bool:
    from image_context.adapters.model_factory import TransformersModelFactory

    fingerprint = _strategy_fingerprint(config, source_fingerprint, "baseline")
    artifacts = ArtifactRepository(output)
    artifacts.initialize(
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "strategy": "baseline",
            "source_image": str(output / "input.png"),
            "source_fingerprint": source_fingerprint,
            "model_loading_strategy": "one-at-a-time",
            "stage_fingerprints": _stage_fingerprints(
                fingerprint, ("vlm", "grounding", "segmentation")
            ),
        },
        overwrite=overwrite,
    )
    result_path = output / "frames" / "image" / "result.json"
    completion_path = output / "complete.json"
    if _baseline_artifacts_complete(output) and not overwrite:
        print("[baseline] using completed strategy artifacts")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        return bool(completion.get("successful"))
    existing_vlm_results = sum(
        (output / "frames" / "image" / f"vlm_{name}.json").is_file()
        for name in ("objects", "environment", "risks")
    )
    strategy_input = output / "input.png"
    shutil.copy2(source, strategy_input)
    sample = _image_sample(strategy_input)
    _reset_cuda_peak_memory()
    started = time.perf_counter()
    pipeline = ImageContextPipeline(
        sampler=RosbagImageSampler(config.dataset.bag_path, config.dataset.image_topic),
        model_factory=TransformersModelFactory(config.qwen, config.grounding, config.sam2),
        artifacts=artifacts,
        progress=ConsoleProgressReporter(),
        minimum_concept_confidence=config.minimum_concept_confidence,
        indoor_fallback_queries=config.indoor_fallback_queries,
        outdoor_fallback_queries=config.outdoor_fallback_queries,
    )
    outcome = pipeline.analyze(sample)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    segmentations = result.get("segmentations", [])
    segmented_ids = {item.get("detection_id") for item in segmentations}
    detections = [
        item for item in result.get("detections", []) if item.get("detection_id") in segmented_ids
    ]
    _write_json_file(
        output / "metrics.json",
        {
            "strategy": "baseline",
            "elapsed_seconds": time.perf_counter() - started,
            "completed_images": len(outcome.completed_frame_ids),
            "failed_images": len(outcome.failed_frame_ids),
            "stage_seconds": pipeline.stage_seconds,
            "semantic_regions": len(segmentations),
            "labeled_regions": len(segmentations),
            "average_confidence": (
                sum(float(item.get("confidence", 0.0)) for item in detections) / len(detections)
                if detections
                else 0.0
            ),
            "vlm_calls": 3 - existing_vlm_results,
            "peak_vram_bytes": _cuda_peak_memory(),
        },
    )
    successful = not outcome.failed_frame_ids
    _write_json_file(
        completion_path,
        {"status": "complete" if successful else "failed", "successful": successful},
    )
    return successful


def _run_region_first_image(
    config: PipelineConfig,
    source: Path,
    output: Path,
    source_fingerprint: str,
    overwrite: bool,
) -> bool:
    from image_context.adapters.region_first import TransformersRegionFirstModelFactory
    from image_context.region_pipeline import RegionFirstContextPipeline
    from image_context.region_prompts import REGION_PROMPT_VERSION

    fingerprint = _strategy_fingerprint(config, source_fingerprint, "region-first")
    manifest_path = output / "manifest.json"
    if output.exists() and overwrite:
        shutil.rmtree(output)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Strategy directory '{output}' has a different configuration. "
                "Use --overwrite or another --run-id."
            )
        completion_path = output / "complete.json"
        if _region_artifacts_complete(output):
            print("[region-first] using completed strategy artifacts")
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            return bool(completion.get("successful"))
    output.mkdir(parents=True, exist_ok=True)
    stage_fingerprints = _stage_fingerprints(
        fingerprint,
        ("region-discovery", "dense-features", "vlm-global-local", "fusion"),
    )
    _write_json_file(
        manifest_path,
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "strategy": "region-first",
            "source_image": str(output / "input.png"),
            "source_fingerprint": source_fingerprint,
            "prompt_version": REGION_PROMPT_VERSION,
            "model_loading_strategy": "one-at-a-time",
            "stage_fingerprints": stage_fingerprints,
        },
    )
    strategy_input = output / "input.png"
    shutil.copy2(source, strategy_input)
    sample = _image_sample(strategy_input)
    _reset_cuda_peak_memory()
    pipeline = RegionFirstContextPipeline(
        TransformersRegionFirstModelFactory(config.sam2, config.region_first, config.qwen),
        output,
        stage_fingerprints,
    )
    outcome = pipeline.analyze(sample)
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["peak_vram_bytes"] = _cuda_peak_memory()
    _write_json_file(metrics_path, metrics)
    return not outcome.discovered_region_count or bool(outcome.semantic_regions)


def _strategy_fingerprint(
    config: PipelineConfig, source_fingerprint: str, strategy: str
) -> str:
    payload: dict[str, object] = {
        "source": source_fingerprint,
        "strategy": strategy,
        "qwen": vars(config.qwen),
        "sam2": vars(config.sam2),
    }
    if strategy == "baseline":
        from image_context.prompts import PROMPT_VERSION

        payload["grounding"] = vars(config.grounding)
        payload["minimum_concept_confidence"] = config.minimum_concept_confidence
        payload["indoor_fallback_queries"] = config.indoor_fallback_queries
        payload["outdoor_fallback_queries"] = config.outdoor_fallback_queries
        payload["prompt_version"] = PROMPT_VERSION
    else:
        from image_context.region_prompts import REGION_PROMPT_VERSION

        payload["region_first"] = vars(config.region_first)
        payload["prompt_version"] = REGION_PROMPT_VERSION
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_comparison(root: Path, source: Path, source_fingerprint: str) -> None:
    strategies: dict[str, object] = {}
    for name in ("baseline", "region-first"):
        metrics_path = root / name / "metrics.json"
        manifest_path = root / name / "manifest.json"
        complete_path = root / name / "complete.json"
        artifacts_complete = (
            _baseline_artifacts_complete(root / name)
            if name == "baseline"
            else _region_artifacts_complete(root / name)
        )
        if (
            artifacts_complete
            and metrics_path.is_file()
            and manifest_path.is_file()
            and complete_path.is_file()
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_fingerprint") != source_fingerprint:
                continue
            strategies[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
    _write_json_file(
        root / "comparison.json",
        {
            "schema_version": 1,
            "source_image": str(source),
            "source_fingerprint": source_fingerprint,
            "strategies": strategies,
        },
    )


def _stage_fingerprints(fingerprint: str, stages: tuple[str, ...]) -> dict[str, str]:
    return {
        stage: hashlib.sha256(f"{fingerprint}:{stage}".encode()).hexdigest()
        for stage in stages
    }


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _image_sample(source: Path) -> ImageSample:
    pixels = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if pixels is None:
        raise FileNotFoundError(f"Image not found or unreadable: '{source}'.")
    height, width = pixels.shape[:2]
    return ImageSample("image", 0, 0, width, height, source)


def _reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_memory() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def _baseline_artifacts_complete(output: Path) -> bool:
    required = (
        output / "complete.json",
        output / "metrics.json",
        output / "frames" / "image" / "result.json",
        output / "frames" / "image" / "overlay.png",
    )
    if not all(path.is_file() for path in required):
        return False
    result = json.loads(required[2].read_text(encoding="utf-8"))
    for segmentation in result.get("segmentations", []):
        mask_path = Path(segmentation.get("mask_path", ""))
        if not mask_path.is_absolute():
            mask_path = output / mask_path
        if not mask_path.is_file():
            return False
    return True


def _region_artifacts_complete(output: Path) -> bool:
    result_path = output / "semantic_regions.json"
    required = (
        output / "complete.json",
        output / "metrics.json",
        output / "global_context.json",
        output / "region_overlay.png",
        result_path,
    )
    if not all(path.is_file() for path in required):
        return False
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if "peak_vram_bytes" not in metrics:
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for region in result.get("semantic_regions", []):
        region_id = region.get("region_id", "")
        associated_paths = (
            output / region.get("mask_path", ""),
            output / region.get("visual_embedding_path", ""),
            output / "regions" / region_id / "context.png",
        )
        if not all(path.is_file() for path in associated_paths):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
