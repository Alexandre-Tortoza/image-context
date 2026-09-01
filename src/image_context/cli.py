"""Command-line entry point for extraction and full perception runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

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


if __name__ == "__main__":
    raise SystemExit(main())
