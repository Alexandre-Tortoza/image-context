"""Command-line entry point for extraction and full perception runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from image_context.adapters.rosbag_sampler import RosbagImageSampler
from image_context.artifacts import ArtifactRepository
from image_context.config import PipelineConfig, load_config
from image_context.pipeline import ConsoleProgressReporter, ImageContextPipeline
from image_context.prompts import PROMPT_VERSION


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        config = _apply_overrides(load_config(args.config), args)
        artifacts = _initialize_artifacts(config, overwrite=args.overwrite)
        sampler = RosbagImageSampler(config.dataset.bag_path, config.dataset.image_topic)
        if args.command == "sample":
            samples = sampler.sample(
                config.dataset.sample_size, config.dataset.seed, artifacts.frames_directory
            )
            artifacts.write_selection(samples)
            print(f"Extracted {len(samples)} images to {artifacts.frames_directory}")
            return 0

        # Heavy model modules are imported only for a full run, never for sampling or --help.
        from image_context.adapters.model_factory import TransformersModelFactory

        pipeline = ImageContextPipeline(
            sampler=sampler,
            model_factory=TransformersModelFactory(config.qwen, config.grounding, config.sam2),
            artifacts=artifacts,
            progress=ConsoleProgressReporter(),
            minimum_concept_confidence=config.minimum_concept_confidence,
        )
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
    for name in ("sample", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("config.yaml"))
        command.add_argument("--sample-size", type=int)
        command.add_argument("--seed", type=int)
        command.add_argument("--output", type=Path)
        command.add_argument("--run-id")
        command.add_argument("--overwrite", action="store_true")
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


def _initialize_artifacts(config: PipelineConfig, *, overwrite: bool) -> ArtifactRepository:
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
        "prompt_version": PROMPT_VERSION,
    }
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


if __name__ == "__main__":
    raise SystemExit(main())
