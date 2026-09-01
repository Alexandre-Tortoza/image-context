from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import image_context.cli as cli
from image_context.cli import _build_parser
from image_context.config import DatasetConfig, PipelineConfig


def test_analyze_strategy_selection() -> None:
    args = _build_parser().parse_args(
        ["analyze", "--image", "frame.png", "--strategy", "region-first"]
    )

    assert args.command == "analyze"
    assert args.strategy == "region-first"


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("baseline", ["baseline"]),
        ("region-first", ["region-first"]),
        ("all", ["baseline", "region-first"]),
    ],
)
def test_analyze_executes_only_selected_strategies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected: list[str],
) -> None:
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.zeros((8, 10, 3), dtype=np.uint8))
    calls: list[str] = []

    def baseline(*args: object, **kwargs: object) -> bool:
        calls.append("baseline")
        return True

    def region_first(*args: object, **kwargs: object) -> bool:
        calls.append("region-first")
        return True

    monkeypatch.setattr(cli, "_run_baseline_image", baseline)
    monkeypatch.setattr(cli, "_run_region_first_image", region_first)
    config = PipelineConfig(
        dataset=DatasetConfig(tmp_path / "unused.bag", "/camera"),
        output_directory=tmp_path / "runs",
        run_id="comparison",
    )
    arguments = Namespace(image=source, strategy=selection, overwrite=False)

    exit_code = cli._run_analyze(config, arguments)

    assert exit_code == 0
    assert calls == expected
    assert (tmp_path / "runs" / "comparison" / "input.png").is_file()
    assert (tmp_path / "runs" / "comparison" / "comparison.json").is_file()


def test_failed_strategy_does_not_delete_other_strategy_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.zeros((8, 10, 3), dtype=np.uint8))
    config = PipelineConfig(
        dataset=DatasetConfig(tmp_path / "unused.bag", "/camera"),
        output_directory=tmp_path / "runs",
        run_id="comparison",
    )
    marker = config.output_directory / config.run_id / "region-first" / "valid.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("valid", encoding="utf-8")
    monkeypatch.setattr(cli, "_run_baseline_image", lambda *args, **kwargs: False)
    monkeypatch.setattr(cli, "_run_region_first_image", lambda *args, **kwargs: True)

    exit_code = cli._run_analyze(
        config, Namespace(image=source, strategy="all", overwrite=False)
    )

    assert exit_code == 2
    assert marker.read_text(encoding="utf-8") == "valid"


def test_region_configuration_has_an_independent_cache_fingerprint(tmp_path: Path) -> None:
    config = PipelineConfig(dataset=DatasetConfig(tmp_path / "bag", "/camera"))
    changed = replace(
        config,
        region_first=replace(config.region_first, maximum_regions=4),
    )

    assert cli._strategy_fingerprint(config, "image-hash", "baseline") == (
        cli._strategy_fingerprint(changed, "image-hash", "baseline")
    )
    assert cli._strategy_fingerprint(config, "image-hash", "region-first") != (
        cli._strategy_fingerprint(changed, "image-hash", "region-first")
    )
