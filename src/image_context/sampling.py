"""Reproducible sampling policies."""

from __future__ import annotations

import random


def random_sample_indices(total: int, count: int, seed: int) -> tuple[int, ...]:
    """Select sorted, unique indices using a reproducible pseudo-random seed."""
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    if count > total:
        raise ValueError(f"Cannot sample {count} images from a source containing {total} images.")
    return tuple(sorted(random.Random(seed).sample(range(total), count)))
