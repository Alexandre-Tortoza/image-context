from __future__ import annotations

import pytest

from image_context.sampling import random_sample_indices


def test_random_sample_is_sorted_unique_and_reproducible() -> None:
    first = random_sample_indices(total=100, count=10, seed=42)
    second = random_sample_indices(total=100, count=10, seed=42)

    assert first == second
    assert first == tuple(sorted(first))
    assert len(set(first)) == 10


def test_random_sample_rejects_more_items_than_available() -> None:
    with pytest.raises(ValueError, match="Cannot sample"):
        random_sample_indices(total=3, count=4, seed=42)
