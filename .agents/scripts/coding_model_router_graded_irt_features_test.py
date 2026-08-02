"""Tests for the frozen prompt-only graded IRT features."""

from __future__ import annotations

import numpy as np
import pytest
from coding_model_router_graded_irt_features import (
    frozen_feature_view,
    frozen_feature_views,
    prompt_shape_features,
    signed_hash_features,
)


def test_frozen_views_are_deterministic_and_aligned() -> None:
    prompts = (
        "Implement parser support and add tests.\n```python\npass\n```",
        "Fix the traceback and optimize the slow loop.",
    )
    first = frozen_feature_views(prompts)
    second = frozen_feature_views(prompts)
    assert {name: values.shape for name, values in first.items()} == {
        "signed-hash-512": (2, 512),
        "signed-hash-2048": (2, 2_048),
        "prompt-shape": (2, 10),
        "combined": (2, 2_058),
    }
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert np.array_equal(first["combined"][:, :2_048], first["signed-hash-2048"])
    assert np.array_equal(first["combined"][:, 2_048:], first["prompt-shape"])
    assert np.allclose(np.linalg.norm(first["signed-hash-512"], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(first["signed-hash-2048"], axis=1), 1.0)
    assert all(
        np.array_equal(frozen_feature_view(prompts, name=name), values)
        for name, values in first.items()
    )


def test_prompt_shape_uses_only_text_observables() -> None:
    rows = prompt_shape_features(
        (
            "Refactor implementation performance and add a test.",
            "Exception traceback: operation failed with an error.",
        )
    )
    assert rows.shape == (2, 10)
    assert rows[0, 5] == rows[0, 7] == rows[0, 8] == rows[0, 9] == 1.0
    assert rows[1, 4] == rows[1, 6] == 1.0


@pytest.mark.parametrize("dimension", [1, 511, 1_024, 4_096])
def test_signed_hash_rejects_unfrozen_dimensions(dimension: int) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        signed_hash_features(("prompt",), dimension=dimension)


@pytest.mark.parametrize("prompts", [(), ("",), ("ok", "")])
def test_feature_views_reject_empty_prompts(prompts: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="nonempty strings"):
        frozen_feature_views(prompts)
