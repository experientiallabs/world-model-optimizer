"""Deterministic pre-call feature views for the conditional graded IRT study.

The transform is stateless and accepts prompt text only. Repository identity, model output,
patches, verifier data, and trajectories cannot enter the feature contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

HASH_DIMENSIONS = (512, 2_048)
HASH_NGRAM_RANGE = (3, 5)
PROMPT_SHAPE_DIMENSION = 10


def prompt_shape_features(texts: Sequence[str]) -> np.ndarray:
    """Return the frozen ten-dimensional prompt-shape view."""
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("IRT feature prompts must be nonempty strings")
    rows: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        rows.append(
            [
                math.log1p(len(text)) / 10.0,
                math.log1p(text.count("\n") + 1) / 10.0,
                math.log1p(len(text.split())) / 10.0,
                float("```" in text),
                float("traceback" in lowered or "exception" in lowered),
                float("test" in lowered),
                float("error" in lowered or "fail" in lowered),
                float("implement" in lowered or "add" in lowered),
                float("refactor" in lowered),
                float("performance" in lowered or "optimiz" in lowered),
            ]
        )
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(texts), PROMPT_SHAPE_DIMENSION) or not np.isfinite(result).all():
        raise RuntimeError("IRT prompt-shape transform produced invalid features")
    return result


def signed_hash_features(texts: Sequence[str], *, dimension: int) -> np.ndarray:
    """Return the frozen signed character n-gram hashing view."""
    if dimension not in HASH_DIMENSIONS:
        raise ValueError(f"unsupported frozen IRT hash dimension: {dimension}")
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("IRT feature prompts must be nonempty strings")
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=HASH_NGRAM_RANGE,
        n_features=dimension,
        alternate_sign=True,
        norm="l2",
    )
    result = np.asarray(vectorizer.transform(texts).toarray(), dtype=np.float64)
    if result.shape != (len(texts), dimension) or not np.isfinite(result).all():
        raise RuntimeError("IRT signed hashing produced invalid features")
    if np.any(np.linalg.norm(result, axis=1) <= 0.0):
        raise ValueError("IRT signed hashing produced an empty prompt row")
    return result


def frozen_feature_views(texts: Sequence[str]) -> dict[str, np.ndarray]:
    """Build all four preregistered views without fitting or retaining state."""
    prompts = tuple(texts)
    shape = prompt_shape_features(prompts)
    hash_512 = signed_hash_features(prompts, dimension=512)
    hash_2048 = signed_hash_features(prompts, dimension=2_048)
    combined = np.concatenate([hash_2048, shape], axis=1)
    return {
        "signed-hash-512": hash_512,
        "signed-hash-2048": hash_2048,
        "prompt-shape": shape,
        "combined": combined,
    }


def frozen_feature_view(texts: Sequence[str], *, name: str) -> np.ndarray:
    """Build exactly one frozen view for an online route decision."""
    prompts = tuple(texts)
    if name == "signed-hash-512":
        return signed_hash_features(prompts, dimension=512)
    if name == "signed-hash-2048":
        return signed_hash_features(prompts, dimension=2_048)
    if name == "prompt-shape":
        return prompt_shape_features(prompts)
    if name == "combined":
        return np.concatenate(
            [
                signed_hash_features(prompts, dimension=2_048),
                prompt_shape_features(prompts),
            ],
            axis=1,
        )
    raise ValueError(f"unsupported frozen IRT feature view: {name}")
