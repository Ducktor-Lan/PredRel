"""Deterministic synthetic tasks used to validate Week 2 anatomy metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .synthetic_types import ContextCompetitionFixture, ContextCondition, SyntheticFixture


def _require_even(value: int, *, name: str, minimum: int = 2) -> None:
    if not isinstance(value, int) or value < minimum or value % 2:
        raise ValueError(f"{name} must be an even integer >= {minimum}")


def _class_ids(prefix: str, labels: np.ndarray, *, role: str) -> tuple[str, ...]:
    per_class: dict[int, int] = {}
    ids: list[str] = []
    for label in labels.tolist():
        index = per_class.get(int(label), 0)
        ids.append(f"{prefix}-{role}-c{int(label)}-{index:03d}")
        per_class[int(label)] = index + 1
    return tuple(ids)


def make_class_only(*, seed: int, support_per_class: int, query_per_class: int) -> SyntheticFixture:
    """Create exact same-class feature duplicates to make exchangeability explicit."""

    del seed  # The construction is intentionally deterministic and noise-free.
    _require_even(support_per_class, name="support_per_class", minimum=2)
    _require_even(query_per_class, name="query_per_class", minimum=2)
    class_centres = np.asarray([[-1.0, 0.0, 0.25], [1.0, 0.0, -0.25]], dtype=np.float64)
    support_labels = np.repeat(np.asarray([0, 1], dtype=np.int64), support_per_class)
    query_labels = np.repeat(np.asarray([0, 1], dtype=np.int64), query_per_class)
    support_features = np.vstack(
        [np.repeat(class_centres[label][np.newaxis, :], support_per_class, axis=0) for label in (0, 1)]
    )
    query_features = np.vstack(
        [np.repeat(class_centres[label][np.newaxis, :], query_per_class, axis=0) for label in (0, 1)]
    )
    return SyntheticFixture(
        name="class-only",
        description="Within each class all feature vectors are exactly equal, so sample identity has no valid signal.",
        support_features=support_features,
        support_labels=support_labels,
        support_ids=_class_ids("class-only", support_labels, role="support"),
        query_features=query_features,
        query_labels=query_labels,
        query_ids=_class_ids("class-only", query_labels, role="query"),
        support_groups={"exchangeability_class": support_labels.copy()},
        query_groups={"exchangeability_class": query_labels.copy()},
        expectations={
            "within_class_beta": "uniform up to numerical tolerance",
            "class_only_js": "approximately zero",
        },
    )


def make_prototype(*, seed: int, support_per_class: int, query_per_class: int) -> SyntheticFixture:
    """Create two distinct, same-label prototypes per class."""

    _require_even(support_per_class, name="support_per_class", minimum=4)
    _require_even(query_per_class, name="query_per_class", minimum=2)
    rng = np.random.default_rng(seed)
    centres = {
        (0, 0): np.asarray([-2.6, -1.2, 0.0], dtype=np.float64),
        (0, 1): np.asarray([-2.6, 1.2, 0.0], dtype=np.float64),
        (1, 0): np.asarray([2.6, -1.2, 0.0], dtype=np.float64),
        (1, 1): np.asarray([2.6, 1.2, 0.0], dtype=np.float64),
    }
    support_rows: list[np.ndarray] = []
    support_labels: list[int] = []
    support_prototypes: list[int] = []
    query_rows: list[np.ndarray] = []
    query_labels: list[int] = []
    query_prototypes: list[int] = []
    for label in (0, 1):
        for prototype in (0, 1):
            support_rows.append(
                rng.normal(centres[(label, prototype)], [0.18, 0.18, 0.10], size=(support_per_class // 2, 3))
            )
            support_labels.extend([label] * (support_per_class // 2))
            support_prototypes.extend([prototype] * (support_per_class // 2))
            query_rows.append(
                rng.normal(centres[(label, prototype)], [0.13, 0.13, 0.08], size=(query_per_class // 2, 3))
            )
            query_labels.extend([label] * (query_per_class // 2))
            query_prototypes.extend([prototype] * (query_per_class // 2))
    support_label_array = np.asarray(support_labels, dtype=np.int64)
    query_label_array = np.asarray(query_labels, dtype=np.int64)
    return SyntheticFixture(
        name="prototype",
        description="Each label has two separated subclusters; same-label matching-prototype support should be preferred.",
        support_features=np.vstack(support_rows),
        support_labels=support_label_array,
        support_ids=_class_ids("prototype", support_label_array, role="support"),
        query_features=np.vstack(query_rows),
        query_labels=query_label_array,
        query_ids=_class_ids("prototype", query_label_array, role="query"),
        support_groups={"prototype": np.asarray(support_prototypes, dtype=np.int64)},
        query_groups={"prototype": np.asarray(query_prototypes, dtype=np.int64)},
        expectations={
            "matching_prototype_beta": "greater than nonmatching same-label prototype beta",
            "class_only_js": "positive for at least some queries",
        },
    )


def _boundary_rows(
    rng: np.random.Generator,
    *,
    per_class: int,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create rows where absolute first coordinate is the known boundary distance."""

    # Half of each class is deliberately near the x=0 decision boundary.
    near_count = per_class // 2
    far_count = per_class - near_count
    distances = np.concatenate(
        [rng.uniform(0.05, 0.45, size=near_count), rng.uniform(0.85, 2.25, size=far_count)]
    )
    rows: list[np.ndarray] = []
    labels: list[int] = []
    row_distances: list[float] = []
    for label, sign in ((0, -1.0), (1, 1.0)):
        jitter = rng.normal(0.0, 0.08 if role == "query" else 0.12, size=per_class)
        x = sign * distances + jitter
        # Preserve label-side consistency after jitter while retaining a known distance value.
        x = sign * np.maximum(np.abs(x), 0.015)
        y = rng.normal(0.0, 0.95, size=per_class)
        z = rng.normal(sign * 0.15, 0.25, size=per_class)
        rows.append(np.column_stack([x, y, z]))
        labels.extend([label] * per_class)
        row_distances.extend(np.abs(x).tolist())
    return (
        np.vstack(rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(row_distances, dtype=np.float64),
    )


def make_boundary(*, seed: int, support_per_class: int, query_per_class: int) -> SyntheticFixture:
    """Create a binary task with a declarative x=0 decision boundary."""

    _require_even(support_per_class, name="support_per_class", minimum=4)
    _require_even(query_per_class, name="query_per_class", minimum=2)
    rng = np.random.default_rng(seed + 101)
    support_features, support_labels, support_distance = _boundary_rows(
        rng, per_class=support_per_class, role="support"
    )
    query_features, query_labels, query_distance = _boundary_rows(
        rng, per_class=query_per_class, role="query"
    )
    return SyntheticFixture(
        name="boundary",
        description="The first feature's absolute value is the declared distance from the x=0 class boundary.",
        support_features=support_features,
        support_labels=support_labels,
        support_ids=_class_ids("boundary", support_labels, role="support"),
        query_features=query_features,
        query_labels=query_labels,
        query_ids=_class_ids("boundary", query_labels, role="query"),
        support_groups={"boundary_distance": support_distance},
        query_groups={"boundary_distance": query_distance},
        expectations={
            "report": "report alpha-weighted support distance, unweighted distance, and rank correlations; do not predeclare a direction",
        },
    )


def make_context_competition(
    *, seed: int, context_replicates: int, context_background_per_class: int
) -> ContextCompetitionFixture:
    """Fix a query and anchors while changing only competing context rows."""

    if not isinstance(context_replicates, int) or context_replicates < 2:
        raise ValueError("context_replicates must be an integer >= 2")
    _require_even(context_background_per_class, name="context_background_per_class", minimum=4)
    rng = np.random.default_rng(seed + 307)
    anchor_features = np.asarray(
        [
            [-1.70, -0.35, 0.00],
            [-1.20, 0.30, 0.05],
            [-0.85, -0.05, -0.10],
            [-1.45, 0.65, 0.15],
            [0.85, -0.25, 0.00],
            [1.20, 0.30, -0.05],
            [1.65, -0.45, 0.10],
            [1.35, 0.70, -0.15],
        ],
        dtype=np.float64,
    )
    anchor_labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    anchor_ids = tuple(f"context-anchor-{index:02d}" for index in range(anchor_features.shape[0]))
    conditions: list[ContextCondition] = []
    for condition_index in range(context_replicates):
        phase = condition_index % 3
        # The class balance stays constant.  Only the remaining observations'
        # feature distribution changes, avoiding a class-count confound.
        if phase == 0:
            class_zero_center, class_one_center = (-2.7, 0.0), (2.7, 0.0)
        elif phase == 1:
            class_zero_center, class_one_center = (-0.45, -0.65), (2.7, 0.0)
        else:
            class_zero_center, class_one_center = (-2.7, 0.0), (0.45, 0.65)
        background_zero = rng.normal(
            [class_zero_center[0], class_zero_center[1], 0.0], [0.34, 0.42, 0.18], size=(context_background_per_class, 3)
        )
        background_one = rng.normal(
            [class_one_center[0], class_one_center[1], 0.0], [0.34, 0.42, 0.18], size=(context_background_per_class, 3)
        )
        features = np.vstack([anchor_features, background_zero, background_one])
        labels = np.concatenate(
            [anchor_labels, np.zeros(context_background_per_class, dtype=np.int64), np.ones(context_background_per_class, dtype=np.int64)]
        )
        background_ids = tuple(
            f"context-{condition_index:02d}-background-c{label}-{row:03d}"
            for label in (0, 1)
            for row in range(context_background_per_class)
        )
        conditions.append(
            ContextCondition(
                condition_id=f"context-{condition_index:02d}-phase-{phase}",
                support_features=features,
                support_labels=labels,
                support_ids=anchor_ids + background_ids,
                anchor_ids=anchor_ids,
            )
        )
    return ContextCompetitionFixture(
        name="context-competition",
        description="Eight anchors and the query are fixed while balanced non-anchor contexts change across conditions.",
        query_features=np.asarray([[-0.72, 0.02, 0.02]], dtype=np.float64),
        query_labels=np.asarray([0], dtype=np.int64),
        query_ids=("context-fixed-query-000",),
        conditions=tuple(conditions),
        expectations={
            "raw_scores": "compare raw-score variation/ranks separately from alpha changes",
            "alpha": "alpha may move through softmax competition even when raw anchor scores are stable",
        },
    )


_FIXTURE_BUILDERS: dict[str, Callable[..., SyntheticFixture]] = {
    "class-only": make_class_only,
    "prototype": make_prototype,
    "boundary": make_boundary,
}


def required_fixture_names() -> tuple[str, ...]:
    return ("class-only", "prototype", "boundary", "context-competition")


def make_fixture(
    name: str,
    *,
    seed: int,
    support_per_class: int,
    query_per_class: int,
) -> SyntheticFixture:
    """Build one non-context synthetic fixture by its required public name."""

    try:
        builder = _FIXTURE_BUILDERS[name]
    except KeyError as error:
        raise ValueError(f"Unknown ordinary fixture {name!r}; expected one of {tuple(_FIXTURE_BUILDERS)}") from error
    return builder(seed=seed, support_per_class=support_per_class, query_per_class=query_per_class)

