"""Pluggable feature selection returning exactly ``k`` feature indices.

Every selector here is fitted on training data only; :mod:`qw.mapping` is the
only caller, and it fits inside a scikit-learn ``fit`` so test rows are
structurally out of reach.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: Default when nothing is specified.
DEFAULT_SELECTOR = "mutual_info"


def _tree_importance(X: np.ndarray, y: np.ndarray, *, seed: int) -> np.ndarray:
    """Impurity-based importances from a seeded extra-trees ensemble."""
    forest = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    forest.fit(X, y)
    return forest.feature_importances_


def _score_fn(name: str, seed: int) -> ScoreFn:
    if name == "mutual_info":
        return partial(mutual_info_classif, random_state=seed)
    if name == "anova_f":
        return f_classif
    if name == "tree":
        return partial(_tree_importance, seed=seed)
    raise KeyError(f"unknown selector {name!r}; available: {sorted(SELECTORS)}")


SELECTORS: tuple[str, ...] = ("mutual_info", "anova_f", "tree")


def make_selector(name: str, k: int, seed: int) -> SelectKBest:
    """Build an unfitted selector that keeps exactly ``k`` features.

    Args:
        name: One of :data:`SELECTORS`.
        k: Number of features to keep -- the walk's parameter count.
        seed: Seed for the stochastic selectors.

    Returns:
        An unfitted ``SelectKBest``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return SelectKBest(score_func=_score_fn(name, seed), k=k)


def selected_indices(selector: SelectKBest) -> np.ndarray:
    """Indices kept by a fitted selector, ascending. Length is exactly ``k``."""
    return selector.get_support(indices=True)
