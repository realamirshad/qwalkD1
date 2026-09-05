"""The four feature spaces under comparison.

Each space is a *transformer*, not a matrix: it sits in front of the classifier
inside one ``Pipeline``, so its own fitting happens on training folds only.
``materialize`` is provided for inspection when the actual arrays are wanted.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from qw.config import ExperimentConfig
from qw.mapping import QuantumWalkFeatures

#: Space names, in the order the results table reports them.
SPACES: tuple[str, ...] = ("original", "polynomial", "qw", "original+qw")


def _identity() -> FunctionTransformer:
    """Pass the original columns through unchanged."""
    return FunctionTransformer(feature_names_out="one-to-one")


def _qw(cfg: ExperimentConfig) -> QuantumWalkFeatures:
    return QuantumWalkFeatures(
        spec=cfg.spec,
        n_steps=cfg.n_steps,
        permutation=cfg.permutation,
        selector=cfg.selector,
        seed=cfg.seed,
    )


def build_space(name: str, cfg: ExperimentConfig) -> TransformerMixin:
    """Build the unfitted transformer for one feature space.

    Args:
        name: One of :data:`SPACES`.
        cfg: Supplies walk spec, ``T``, permutation, selector, seed, degree.

    Returns:
        An unfitted transformer mapping the raw design matrix to that space.
    """
    if name == "original":
        return _identity()
    if name == "polynomial":
        return PolynomialFeatures(degree=cfg.poly_degree, include_bias=False)
    if name == "qw":
        return _qw(cfg)
    if name == "original+qw":
        # FeatureUnion concatenates columns: X_original first, then X_QW.
        return FeatureUnion([("original", _identity()), ("qw", _qw(cfg))])
    raise KeyError(f"unknown space {name!r}; available: {list(SPACES)}")


def materialize(
    name: str,
    cfg: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a space on train and return the transformed train/test matrices.

    Only for inspection and plotting -- the experiment itself never handles the
    matrices directly.  ``X_test`` reaches ``transform`` only.

    Args:
        name: One of :data:`SPACES`.
        cfg: Experiment configuration.
        X_train: Training design matrix.
        y_train: Training labels.
        X_test: Test design matrix.

    Returns:
        ``(Z_train, Z_test)`` in the requested space.
    """
    space = build_space(name, cfg)
    Z_train = space.fit_transform(X_train, y_train)
    Z_test = space.transform(X_test)
    return np.asarray(Z_train, dtype=float), np.asarray(Z_test, dtype=float)
