"""Dataset loading and the one train/test split every space shares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from qw.config import ExperimentConfig


@dataclass(frozen=True)
class Dataset:
    """A stratified split. Test arrays are never passed to any ``fit``."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def summary(self) -> str:
        """One-line description of split sizes and class balance."""
        pos_train = float(np.mean(self.y_train))
        pos_test = float(np.mean(self.y_test))
        return (
            f"train={self.X_train.shape[0]} test={self.X_test.shape[0]} "
            f"features={self.n_features} "
            f"positive_rate: train={pos_train:.3f} test={pos_test:.3f}"
        )


def load_breast_cancer_split(cfg: ExperimentConfig) -> Dataset:
    """Load Breast Cancer Wisconsin and split it, stratified, on ``cfg.seed``.

    Args:
        cfg: Supplies ``test_size`` and ``seed``.

    Returns:
        A :class:`Dataset` holding the only split the experiment uses.
    """
    bunch = load_breast_cancer()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )
    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=tuple(bunch.feature_names),
        target_names=tuple(bunch.target_names),
    )
