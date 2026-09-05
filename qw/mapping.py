"""Feature -> walk-parameter mapping, packaged as a scikit-learn transformer.

``QuantumWalkFeatures`` owns the three things that must never see test data:
the feature selector, the min-max scaler, and (implicitly) the permutation
choice.  All of them are fitted in :meth:`fit`; :meth:`transform` only applies
stored state.  Dropping this object into a ``Pipeline`` makes leakage a
structural impossibility rather than a convention, and makes the 24-permutation
search a plain ``GridSearchCV`` over ``qw__permutation``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from qw.selection import DEFAULT_SELECTOR, make_selector, selected_indices
from qw.walkspec import WalkSpec


class QuantumWalkFeatures(BaseEstimator, TransformerMixin):
    """Expand each row into walk features via one independent quantum walk.

    The per-row pipeline is::

        select k -> min-max to [0, 1] -> permute -> scale to parameter ranges
                 -> build |psi_0> -> T steps -> extract features

    Args:
        spec: Walk variant; fixes the parameter count, names and ranges.
        n_steps: Walk steps ``T``.
        permutation: ``permutation[i]`` is the index, among the selected
            features in ascending-index order, feeding parameter ``i``.
        selector: Feature-selection method key.
        seed: Seed for the stochastic selectors.
    """

    def __init__(
        self,
        spec: WalkSpec,
        n_steps: int = 30,
        permutation: Sequence[int] | None = None,
        selector: str = DEFAULT_SELECTOR,
        seed: int = 42,
    ) -> None:
        self.spec = spec
        self.n_steps = n_steps
        self.permutation = permutation
        self.selector = selector
        self.seed = seed

    def _resolved_permutation(self) -> tuple[int, ...]:
        perm = (
            tuple(range(self.spec.n_params))
            if self.permutation is None
            else self.permutation
        )
        return self.spec.validate_permutation(perm)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumWalkFeatures":
        """Fit selector and scaler. Only ever called with training rows.

        Args:
            X: Shape ``(n_samples, n_features)`` training design matrix.
            y: Shape ``(n_samples,)`` training labels.

        Returns:
            ``self``.
        """
        X = np.asarray(X, dtype=float)
        self.permutation_ = self._resolved_permutation()
        self.selector_ = make_selector(
            self.selector, k=self.spec.n_params, seed=self.seed
        )
        self.selector_.fit(X, y)
        self.indices_ = selected_indices(self.selector_)
        # clip=True keeps test rows inside the declared parameter ranges
        # without ever refitting on them.
        self.scaler_ = MinMaxScaler(clip=True).fit(X[:, self.indices_])
        self.n_features_in_ = X.shape[1]
        return self

    def walk_parameters(self, X: np.ndarray) -> np.ndarray:
        """Physical walk parameters for each row, for inspection and logging.

        Args:
            X: Shape ``(n_samples, n_features_in_)``.

        Returns:
            Shape ``(n_samples, n_params)``, columns ordered as
            ``spec.param_names``, each within its declared range.
        """
        check_is_fitted(self, "scaler_")
        X = np.asarray(X, dtype=float)
        unit = self.scaler_.transform(X[:, self.indices_])
        unit = unit[:, list(self.permutation_)]
        return self.spec.scale_to_ranges(unit)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Run the walks and return the extracted features.

        Args:
            X: Shape ``(n_samples, n_features_in_)``.

        Returns:
            Shape ``(n_samples, spec.n_features_out)``.
        """
        check_is_fitted(self, "scaler_")
        params = self.walk_parameters(X)
        psi = self.spec.run(params, self.n_steps)
        return np.asarray(self.spec.extract(psi), dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the emitted columns, e.g. ``qw1d_00``."""
        check_is_fitted(self, "scaler_")
        return np.array(
            [f"qw{self.spec.name}_{i:02d}" for i in range(self.spec.n_features_out)],
            dtype=object,
        )

    def describe(self) -> str:
        """Log line naming the selected features and the permutation applied."""
        check_is_fitted(self, "scaler_")
        assignment = " ".join(
            f"{name}<-x{self.indices_[src]}"
            for name, src in zip(self.spec.param_names, self.permutation_)
        )
        return (
            f"selector={self.selector} selected={[int(i) for i in self.indices_]} "
            f"perm={list(self.permutation_)} [{assignment}] T={self.n_steps}"
        )
