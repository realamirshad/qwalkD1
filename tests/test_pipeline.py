"""Leakage, reproducibility and configuration guarantees."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from qw.config import ExperimentConfig
from qw.data import load_breast_cancer_split
from qw.mapping import QuantumWalkFeatures
from qw.spaces import SPACES, build_space, materialize

CFG = ExperimentConfig(n_steps=8)


@pytest.fixture(scope="module")
def data():
    return load_breast_cancer_split(CFG)


def test_selector_and_scaler_never_see_test_rows(data, monkeypatch) -> None:
    """Every fit inside the space transformers is handed training rows only."""
    seen: list[int] = []
    original_fit = MinMaxScaler.fit

    def spy(self, X, y=None):
        seen.append(X.shape[0])
        return original_fit(self, X, y)

    monkeypatch.setattr(MinMaxScaler, "fit", spy)

    qw = build_space("qw", CFG)
    qw.fit(data.X_train, data.y_train)
    qw.transform(data.X_test)

    assert seen, "scaler was never fitted"
    assert all(n == data.X_train.shape[0] for n in seen), (
        f"a scaler saw {seen} rows; training set has {data.X_train.shape[0]}"
    )


def test_transform_is_row_independent(data) -> None:
    """One walk per row: transforming a subset equals subsetting the transform."""
    qw = build_space("qw", CFG).fit(data.X_train, data.y_train)
    full = qw.transform(data.X_test)
    subset = qw.transform(data.X_test[:5])
    assert np.allclose(full[:5], subset, atol=1e-12)


def test_selection_uses_exactly_n_params(data) -> None:
    """Exactly one feature is selected per walk parameter."""
    qw = build_space("qw", CFG).fit(data.X_train, data.y_train)
    assert len(qw.indices_) == CFG.spec.n_params == 4
    assert len(set(qw.indices_)) == 4


def test_parameters_stay_inside_declared_ranges(data) -> None:
    """Test rows map inside [lo, hi] for every parameter, thanks to clipping."""
    qw = build_space("qw", CFG).fit(data.X_train, data.y_train)
    params = qw.walk_parameters(data.X_test)
    for i, (lo, hi) in enumerate(CFG.spec.param_ranges):
        assert params[:, i].min() >= lo - 1e-12
        assert params[:, i].max() <= hi + 1e-12


def test_permutation_changes_the_mapping(data) -> None:
    """The permutation is a real, loggable knob, not a no-op."""
    identity = QuantumWalkFeatures(
        spec=CFG.spec, n_steps=CFG.n_steps, permutation=(0, 1, 2, 3), seed=CFG.seed
    ).fit(data.X_train, data.y_train)
    swapped = QuantumWalkFeatures(
        spec=CFG.spec, n_steps=CFG.n_steps, permutation=(3, 2, 1, 0), seed=CFG.seed
    ).fit(data.X_train, data.y_train)

    assert not np.allclose(
        identity.walk_parameters(data.X_test), swapped.walk_parameters(data.X_test)
    )
    assert "theta<-" in identity.describe()


def test_invalid_permutation_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(permutation=(0, 1, 2, 2))
    with pytest.raises(ValueError):
        ExperimentConfig(permutation=(0, 1, 2))


@pytest.mark.parametrize("space", SPACES)
def test_spaces_have_expected_widths(space, data) -> None:
    """Each space produces the column count the design implies."""
    Z_train, Z_test = materialize(space, CFG, data.X_train, data.y_train, data.X_test)
    n_raw = data.n_features
    expected = {
        "original": n_raw,
        "polynomial": n_raw + n_raw * (n_raw + 1) // 2,
        "qw": CFG.spec.n_features_out,
        "original+qw": n_raw + CFG.spec.n_features_out,
    }[space]
    assert Z_train.shape == (data.X_train.shape[0], expected)
    assert Z_test.shape == (data.X_test.shape[0], expected)


def test_original_plus_qw_is_concatenation(data) -> None:
    """The combined space is column concatenation, originals first."""
    Z_orig, _ = materialize("original", CFG, data.X_train, data.y_train, data.X_test)
    Z_qw, _ = materialize("qw", CFG, data.X_train, data.y_train, data.X_test)
    Z_both, _ = materialize("original+qw", CFG, data.X_train, data.y_train, data.X_test)
    assert np.allclose(Z_both, np.hstack([Z_orig, Z_qw]))


def test_runs_are_reproducible(data) -> None:
    """Same config, same seed, bit-identical features."""
    a, _ = materialize("qw", CFG, data.X_train, data.y_train, data.X_test)
    b, _ = materialize("qw", CFG, data.X_train, data.y_train, data.X_test)
    assert np.array_equal(a, b)
