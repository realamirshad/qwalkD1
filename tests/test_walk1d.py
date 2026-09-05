"""Physics checks on the 1D walk. These pin the spec, not the implementation."""

from __future__ import annotations

import numpy as np
import pytest

from qw.walk1d import (
    N_WALK_FEATURES,
    extract_features,
    n_positions_for,
    position_axis,
    position_probabilities,
    run_walk_1d,
)

TOL = 1e-10


def _params(n: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Random parameters inside the declared ranges."""
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0.0, np.pi, n),
        rng.uniform(0.0, 2.0 * np.pi, n),
        rng.uniform(0.0, np.pi / 2.0, n),
        rng.uniform(0.0, 2.0 * np.pi, n),
    )


@pytest.mark.parametrize("n_steps", [0, 1, 2, 5, 15, 30, 50])
def test_norm_is_preserved(n_steps: int) -> None:
    """sum_x P_T(x) == 1 to within 1e-10, for every sample and every T."""
    theta, phi, alpha, beta = _params(16, seed=n_steps)
    psi = run_walk_1d(theta, phi, alpha, beta, n_steps)
    total = position_probabilities(psi).sum(axis=1)
    assert np.allclose(total, 1.0, atol=TOL, rtol=0.0)


def test_hadamard_peak_position() -> None:
    """alpha=pi/4, beta=0, theta=0, phi=0 at T=30 peaks near x = -0.7*T."""
    n_steps = 30
    psi = run_walk_1d(
        theta=np.array([0.0]),
        phi=np.array([0.0]),
        alpha=np.array([np.pi / 4.0]),
        beta=np.array([0.0]),
        n_steps=n_steps,
    )
    prob = position_probabilities(psi)[0]
    x = position_axis(prob.shape[0])
    peak = x[np.argmax(prob)]

    expected = -0.7 * n_steps
    assert peak == pytest.approx(expected, abs=0.1 * n_steps), (
        f"dominant peak at x={peak}, expected near {expected}"
    )
    # The distribution must be left-skewed, not merely have a left-side peak.
    assert prob[x < 0].sum() > prob[x > 0].sum()


@pytest.mark.parametrize("n_steps", [1, 5, 30, 60])
def test_no_wraparound(n_steps: int) -> None:
    """The padded lattice edges stay at exactly zero probability."""
    theta, phi, alpha, beta = _params(8, seed=n_steps + 100)
    psi = run_walk_1d(theta, phi, alpha, beta, n_steps)
    prob = position_probabilities(psi)

    assert prob.shape[1] == n_positions_for(n_steps)
    assert np.all(prob[:, 0] == 0.0)
    assert np.all(prob[:, -1] == 0.0)

    # Nothing outside the light cone |x| <= T either.
    x = position_axis(prob.shape[1])
    assert np.all(prob[:, np.abs(x) > n_steps] == 0.0)


def test_initial_state_matches_spec() -> None:
    """|psi_0> = |0>_x (x) (cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>)."""
    theta = np.array([0.0, np.pi / 2.0, np.pi])
    phi = np.array([0.0, 0.3, 1.1])
    zeros = np.zeros_like(theta)
    psi = run_walk_1d(theta, phi, zeros, zeros, n_steps=0)

    centre = psi.shape[1] // 2
    assert np.allclose(psi[:, centre, 0], np.cos(theta / 2.0), atol=TOL)
    assert np.allclose(
        psi[:, centre, 1], np.exp(1j * phi) * np.sin(theta / 2.0), atol=TOL
    )
    off_centre = np.delete(np.arange(psi.shape[1]), centre)
    assert np.all(psi[:, off_centre, :] == 0.0)


def test_coin_is_unitary_over_the_range() -> None:
    """C(a, b) as specified is unitary for all (alpha, beta) in range."""
    rng = np.random.default_rng(7)
    for a, b in zip(
        rng.uniform(0.0, np.pi / 2.0, 32), rng.uniform(0.0, 2.0 * np.pi, 32)
    ):
        C = np.array(
            [
                [np.cos(a) * np.exp(1j * b), np.sin(a)],
                [-np.sin(a), np.cos(a) * np.exp(-1j * b)],
            ]
        )
        assert np.allclose(C.conj().T @ C, np.eye(2), atol=TOL)


def test_extract_features_shape_and_finiteness() -> None:
    """extract_features returns (n_samples, 10) with no NaNs or infinities."""
    theta, phi, alpha, beta = _params(24, seed=3)
    psi = run_walk_1d(theta, phi, alpha, beta, n_steps=20)
    feats = extract_features(psi)
    assert feats.shape == (24, N_WALK_FEATURES)
    assert np.all(np.isfinite(feats))


def test_walk_is_deterministic() -> None:
    """Identical parameters give bit-identical states."""
    theta, phi, alpha, beta = _params(4, seed=11)
    a = run_walk_1d(theta, phi, alpha, beta, 25)
    b = run_walk_1d(theta, phi, alpha, beta, 25)
    assert np.array_equal(a, b)
