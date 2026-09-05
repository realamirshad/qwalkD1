"""1D discrete-time quantum walk: state-vector simulation and readout.

Fixed physics (challenge spec -- do not "improve"):

    C(a, b) = [[cos(a)*exp(i*b),  sin(a)         ],
               [-sin(a),          cos(a)*exp(-i*b)]]

    |coin_0> = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
    |psi_{t+1}> = S C |psi_t>          (coin first, then shift)
    S: coin |0> moves left, coin |1> moves right
    P_T(x) = sum_c |<x, c | psi_T>|^2

Pure NumPy; one independent walk per dataset row, all rows advanced together
by vectorising over the leading axis.
"""

from __future__ import annotations

import numpy as np

from qw.walkspec import WalkSpec

PARAM_NAMES: tuple[str, ...] = ("theta", "phi", "alpha", "beta")
PARAM_RANGES: tuple[tuple[float, float], ...] = (
    (0.0, np.pi),        # theta
    (0.0, 2.0 * np.pi),  # phi
    (0.0, np.pi / 2.0),  # alpha
    (0.0, 2.0 * np.pi),  # beta
)

#: Number of columns returned by :func:`extract_features`.
N_WALK_FEATURES = 10

FEATURE_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "skew",
    "kurtosis",
    "entropy",
    "ipr",
    "p_max",
    "argmax_x",
    "p_left",
    "p_origin",
)


def n_positions_for(n_steps: int) -> int:
    """Lattice size used by :func:`run_walk_1d`: ``2*n_steps + 3``.

    Two sites of padding beyond the ``2*n_steps + 1`` light cone keep the
    outermost sites unreachable in ``n_steps`` steps, so the periodic shift can
    never wrap probability around the array.
    """
    return 2 * int(n_steps) + 3


def position_axis(n_positions: int) -> np.ndarray:
    """Integer positions ``x`` for a lattice of ``n_positions`` sites, centred on 0."""
    centre = n_positions // 2
    return np.arange(n_positions, dtype=float) - centre


# --- BEGIN pasted implementation -------------------------------------------
# Reference vectorised implementation; replace this block with your own.
# Contract: run_walk_1d(theta, phi, alpha, beta, n_steps) -> (n_samples, n_positions, 2)
#           extract_features(psi) -> (n_samples, 10)


def run_walk_1d(
    theta: np.ndarray,
    phi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    n_steps: int,
    n_positions: int | None = None,
) -> np.ndarray:
    """Run one independent 1D walk per sample for ``n_steps`` steps.

    Args:
        theta: Shape ``(n_samples,)`` initial-coin polar angles.
        phi: Shape ``(n_samples,)`` initial-coin azimuthal angles.
        alpha: Shape ``(n_samples,)`` coin-operator ``a``.
        beta: Shape ``(n_samples,)`` coin-operator ``b``.
        n_steps: Number of walk steps ``T``.
        n_positions: Lattice size; defaults to :func:`n_positions_for`.

    Returns:
        Complex array ``psi`` of shape ``(n_samples, n_positions, 2)``, where
        ``psi[s, x, c]`` is the amplitude of sample ``s`` at lattice index ``x``
        with coin state ``c``.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if not (theta.shape == phi.shape == alpha.shape == beta.shape):
        raise ValueError("theta, phi, alpha, beta must share the same shape")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    n_samples = theta.shape[0]
    n_pos = n_positions_for(n_steps) if n_positions is None else int(n_positions)
    if n_pos < 2 * n_steps + 1:
        raise ValueError(
            f"n_positions={n_pos} is too small for {n_steps} steps; "
            f"probability would wrap around"
        )
    centre = n_pos // 2

    # |psi_0> = |centre> (x) (cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>)
    psi = np.zeros((n_samples, n_pos, 2), dtype=np.complex128)
    psi[:, centre, 0] = np.cos(theta / 2.0)
    psi[:, centre, 1] = np.exp(1j * phi) * np.sin(theta / 2.0)

    # C(alpha, beta), one 2x2 per sample, held as four broadcastable columns.
    cos_a = np.cos(alpha)[:, None]
    sin_a = np.sin(alpha)[:, None]
    c00 = cos_a * np.exp(1j * beta)[:, None]
    c01 = sin_a.astype(np.complex128)
    c10 = -sin_a.astype(np.complex128)
    c11 = cos_a * np.exp(-1j * beta)[:, None]

    for _ in range(n_steps):
        up, down = psi[:, :, 0], psi[:, :, 1]
        coined_up = c00 * up + c01 * down
        coined_down = c10 * up + c11 * down
        # Coin |0> moves left (index -1), coin |1> moves right (index +1).
        # Padding guarantees the wrapped-in sites carry zero amplitude.
        psi = np.stack(
            (np.roll(coined_up, -1, axis=1), np.roll(coined_down, 1, axis=1)),
            axis=-1,
        )

    return psi


def position_probabilities(psi: np.ndarray) -> np.ndarray:
    """``P_T(x) = sum_c |psi[s, x, c]|^2``, shape ``(n_samples, n_positions)``."""
    return np.sum(np.abs(psi) ** 2, axis=-1)


def extract_features(psi: np.ndarray) -> np.ndarray:
    """Summarise each sample's final position distribution as 10 numbers.

    Columns follow :data:`FEATURE_NAMES`: mean, standard deviation, skewness,
    kurtosis (non-excess), Shannon entropy in nats, inverse participation
    ratio, peak probability, peak position, total probability on ``x < 0``, and
    probability at the origin.

    Args:
        psi: Shape ``(n_samples, n_positions, 2)``.

    Returns:
        Shape ``(n_samples, 10)`` float array.
    """
    prob = position_probabilities(psi)
    x = position_axis(prob.shape[1])

    mean = prob @ x
    dx = x[None, :] - mean[:, None]
    var = np.sum(prob * dx**2, axis=1)
    std = np.sqrt(var)
    safe_std = np.where(std > 0, std, 1.0)
    skew = np.sum(prob * dx**3, axis=1) / safe_std**3
    kurt = np.sum(prob * dx**4, axis=1) / safe_std**4

    entropy = -np.sum(prob * np.log(np.where(prob > 0, prob, 1.0)), axis=1)
    ipr = np.sum(prob**2, axis=1)
    p_max = np.max(prob, axis=1)
    argmax_x = x[np.argmax(prob, axis=1)]
    p_left = np.sum(prob[:, x < 0], axis=1)
    p_origin = prob[:, np.argmin(np.abs(x))]

    return np.column_stack(
        [mean, std, skew, kurt, entropy, ipr, p_max, argmax_x, p_left, p_origin]
    )


# --- END pasted implementation ---------------------------------------------


def _run(params: np.ndarray, n_steps: int) -> np.ndarray:
    """Adapt the column-matrix calling convention of :class:`WalkSpec`."""
    theta, phi, alpha, beta = (params[:, i] for i in range(4))
    return run_walk_1d(theta, phi, alpha, beta, n_steps)


WALK_1D = WalkSpec(
    name="1d",
    param_names=PARAM_NAMES,
    param_ranges=PARAM_RANGES,
    n_features_out=N_WALK_FEATURES,
    run=_run,
    extract=extract_features,
)
