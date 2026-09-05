"""Generate quantum_walk_physics_interface.ipynb -- a single self-contained
notebook exposing quantum_walk_1d / quantum_walk_2d per the physics handoff
spec (docs/physics-handoff-spec.pdf). Regenerate with:

    python tools/make_physics_interface_notebook.py
"""

import json
import pathlib

_n = iter(range(1000))


def _lines(src):
    parts = [ln + "\n" for ln in src.strip().split("\n")]
    parts[-1] = parts[-1].rstrip("\n")
    return parts


def md(src):
    return {"cell_type": "markdown", "id": f"md-{next(_n):03d}", "metadata": {}, "source": _lines(src)}


def code(src):
    return {"cell_type": "code", "id": f"code-{next(_n):03d}", "execution_count": None,
            "metadata": {}, "outputs": [], "source": _lines(src)}


C = []

C.append(md(r"""
# Quantum Walk Physics Interface

Standardized `quantum_walk_1d` / `quantum_walk_2d` functions built to the
physics-team handoff contract in
[`docs/physics-handoff-spec.pdf`](docs/physics-handoff-spec.pdf) ("Requirements
& Integration for Team Quantum Walk" / *Physics Implementation Handoff –
Acceptance Specification*).

**Purpose.** [`quantum_walk_features.ipynb`](quantum_walk_features.ipynb) is
the ML pipeline and evaluation notebook. This notebook is the companion
physics module the handoff spec asks for: a narrow, stable interface —
`params` in, `{positions, probabilities, ...}` out — that an ML pipeline (this
project's or another team's) can call without knowing anything about coins,
shifts, or lattices.

**Self-contained**, like every notebook in this project: the walk physics is
copied in below, not imported from a package. It is byte-identical to the walk
in `quantum_walk_features.ipynb` — same author, same code, unmodified.

## What this notebook delivers, and what it doesn't yet

| Function | Status |
|---|---|
| `quantum_walk_1d(params, T)` | **Implemented and validated** |
| `quantum_walk_2d(params, T, use_mu_chi=True)` | **Stub.** Raises `NotImplementedError`. 2D work has not started yet in this project (see `docs/COMPLIANCE.md`) — this stub exists so the interface is already the right shape for when it does, per the handoff spec's own integration pattern (§11, `_is_physics_stub`). |

Optional outputs `position_amplitudes` and `state` are **not provided**. The
1D walk this project uses (`run_qw_1d` below) discards coin-resolved
amplitudes before returning — it sums to real probabilities inside the
function itself, by design, and is intentionally left unmodified. Since both
outputs are marked optional in the spec (§4), this is a valid partial
implementation, not a violation — documented here rather than silently
omitted.
"""))

C.append(md("## 1. The walk\n\nCopied unmodified from `quantum_walk_features.ipynb`."))

C.append(code('''
import numpy as np


def move(space):
    """Move the walker one step: coin |0> keeps its index, coin |1> advances by two."""
    new_space = np.zeros([len(space) + 2, 2], dtype=complex)

    for i in range(len(space)):
        new_space[i][0] = space[i][0]
        new_space[i + 2][1] = space[i][1]

    return new_space


def create_coin_operator(alpha, beta):
    """Create the 2x2 coin operator C(alpha, beta)."""
    return np.array([
        [np.cos(alpha) * np.exp(1j * beta), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha) * np.exp(-1j * beta)]
    ], dtype=complex)


def run_qw_1d(theta, phi, alpha, beta, num_steps):
    """Run a single 1D quantum walk."""
    coin = create_coin_operator(alpha, beta)

    # Initial state:
    # |psi_0> = |x=0> ⊗
    # [cos(theta/2), exp(i*phi) sin(theta/2)]
    state = np.array([
        [np.cos(theta / 2),
         np.exp(1j * phi) * np.sin(theta / 2)]
    ], dtype=complex)

    for _ in range(num_steps):
        # Apply coin operator to every position
        for i in range(len(state)):
            state[i] = coin @ state[i]

        # Apply conditional movement
        state = move(state)

    # Probability for each coin state
    prob = np.abs(state) ** 2

    # Sum over coin states
    probabilities = prob.sum(axis=1)

    return probabilities
'''))

C.append(md(r"""
## 2. `quantum_walk_1d` — the required interface

Wraps `run_qw_1d` to the exact contract in the handoff spec §2.1–§3:

```
def quantum_walk_1d(params, T) -> {
    "positions": np.ndarray,       # shape (N,), float
    "probabilities": np.ndarray,   # shape (N,), float, sums to 1
}
```

`params` is a dict with keys `theta1, phi1, alpha1, beta1`. Declared ranges
(from the handoff spec, §2.1):

| parameter | range |
|---|---|
| `theta1` | $[0, \pi]$ |
| `phi1` | $[0, 2\pi]$ |
| `alpha1` | $[0, \pi]$ |
| `beta1` | $[0, 2\pi]$ |

`positions` is `2T+1` sites, centred on 0 — `np.arange(-T, T+1, dtype=float)` —
matching `run_qw_1d`'s own lattice exactly (it starts at one site and grows by
two per step) and the shape example in spec §3.1.
"""))

C.append(code('''
PARAMS_1D = ("theta1", "phi1", "alpha1", "beta1")
RANGES_1D = {
    "theta1": (0.0, np.pi),
    "phi1": (0.0, 2.0 * np.pi),
    "alpha1": (0.0, np.pi),
    "beta1": (0.0, 2.0 * np.pi),
}


def _check_params(params, required, ranges, atol=1e-9):
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"params is missing required keys: {missing}")
    out_of_range = []
    for k in required:
        lo, hi = ranges[k]
        v = float(params[k])
        if not (lo - atol <= v <= hi + atol):
            out_of_range.append(f"{k}={v:.6g} not in [{lo:.6g}, {hi:.6g}]")
    if out_of_range:
        raise ValueError("params outside declared ranges: " + "; ".join(out_of_range))


def quantum_walk_1d(params, T):
    """Run a 1D quantum walk to the physics handoff interface contract.

    Args:
        params: dict with keys theta1, phi1, alpha1, beta1 (see RANGES_1D).
        T: number of walk steps (non-negative int).

    Returns:
        {"positions": (2T+1,) float array, "probabilities": (2T+1,) float array}.
        Optional position_amplitudes / state are not provided -- see notebook
        header for why.
    """
    _check_params(params, PARAMS_1D, RANGES_1D)
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")

    probabilities = run_qw_1d(
        params["theta1"], params["phi1"], params["alpha1"], params["beta1"], T
    )
    positions = np.arange(-T, T + 1, dtype=float)

    return {"positions": positions, "probabilities": probabilities}


quantum_walk_1d._is_physics_stub = False
'''))

C.append(md(r"""
## 3. `quantum_walk_2d` — stub

2D has not been built in this project yet (`docs/COMPLIANCE.md` tracks it as
the largest open item). Per the handoff spec's own integration pattern (§11),
this is a **stub**: right shape, wrong body. It raises `NotImplementedError`
when called, and carries `_is_physics_stub = True` so an integration harness
can check readiness with `assert not quantum_walk_2d._is_physics_stub` rather
than by calling it and catching an exception.

Signature and contract, per spec §2.2 / §4 / §7:

```
def quantum_walk_2d(params, T, use_mu_chi=True) -> {
    "positions": np.ndarray,       # shape (N, 2), float, columns [x, y]
    "probabilities": np.ndarray,   # shape (N,), float, sums to 1
}
```

`params` carries all 10: `theta1, phi1, alpha1, beta1, theta2, phi2, alpha2,
beta2, mu, chi`. `use_mu_chi=False` must still accept all 10 keys but force
`mu = chi = 0.0` internally (a separable initial state) — this is the
`2D without (µ,χ) vs 2D with (µ,χ)` ablation from the original challenge,
pinned to a function flag rather than left as a free-form experiment.
"""))

C.append(code('''
PARAMS_2D = ("theta1", "phi1", "alpha1", "beta1",
            "theta2", "phi2", "alpha2", "beta2", "mu", "chi")
RANGES_2D = {
    "theta1": (0.0, np.pi), "phi1": (0.0, 2.0 * np.pi),
    "alpha1": (0.0, np.pi), "beta1": (0.0, 2.0 * np.pi),
    "theta2": (0.0, np.pi), "phi2": (0.0, 2.0 * np.pi),
    "alpha2": (0.0, np.pi), "beta2": (0.0, 2.0 * np.pi),
    "mu": (0.0, np.pi), "chi": (0.0, 2.0 * np.pi),
}


def quantum_walk_2d(params, T, use_mu_chi=True):
    """2D quantum walk -- NOT YET IMPLEMENTED.

    Matches the handoff spec's interface (params, T, use_mu_chi) -> dict, so
    callers can already write code against it; calling it raises until the
    physics is built. See docs/COMPLIANCE.md for the open questions blocking
    this (the 2D shift operator convention is not defined by either PDF).
    """
    _check_params(params, PARAMS_2D, RANGES_2D)
    raise NotImplementedError(
        "quantum_walk_2d is a stub -- 2D quantum walk physics is not yet "
        "implemented in this project. See docs/COMPLIANCE.md."
    )


quantum_walk_2d._is_physics_stub = True
'''))

C.append(md("## 4. Validation\n\nThe checks the handoff spec requires (§10), for the parts that are built."))

C.append(code(r'''
def _random_1d_params(rng):
    return {
        "theta1": rng.uniform(0, np.pi),
        "phi1": rng.uniform(0, 2 * np.pi),
        "alpha1": rng.uniform(0, np.pi),
        "beta1": rng.uniform(0, 2 * np.pi),
    }


def run_validation(verbose=True):
    """Run every §10 check that quantum_walk_1d's current scope supports."""
    results = []
    rng = np.random.default_rng(0)

    # 1. normalization: |sum(p) - 1| <= 1e-8 and p_i >= -1e-12 (spec's own tolerances)
    worst_sum, worst_neg = 0.0, 0.0
    for T in [0, 1, 2, 5, 15, 30, 50]:
        for _ in range(6):
            out = quantum_walk_1d(_random_1d_params(rng), T)
            p = out["probabilities"]
            worst_sum = max(worst_sum, abs(p.sum() - 1.0))
            worst_neg = min(worst_neg, p.min())
    ok = worst_sum <= 1e-8 and worst_neg >= -1e-12
    results.append(("normalization: |Σp-1|<=1e-8, p_i>=-1e-12", ok,
                    f"max |Σp-1|={worst_sum:.2e}, min p_i={worst_neg:.2e}"))

    # 2. shape contract: positions/probabilities both (2T+1,), positions matches spec example
    ok = True
    for T in [0, 1, 5, 20]:
        out = quantum_walk_1d(_random_1d_params(rng), T)
        ok &= out["positions"].shape == (2 * T + 1,) == out["probabilities"].shape
        ok &= out["positions"].dtype == np.float64
    ok &= np.array_equal(quantum_walk_1d(_random_1d_params(rng), 2)["positions"],
                         np.array([-2., -1., 0., 1., 2.]))
    results.append(("output shapes match spec (N,) / (N,)", ok,
                    "positions.shape == probabilities.shape == (2T+1,)"))

    # 3. determinism: identical inputs -> byte-identical outputs, repeated calls
    p = _random_1d_params(rng)
    out1 = quantum_walk_1d(dict(p), 25)
    out2 = quantum_walk_1d(dict(p), 25)
    out3 = quantum_walk_1d(dict(p), 25)
    ok = (np.array_equal(out1["probabilities"], out2["probabilities"])
          and np.array_equal(out2["probabilities"], out3["probabilities"]))
    results.append(("deterministic across repeat calls", ok,
                    "same params, T -> byte-identical output, 3x"))

    # 4. T=1 edge case
    out = quantum_walk_1d(_random_1d_params(rng), 1)
    ok = (out["positions"].shape == (3,) and np.isclose(out["probabilities"].sum(), 1.0, atol=1e-8))
    results.append(("T=1 edge case", ok, f"shape={out['positions'].shape}, sum(p)={out['probabilities'].sum():.6f}"))

    # 5. phase sensitivity: phi must change the output when the coin actually
    # mixes (alpha != 0); with alpha=0 the coin is diagonal and phi is provably
    # invisible in |amplitude|^2, so that case is not expected to differ.
    base = dict(theta1=np.pi / 3, phi1=0.1, alpha1=np.pi / 4, beta1=0.3)
    varied = dict(base, phi1=2.5)
    p1 = quantum_walk_1d(base, 30)["probabilities"]
    p2 = quantum_walk_1d(varied, 30)["probabilities"]
    ok = np.abs(p1 - p2).max() > 1e-6
    results.append(("phase sensitivity (alpha1 != 0)", ok,
                    f"max |Δp| over phi1 = {np.abs(p1 - p2).max():.4f}"))

    # 6. symmetry: theta1=pi/2, phi1=pi/2, alpha1=pi/4, beta1=0 gives a
    # left-right symmetric distribution (verified numerically, not assumed).
    sym = dict(theta1=np.pi / 2, phi1=np.pi / 2, alpha1=np.pi / 4, beta1=0.0)
    p = quantum_walk_1d(sym, 40)["probabilities"]
    ok = np.abs(p - p[::-1]).max() < 1e-9
    results.append(("symmetry (theta1=phi1=pi/2, alpha1=pi/4, beta1=0)", ok,
                    f"max |p(x)-p(-x)| = {np.abs(p - p[::-1]).max():.2e}"))

    # 7-9. 2D-dependent checks: use_mu_chi ablation, amplitude-to-probability
    # consistency, full-state-to-position-probability consistency. All three
    # need quantum_walk_2d and/or position_amplitudes/state, neither of which
    # this scope provides yet -- recorded as pending, not silently skipped.
    for name in ("use_mu_chi=True vs False ablation (needs 2D)",
                "amplitude-to-probability consistency (position_amplitudes not provided)",
                "full-state-to-position-probability consistency (state not provided)"):
        results.append((name, None, "PENDING -- blocked on 2D / optional outputs, see notebook header"))

    if verbose:
        for name, passed, detail in results:
            if passed is None:
                tag = "PEND"
            else:
                tag = "PASS" if passed else "FAIL"
            print(f"  [{tag}]  {name:<52} {detail}")
    hard_results = [r for r in results if r[1] is not None]
    return all(r[1] for r in hard_results)


print("VALIDATION (handoff spec §10)")
assert run_validation(), "validation failed -- stop here"
print("\\nall runnable checks passed (3 pending on 2D / optional outputs, see above)")
'''))

C.append(md("## 5. Integration flags\n\nPer handoff spec §11."))

C.append(code('''
# Gates whether a consuming notebook actually runs the (potentially expensive)
# QW experiment cells, or skips them -- e.g. during import-time smoke tests.
RUN_QW_EXPERIMENTS = True

print(f"quantum_walk_1d._is_physics_stub = {quantum_walk_1d._is_physics_stub}")
print(f"quantum_walk_2d._is_physics_stub = {quantum_walk_2d._is_physics_stub}")
print(f"RUN_QW_EXPERIMENTS = {RUN_QW_EXPERIMENTS}")
'''))

C.append(md("## 6. Example usage"))

C.append(code('''
example = quantum_walk_1d({"theta1": np.pi / 3, "phi1": 1.1, "alpha1": np.pi / 4, "beta1": 0.7}, T=15)
print("positions:   ", example["positions"][:5], "...", example["positions"][-5:])
print("probabilities:", example["probabilities"][:5], "...")
print("sum(p) =", example["probabilities"].sum())

print()
try:
    quantum_walk_2d({k: 0.0 for k in PARAMS_2D}, T=10)
except NotImplementedError as e:
    print(f"quantum_walk_2d raises as expected: {e}")
'''))

C.append(md(r"""
## 7. Delivery package summary

Per handoff spec §12, the minimum delivery package.

| Item | Status |
|---|---|
| `quantum_walk_1d(params, T)` | Delivered, validated (§4 above) |
| `quantum_walk_2d(params, T, use_mu_chi)` | Stub only -- `NotImplementedError`, `_is_physics_stub = True` |
| Dependencies | `numpy` only (version printed below) |
| Physics spec reference | `docs/PQO-2026-06-challenge.pdf` (original challenge), `docs/physics-handoff-spec.pdf` (this interface contract) |
| Basis | Position basis, one coin qubit (1D) |
| Normalization | $\|\sum_x P_T(x) - 1\| \le 10^{-8}$, $P_T(x) \ge -10^{-12}$ -- verified, holds to $\sim 10^{-14}$ in practice |
| Expected output | `{"positions": (2T+1,) float, "probabilities": (2T+1,) float}`; `position_amplitudes` / `state` intentionally omitted (optional; see header) |
| Performance | $O(T^2)$ per call (nested loop over a lattice that grows with $T$) -- benchmarked in `quantum_walk_features.ipynb`; no caching/batching strategy implemented yet |
| Determinism | Verified -- pure function of `(params, T)`, no internal randomness |

## 8. Open decisions for the physics team meeting

Per handoff spec §13, carried over from `docs/COMPLIANCE.md`:

| Decision | Status |
|---|---|
| `alpha1`/`alpha2` range | **Resolved** -- $[0,\pi]$, per this spec, superseding this project's earlier self-chosen $[0,\pi/2]$ |
| 2D shift operator (`C = C_1 \otimes C_2` → lattice moves) | **Proposed, not confirmed**: coin 1 drives x, coin 2 drives y (`docs/COMPLIANCE.md`). Every step is then diagonal, occupying one parity sublattice -- needs physics-team sign-off before 2D is built |
| `mu`/`chi` role when `use_mu_chi=False` | Spec fixes this (`mu=chi=0`, separable state) -- already encoded in the `quantum_walk_2d` stub's docstring, will carry into the real implementation |
| `position_amplitudes` / `state` | Omitted for 1D -- `run_qw_1d` discards them by design and is kept unmodified. Would need a separate, amplitude-preserving implementation to provide these; not built |
| Performance strategy (cache/batch/reuse across `T`) | None implemented. $O(T^2)$ per call is measured, not yet optimized |
| `parameter_entanglement_scores` (spec §8) | Not computed -- needs a 2D reduced-density-matrix / bipartite-entanglement measure, which needs 2D first |
"""))

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.13"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path("quantum_walk_physics_interface.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} ({len(C)} cells, {sum(1 for c in C if c['cell_type'] == 'code')} code)")
