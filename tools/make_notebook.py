"""Generate qwalk_demo.ipynb. Regenerate with: python tools/make_notebook.py"""
import json
import pathlib

def lines(src):
    """ipynb `source` is a list of lines that keep their trailing newlines."""
    parts = [line + "\n" for line in src.strip().split("\n")]
    parts[-1] = parts[-1].rstrip("\n")
    return parts

_counter = iter(range(1000))

def _cell_id(kind):
    return f"{kind}-{next(_counter):03d}"

def md(src):
    return {"cell_type": "markdown", "id": _cell_id("md"), "metadata": {},
            "source": lines(src)}

def code(src):
    return {"cell_type": "code", "id": _cell_id("code"), "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines(src)}

cells = []

cells.append(md("""
# Quantum walk feature expansion

A discrete-time quantum walk used as a feature-expansion mechanism for binary
classification on Breast Cancer Wisconsin, compared against classical baselines.

Pure NumPy state-vector simulation — no Qiskit, no Pennylane. One independent
walk per dataset row.

```
features -> select k -> scale to parameter ranges -> map to walk parameters
         -> build |psi_0> -> run T steps -> extract features -> ML model
```

This notebook drives the `qw` package rather than restating it, so the physics
and the leakage guarantees are the same ones the test suite checks.

## Fixed physics

Taken from the challenge spec and not modified:

$$C(a,b) = \\begin{pmatrix} \\cos a\\, e^{ib} & \\sin a \\\\ -\\sin a & \\cos a\\, e^{-ib} \\end{pmatrix}$$

$$|\\text{coin}_0\\rangle = \\cos(\\theta/2)|0\\rangle + e^{i\\phi}\\sin(\\theta/2)|1\\rangle$$

$$|\\psi_{t+1}\\rangle = S\\,C\\,|\\psi_t\\rangle, \\qquad P_T(x) = \\sum_c |\\langle x,c|\\psi_T\\rangle|^2$$

Coin $|0\\rangle$ moves left, coin $|1\\rangle$ moves right. Ranges:
$\\theta\\in[0,\\pi]$, $\\phi\\in[0,2\\pi]$, $\\alpha\\in[0,\\pi]$, $\\beta\\in[0,2\\pi]$.
"""))

cells.append(md("## 1. Setup"))
cells.append(code("""
import numpy as np
import matplotlib.pyplot as plt

from qw.config import ExperimentConfig
from qw.data import load_breast_cancer_split
from qw.mapping import QuantumWalkFeatures
from qw.spaces import SPACES, build_space, materialize
from qw.evaluate import evaluate_space, format_results_table
from qw.walk1d import (
    FEATURE_NAMES,
    extract_features,
    position_axis,
    position_probabilities,
    run_qw_1d,
)

plt.rcParams.update({"figure.figsize": (9, 4), "figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})
"""))

cells.append(md("""
## 2. Configuration

Every knob lives in one frozen dataclass. Nothing below reads a literal seed,
step count or parameter count.
"""))
cells.append(code("""
cfg = ExperimentConfig(n_steps=30, permutation=(0, 1, 2, 3), selector="mutual_info")
print(cfg.summary())
print()
print("walk parameters:", cfg.spec.param_names)
print("ranges:         ", [(round(lo, 3), round(hi, 3)) for lo, hi in cfg.spec.param_ranges])
print("features out:   ", cfg.spec.n_features_out)
"""))

cells.append(md("""
## 3. Data

One stratified split, seeded, shared by all four feature spaces. The test arrays
below are never handed to any `fit`.
"""))
cells.append(code("""
data = load_breast_cancer_split(cfg)
print(data.summary())
print("classes:", data.target_names)
"""))

cells.append(md("""
## 4. Feature selection and parameter mapping

`QuantumWalkFeatures` owns the selector and the `MinMaxScaler` and fits both in
`fit()`. Test rows only ever reach `transform()`.
"""))
cells.append(code("""
qw = QuantumWalkFeatures(spec=cfg.spec, n_steps=cfg.n_steps,
                         permutation=cfg.permutation, selector=cfg.selector, seed=cfg.seed)
qw.fit(data.X_train, data.y_train)

print(qw.describe())
print()
for name, idx in zip(cfg.spec.param_names, [qw.indices_[s] for s in qw.permutation_]):
    print(f"  {name:<6} <- x{idx:<3} {data.feature_names[idx]}")
"""))

cells.append(code("""
params_train = qw.walk_parameters(data.X_train)
params_test = qw.walk_parameters(data.X_test)

print("walk parameters per row (train):", params_train.shape)
for i, (name, (lo, hi)) in enumerate(zip(cfg.spec.param_names, cfg.spec.param_ranges)):
    col = params_train[:, i]
    print(f"  {name:<6} range [{lo:.3f}, {hi:.3f}]  observed [{col.min():.3f}, {col.max():.3f}]")

# clip=True keeps test rows inside the declared ranges without refitting on them
assert all(params_test[:, i].min() >= lo - 1e-12 and params_test[:, i].max() <= hi + 1e-12
           for i, (lo, hi) in enumerate(cfg.spec.param_ranges))
print("\\ntest rows stay inside every declared range")
"""))

cells.append(md("""
## 5. The walk

Two rows of the dataset, one per class, become two different walks. The final
position distribution $P_T(x)$ is the readout.
"""))
cells.append(code("""
psi_train = cfg.spec.run(params_train, cfg.n_steps)
prob = position_probabilities(psi_train)
x = position_axis(prob.shape[1])

print("psi shape:", psi_train.shape, "(n_samples, n_positions, 2)")
print("norm deviation (max):", np.abs(prob.sum(axis=1) - 1.0).max())

benign = np.where(data.y_train == 1)[0][0]
malignant = np.where(data.y_train == 0)[0][0]

fig, ax = plt.subplots()
for idx, label, style in [(benign, data.target_names[1], "-"), (malignant, data.target_names[0], "--")]:
    ax.plot(x, prob[idx], style, lw=1.4, label=f"{label} (row {idx})")
ax.set_xlabel("position $x$"); ax.set_ylabel("$P_T(x)$")
ax.set_title(f"Final position distribution, T = {cfg.n_steps}")
ax.legend(); plt.show()
"""))

cells.append(code("""
# Class-averaged distributions: what the classifier actually has to separate.
fig, ax = plt.subplots()
for cls, name in enumerate(data.target_names):
    ax.plot(x, prob[data.y_train == cls].mean(axis=0), lw=1.6, label=f"mean, {name}")
ax.set_xlabel("position $x$"); ax.set_ylabel(r"$\\langle P_T(x) \\rangle$")
ax.set_title("Class-averaged position distributions")
ax.legend(); plt.show()
"""))

cells.append(md("""
### Sanity check: the Hadamard walk

$\\alpha=\\pi/4$, $\\beta=0$, $\\theta=0$, $\\phi=0$ is the Hadamard-like walk from
$|0\\rangle$. It is ballistic and left-asymmetric, peaking near $-T/\\sqrt{2}$.
Parity restricts support to even sites at even $T$, so the discrete peak lands
at $x=-20$ for $T=30$.
"""))
cells.append(code("""
T = 30
zero = np.array([0.0])
psi_h = run_qw_1d(zero, zero, np.array([np.pi / 4]), zero, T)
p_h = position_probabilities(psi_h)[0]
x_h = position_axis(p_h.shape[0])
peak = x_h[np.argmax(p_h)]

print(f"peak at x = {peak:.0f}   target -0.7*T = {-0.7 * T:.1f}   -T/sqrt(2) = {-T / np.sqrt(2):.2f}")
print(f"norm = {p_h.sum():.12f}   edges = {p_h[0]}, {p_h[-1]}  (no wraparound)")

fig, ax = plt.subplots()
ax.plot(x_h, p_h, lw=1.2)
ax.axvline(peak, color="crimson", ls="--", lw=1, label=f"peak x={peak:.0f}")
ax.axvline(-T / np.sqrt(2), color="gray", ls=":", lw=1, label=r"$-T/\\sqrt{2}$")
ax.set_xlabel("position $x$"); ax.set_ylabel("$P_T(x)$")
ax.set_title(f"Hadamard walk, T = {T}")
ax.legend(); plt.show()
"""))

cells.append(md("""
## 6. Feature extraction

Each distribution collapses to 10 numbers.
"""))
cells.append(code("""
feats = extract_features(psi_train)
print("extracted:", feats.shape)
print()
header = f"{'feature':<10}{'mean(benign)':>14}{'mean(malignant)':>17}"
print(header); print("-" * len(header))
for j, name in enumerate(FEATURE_NAMES):
    print(f"{name:<10}{feats[data.y_train == 1, j].mean():>14.4f}"
          f"{feats[data.y_train == 0, j].mean():>17.4f}")
"""))

cells.append(md("""
## 7. The four feature spaces

Identical model, split, seed and metrics throughout. The only thing that varies
is the feature matrix.
"""))
cells.append(code("""
for space in SPACES:
    Z_tr, Z_te = materialize(space, cfg, data.X_train, data.y_train, data.X_test)
    print(f"{space:<14} train {str(Z_tr.shape):<14} test {Z_te.shape}")
"""))

cells.append(code("""
results = [evaluate_space(space, data, cfg) for space in SPACES]
print(format_results_table(results))
"""))

cells.append(code("""
fig, ax = plt.subplots(figsize=(9, 3.6))
metrics = ["accuracy", "f1", "precision", "recall"]
width, pos = 0.2, np.arange(len(SPACES))
for i, m in enumerate(metrics):
    ax.bar(pos + i * width, [getattr(r.metrics, m) for r in results], width, label=m)
ax.set_xticks(pos + 1.5 * width); ax.set_xticklabels([r.space for r in results])
ax.set_ylim(0.90, 1.0); ax.set_ylabel("score")
ax.set_title(f"Test-set performance, T = {cfg.n_steps}, perm = {list(cfg.permutation)}")
ax.legend(ncol=4, fontsize=9); plt.show()
"""))

cells.append(md("""
## 8. Permutation search

Which feature drives which parameter is a free choice: $4! = 24$ of them. The
search runs by cross-validation **on the training set only**; the held-out score
is read once, for the winner.

Choosing the permutation by test score instead would inflate the result — on
this split, the best test score over the same 24 is 0.9930 against the 0.9860
the honest procedure gives. That gap is selection leakage, not signal.
"""))
cells.append(code("""
from itertools import permutations
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from qw.evaluate import make_model

search = GridSearchCV(
    Pipeline([("space", build_space("original+qw", cfg)), ("model", make_model(cfg.model, cfg.seed))]),
    param_grid={"space__qw__permutation": list(permutations(range(cfg.spec.n_params)))},
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed),
    n_jobs=-1,
)
search.fit(data.X_train, data.y_train)

best = tuple(search.best_params_["space__qw__permutation"])
print(f"best permutation: {list(best)}  [{cfg.spec.describe_permutation(best)}]")
print(f"cv accuracy:      {search.best_score_:.4f}  (train only)")
"""))

cells.append(code("""
best_cfg = ExperimentConfig(n_steps=cfg.n_steps, permutation=best, selector=cfg.selector)
print(format_results_table([evaluate_space(s, data, best_cfg) for s in SPACES]))
"""))

cells.append(md("""
## 9. Effect of T

The step count sets how far the walk spreads and how much structure the ten
summary features can carry.
"""))
cells.append(code("""
steps_grid = [5, 10, 20, 30, 40, 60]
curve = {"qw": [], "original+qw": []}
for T_i in steps_grid:
    cfg_i = ExperimentConfig(n_steps=T_i, permutation=cfg.permutation, selector=cfg.selector)
    for space in curve:
        curve[space].append(evaluate_space(space, data, cfg_i).metrics.accuracy)

baseline = evaluate_space("original", data, cfg).metrics.accuracy
fig, ax = plt.subplots()
for space, ys in curve.items():
    ax.plot(steps_grid, ys, "o-", lw=1.4, label=space)
ax.axhline(baseline, color="gray", ls="--", lw=1, label="original (baseline)")
ax.set_xlabel("walk steps $T$"); ax.set_ylabel("test accuracy")
ax.set_title("Accuracy vs walk length"); ax.legend(); plt.show()
"""))

cells.append(md("""
## Notes

- The quantum-walk space alone (10 dimensions) trails the raw 30 features, and
  concatenating the two ties the baseline rather than beating it. The
  permutation search does not change that once it is scored honestly.
- Remaining knobs for the 1D case: $T$, the selector, and the classifier.
- **2D is not implemented.** Adding it means writing `qw/walk2d.py` with 10
  parameters, two coins and the $\\mu/\\chi$ entangled initial state, then one
  line in `qw/registry.py`. Selection, mapping, spaces, evaluation and this
  notebook all read `spec.n_params` and `spec.param_ranges`, so nothing here
  needs rewriting.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path("qwalk_demo.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} ({len(cells)} cells)")
