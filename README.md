# qwalk — quantum walk feature expansion

A discrete-time quantum walk used as a feature expansion for binary
classification, compared against classical baselines on Breast Cancer
Wisconsin. Pure NumPy state-vector simulation — no Qiskit, no Pennylane.

```
features -> select k -> scale to parameter ranges -> map to walk parameters
         -> build |psi_0> -> run T steps -> extract features -> ML model
```

One independent quantum walk per dataset row. **1D only** (4 parameters); the
2D case is not implemented yet, but the seam for it is in place.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python run_experiment.py --steps 30 --permutation 0,1,2,3
```

```
space           dims   accuracy       f1  precision   recall
------------------------------------------------------------
original          30     0.9860   0.9889     0.9889   0.9889 *
polynomial       495     0.9790   0.9836     0.9677   1.0000
qw                10     0.9510   0.9609     0.9663   0.9556
original+qw       40     0.9860   0.9889     0.9889   0.9889 *
```

Flags: `--steps` (T), `--permutation`, `--walk`, `--selector`
(`mutual_info` | `anova_f` | `tree`), `--model`, `--seed`, `--test-size`,
`--poly-degree`. All default to `ExperimentConfig`.

## Permutation search

`sweep_permutations.py` searches all 4! = 24 permutations by cross-validation
on the **training set only**, then reports the held-out score for the winner
once. Takes a few seconds.

```bash
.venv/bin/python sweep_permutations.py --space original+qw --folds 5
```

Do not pick the permutation by test-set score. On this split the CV-chosen
permutation scores 0.9860 on test, while the best test-set score over the same
24 is 0.9930 — that 0.7-point gap is selection leakage, not signal.

## Standalone notebook (the submission deliverable)

`quantum_walk_features.ipynb` is **fully self-contained** — it imports nothing
from `qw/` and runs on its own with only numpy, pandas, scikit-learn and
matplotlib. This is the file to hand in.

```bash
.venv/bin/jupyter lab quantum_walk_features.ipynb
```

Put `diabetes.csv`, `data.csv` or `spambase.csv` next to it and set `DATASET` in
the config cell; with no CSV present it falls back to scikit-learn's built-in
Breast Cancer Wisconsin so it always runs. Executes end to end in about 15
seconds and ships with outputs already embedded.

It contains its own physics validation (section 4) standing in for the pytest
suite: norm preservation, the Hadamard peak, no wraparound, the initial state,
and coin unitarity. Regenerate it with `python tools/make_standalone_notebook.py`
(this clears outputs, so re-execute afterwards).

## Package demo notebook


`qwalk_demo.ipynb` walks the whole pipeline with plots — position distributions
per class, the Hadamard sanity check, the results table, and accuracy vs `T`.
It imports the `qw` package rather than restating it, so it cannot drift from
the tested code.

```bash
.venv/bin/jupyter lab qwalk_demo.ipynb
```

It ships already executed, so it reads correctly on GitHub without running
anything. To re-run it in place after changing the code:

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace qwalk_demo.ipynb
```

The notebook is generated, not hand-edited. Cell sources live in
`tools/make_notebook.py`; regenerate with `python tools/make_notebook.py`
(this clears outputs, so re-execute afterwards).

## Tests

```bash
.venv/bin/python -m pytest -q
```

`tests/test_walk1d.py` covers the physics: norm preservation to 1e-10 across
several T, the Hadamard check (`alpha=pi/4, beta=0, theta=0, phi=0` at T=30
peaks at x=-20, within tolerance of the ballistic `-T/sqrt(2) ~ -21.2`), no
wraparound at the lattice edges, the initial state, and coin unitarity.
`tests/test_pipeline.py` covers leakage, reproducibility and space widths.

## Fixed physics

Taken from the challenge spec and not to be "improved":

```
C(a, b) = [[cos(a)*exp(i*b),  sin(a)          ],
           [-sin(a),          cos(a)*exp(-i*b)]]

|coin_0>    = cos(theta/2)|0> + exp(i*phi)*sin(theta/2)|1>
|psi_{t+1}> = S C |psi_t>            (coin first, then shift)
S           : coin |0> moves left, coin |1> moves right
P_T(x)      = sum_c |<x, c | psi_T>|^2
```

Parameter ranges: `theta in [0, pi]`, `phi in [0, 2pi]`, `alpha in [0, pi/2]`,
`beta in [0, 2pi]`.

The lattice holds `2T+3` sites, two more than the `2T+1` light cone, so the
periodic shift can never wrap probability around the array. Position `x=0` is
the centre index.

## Layout

| file | role |
| --- | --- |
| `qw/config.py` | `ExperimentConfig` — seed, T, selector, model, permutation, degree. The only place these live. |
| `qw/walkspec.py` | `WalkSpec` — the 1D/2D seam: parameter names, ranges, `run`, `extract`. |
| `qw/walk1d.py` | `run_walk_1d` and `extract_features`, exported as `WALK_1D`. |
| `qw/registry.py` | `WALKS` — where a new variant is wired in. |
| `qw/data.py` | Breast Cancer Wisconsin, one stratified split. |
| `qw/selection.py` | Pluggable selectors returning exactly `k` indices. |
| `qw/mapping.py` | `QuantumWalkFeatures` — select, scale, permute, walk, extract. |
| `qw/spaces.py` | The four feature spaces as transformers. |
| `qw/evaluate.py` | One `evaluate_space` used for all four. |

### No test-set leakage

`QuantumWalkFeatures` is a scikit-learn transformer that owns the selector and
the `MinMaxScaler`, and fits both in its `fit`. Every space goes inside a
`Pipeline` with the classifier, so `X_test` only ever reaches `transform` and
`predict` — leakage is structurally impossible, not a convention. The scaler
uses `clip=True` so test rows outside the training range still land inside the
declared parameter intervals without refitting.

`tests/test_pipeline.py::test_selector_and_scaler_never_see_test_rows` asserts
this by spying on every `MinMaxScaler.fit` call.

### The permutation

`permutation[i]` is the index — among the selected features in ascending order
— of the feature feeding parameter `i`, with parameters ordered
`(theta, phi, alpha, beta)`. It is an explicit argument everywhere and is
logged on every run:

```
walk: selector=mutual_info selected=[7, 20, 22, 23] perm=[0, 1, 2, 3] [theta<-x7 phi<-x20 alpha<-x22 beta<-x23] T=30
```

Because it is a constructor argument of a scikit-learn estimator, the 4! = 24
search later is a plain grid search with no new plumbing:

```python
from itertools import permutations
GridSearchCV(pipeline, {"space__qw__permutation": list(permutations(range(4)))})
```

### Adding the 2D case

Write `qw/walk2d.py` exporting a `WalkSpec` with 10 parameter names, their
ranges, two coins and the mu/chi entangled initial state, then add one line to
`qw/registry.py`. Selection, mapping, spaces, evaluation and the CLI all read
`spec.n_params` and `spec.param_ranges` and need no changes.
