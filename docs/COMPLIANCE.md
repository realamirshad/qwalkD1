# Compliance against PQO–2026–06

Where the current submission stands against the rubric in
[`CHALLENGE.md`](CHALLENGE.md). Assessed against
[`quantum_walk_features.ipynb`](../quantum_walk_features.ipynb), the standalone
deliverable.

## Score estimate

| § | Rubric section | Points | Status | Est. |
|---|---|---:|---|---:|
| 23.1 | Correct 1D walk + feature-space generation | 20 | complete, physics verified in-notebook | **20** |
| 23.2 | Correct generalisation to 2D | 20 | **not implemented** | **0** |
| 23.3 | Rich feature extraction (≥5 across categories) | 15 | category A strong, B thin, C absent | **~7** |
| 23.4 | Feature selection + correct mapping | 10 | complete, permutation search included | **10** |
| 23.5 | Four feature spaces, fair comparison | 10 | complete, model/split/seed held fixed | **10** |
| 23.6 | Systematic study of `T` | 10 | complete, CV-selected `T*` plus curve | **10** |
| 23.7 | 1D/2D effect and `µ, χ` | 10 | blocked on 2D | **0** |
| 23.8 | Analysis, report, presentation | 5 | notebook good; no written report yet | **~3** |
| | **Total** | **100** | | **~60** |

**The 2D walk is worth 20 points directly and gates another 10.** It is the
single largest gap by a wide margin.

## Acceptance criteria (§20)

| # | Criterion | Status |
|---|---|---|
| 1 | 1D walk correct | pass |
| 2 | 2D walk correct | **fail — not implemented** |
| 3 | Coin matches definition | pass |
| 4 | Initial state matches definition | pass |
| 5 | 2D state with `µ, χ` | **fail — not implemented** |
| 6 | `Σ_x P_T(x) ≈ 1` | pass — asserted to 1e-10, max deviation 9.3e-15 |
| 7 | Feature selection correct | pass |
| 8 | #features == #parameters | pass — enforced by `N_PARAMS` |
| 9 | Mapping explicit | pass — `describe_permutation`, logged every run |
| 10 | Four spaces evaluated | pass (1D only) |
| 11 | Several values of `T` | pass — 7 values |
| 12 | 1D vs 2D comparison | **fail** |
| 13 | Effect of `µ, χ` | **fail** |
| 14 | Conditions identical across methods | pass |
| 15 | Model fixed across comparisons | pass |
| 16 | **No test-set info in selection or parameter choice** | pass — structurally enforced |
| 17 | Reproducible | pass — single seed |

Criterion 16 is worth dwelling on: the permutation and `T` are chosen by
cross-validation on train only, and the notebook *measures* what tuning on test
would have added (+0.0260 on diabetes). Most submissions get this wrong.

## Gap 1 — the 2D walk (30 points)

Nothing is implemented. Required:

- Two coins, `C = C₁ ⊗ C₂`, so a 4-dimensional coin space
- 2D initial state with `µ, χ` and the orthogonal components `|ψ₁^⊥⟩, |ψ₂^⊥⟩`
- 10 selected features → 10 parameters
- A 2D shift on a 2D lattice
- Permutation search over `10!` — exhaustive is impossible, so a documented
  heuristic is needed (random restarts, greedy swaps, or a fixed sample)
- `QW_1D vs QW_2D` and `2D with vs without (µ,χ)` comparisons

**The PDF does not define the 2D shift operator.** `C = C₁ ⊗ C₂` and the initial
state are given, but how the four coin states map onto lattice moves is left
open. This needs a decision, and the report must state it.

## Gap 2 — rich feature extraction (~8 points)

§21 asks for ≥5 features spanning three categories, **at least one from each**,
explicitly to move beyond "the position distribution alone". Current set:

| Feature | Category | Notes |
|---|---|---|
| `mean` | A / C | doubles as `⟨x̂⟩` |
| `std` | A | spec asks for **Variance**, not std |
| `skew` | A | matches |
| `kurtosis` | A | spec subtracts 3 (**excess** kurtosis); ours does not |
| `entropy` | A | matches |
| `ipr` | — | not in the spec list (harmless) |
| `p_max` | B | this *is* the Localization Measure |
| `argmax_x` | — | not in the spec list |
| `p_left` | — | not in the spec list |
| `p_origin` | — | not in the spec list |

Missing outright:

- **Interference Pattern** `Σ_x |⟨x|ψ_T⟩|·cos(arg⟨x|ψ_T⟩)` — category B
- **Spread Measure** `#{x : P_T(x) > ε} / total` — category B
- **`⟨x̂²⟩`** and **`⟨p̂⟩`** — category C

The first and last need **amplitudes**, not just `P_T(x)`. Our
`extract_features(psi)` already receives the full state, so the signature
supports them; they simply were not written. Four of our ten slots are spent on
features the rubric never asks for, which is the cheapest thing to fix here.

> Note: the interference formula writes `⟨x|ψ_T⟩` as a scalar, but with a coin
> the amplitude at `x` is a 2-vector. Summing amplitudes over coin states before
> taking modulus and phase is the natural reading; whichever is chosen must be
> stated.

## Gap 3 — reporting details

- **`Time` column** in the §18 results table — runtime is not currently measured.
- **Statistical significance at `α = 0.05`** (question 8) — no test is run.
  Paired comparison across CV folds, or McNemar on the test predictions.
- **Question 7** — whether features correlated with others map, in the optimal
  parameterisation, to parameters contributing more to entanglement. Requires 2D
  (entanglement only exists there) plus a correlation analysis.
- **Required table shape** — §18 wants 1D and 2D rows side by side with
  `Selected Features` and `Final Dimension` columns.
- Kurtosis should subtract 3; variance should be reported rather than std.

## Note on parameter ranges

The PDF **does not prescribe** `θ ∈ [0,π]`, `φ ∈ [0,2π]`, `α ∈ [0,π/2]`,
`β ∈ [0,2π]`. §8 only requires that a scaling method be defined, applied before
the walk, and reported. Our ranges are a defensible choice and are logged on
every run, which is what §8 asks for — but they are ours, not the challenge's.
An alternative such as `α ∈ [0,π]` is equally admissible.

This also revises an earlier criticism of `first.ipynb`: its `α ∈ [0,π]` is
**not** a spec violation. Its real problems stand — the walker starts at the
lattice edge so `Σ_x P_T(x)` decays to a median of 0.17 (violating acceptance
criterion 6), and its permutation search selects on the test set (violating
criterion 16).

Conversely, `first.ipynb`'s feature set — FFT-based interference, a
gradient-based momentum proxy, `⟨x²⟩`, and a spread measure — maps onto
categories B and C **better than ours does**. That part is worth porting.

---

## Decisions taken (not yet implemented)

Recorded so the report can state them. **No 2D code has been written yet.**

### 2D shift convention

The PDF leaves the 2D shift undefined. Chosen: **coin 1 drives x, coin 2 drives
y**. For coin basis `|c₁c₂⟩` on lattice site `(x, y)`:

| coin state | move |
|---|---|
| `|00⟩` | `(x−1, y−1)` |
| `|01⟩` | `(x−1, y+1)` |
| `|10⟩` | `(x+1, y−1)` |
| `|11⟩` | `(x+1, y+1)` |

Each coin controls its own axis, which is the natural partner to `C = C₁ ⊗ C₂` —
the coin factorises across axes, so the shift does too. Every step is diagonal,
so the walker occupies one parity sublattice; this must be accounted for in the
position readout and noted in the report. The alternative (alternating x/y
flip-flop, Grover-style) was considered and rejected as a worse fit to the
tensor-product coin.

### Datasets

The full experiment runs on **all three** — Breast Cancer Wisconsin, Spambase,
and Pima Diabetes — rather than the single dataset §14 requires. This directly
supports task §16.19 ("conclude under what conditions the walk does or does not
improve performance"), for which the three datasets already give contrasting
evidence in 1D:

| dataset | original | qw | original+qw |
|---|---:|---:|---:|
| diabetes | 0.7489 | 0.8485 | **0.8528** |
| breast_cancer | 0.9708 | 0.9298 | **0.9766** |
| spam | **0.9283** | 0.7683 | 0.9276 |

(default settings, untuned; the walk helps where the linear baseline is weak and
does not where it is already strong.)

**Note for 2D on diabetes:** it has only 8 features but 2D needs 10 selected
features mapped to 10 parameters (§7). Selecting 10 of 8 is impossible, so this
needs a stated resolution — reuse features across parameters, pad with
polynomial terms, or restrict 2D to the two wider datasets. To be decided before
2D work starts.

### Sequencing

On hold at the user's request. Remaining work, in the order proposed:

1. 2D walk — `C₁⊗C₂`, `µ/χ` initial state, 10-feature selection, `10!` search
   heuristic, `QW_1D vs QW_2D` and `2D ± (µ,χ)` comparisons *(30 pts)*
2. Feature categories — Interference Pattern, Spread Measure, `⟨x̂²⟩`, `⟨p̂⟩`;
   excess kurtosis and variance *(~8 pts)*
3. Significance testing at `α = 0.05` (question 8), `Time` column, §18 table shape
