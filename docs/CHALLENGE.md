# PQO–2026–06 — Feature Space Expansion Using Quantum Walks

PolyHaq Quantum Hackathon. Source: [`PQO-2026-06-challenge.pdf`](PQO-2026-06-challenge.pdf)
(Persian). This is a working English transcription of the requirements — the PDF
is authoritative where they disagree.

> *Walk. Expand. Compare. Find the best feature space.*

Duration: 18 hours, team-based.

---

## 1. Mission

Use a discrete-time quantum walk as a feature-expansion mechanism and compare it
against classical alternatives:

```
X_original  vs  X_polynomial  vs  X_QW  vs  X_original ⊕ X_QW
```

Participants must implement a **1D walk first, then generalise it to 2D**.

---

## 2–3. Physics (fixed by the challenge)

One step is a coin followed by a shift:

```
|ψ_{t+1}⟩ = S C |ψ_t⟩          |ψ_T⟩ = (S C)^T |ψ_0⟩
P_T(x) = Σ_c |⟨x, c | ψ_T⟩|²
```

Coin operator, exactly:

```
C(α, β) = [[cos(α)·e^{iβ},   sin(α)      ],
           [-sin(α),         cos(α)·e^{-iβ}]]
```

Single-qubit initial state:

```
|ψ_k⟩ = cos(θ_k/2)|0⟩ + e^{iφ_k} sin(θ_k/2)|1⟩
```

All parameters must be implemented as **configurable**.

## 4. 1D parameters

One qubit, one coin: `C₁ = C(α₁, β₁)`.

```
θ₁, φ₁, α₁, β₁          N_param^1D = 4
```

Four dataset features are selected and mapped onto these four parameters.

## 5–6. 2D parameters

A second degree of freedom enters. The total coin is a tensor product:

```
C = C₁ ⊗ C₂
```

Extra parameters `θ₂, φ₂` (second qubit state), `α₂, β₂` (second coin), plus
`µ, χ` which define how the two initial-state components combine:

```
|Ψ₀⟩ = cos(µ/2)|ψ₁⟩|ψ₂⟩ + e^{iχ} sin(µ/2)|ψ₁^⊥⟩|ψ₂^⊥⟩

|ψ₁^⊥⟩ = sin(θ₁/2)|0⟩ − e^{iφ₁} cos(θ₁/2)|1⟩
|ψ₂^⊥⟩ = sin(θ₂/2)|0⟩ − e^{iφ₂} cos(θ₂/2)|1⟩
```

`µ` sets the relative amplitude of the two components, `χ` the relative phase.
At `µ = 0` the state is separable (`|Ψ₀⟩ = |ψ₁⟩|ψ₂⟩`); at general `µ` it can be
**non-separable / entangled**. The effect of `µ, χ` must be studied.

```
θ₁, φ₁, α₁, β₁, θ₂, φ₂, α₂, β₂, µ, χ          N_param^2D = 10
```

## 7. Feature selection and mapping

The number of selected features must **equal** the number of walk parameters:

```
1D:   4 features  →  4 QW parameters
2D:  10 features  →  10 QW parameters
```

The selection method is the team's choice but **must be reported**, and the
feature→parameter assignment must be explicit and reproducible.

**Mapping optimisation.** The assignment matters. There are `4! = 24`
permutations in 1D and `10! = 3,628,800` in 2D. Teams must search for the best
mapping and **explain analytically** why a particular parameterisation wins.

## 8. Scaling features to parameters

Selected features generally sit in a different range from the quantum
parameters, so a scaling/normalisation `x_i → f(x_i) → θ` must be defined and
applied **before** running the walk. The method must be reported.

> **The PDF does not prescribe the parameter ranges themselves.** Choosing them
> (and reporting the choice) is part of the submission.

## 10. The four feature spaces

| # | Space | Description |
|---|---|---|
| 1 | `X_original` | no expansion — the baseline |
| 2 | `X_polynomial` | polynomial expansion, **degree 2 mandatory at minimum** |
| 3 | `X_QW` | walk features only |
| 4 | `X_original ⊕ X_QW` | originals kept, walk features appended |

## 11–13. Required studies

- All four spaces, for **both 1D and 2D**.
- `QW_1D vs QW_2D` — does the second degree of freedom help?
- `2D without (µ,χ) vs 2D with (µ,χ)`.
- Number of steps: run `T = 1, 2, …, T_max`, report `A(T)`, identify
  `T* = argmax_T A(T)` — **and** describe behaviour *around* the optimum, not
  just a single value.

## 14. Datasets

At least one of: **Breast Cancer Wisconsin**, **Spambase**, **Pima Indians
Diabetes**. The full experiment must be run on the chosen dataset.

## 15. Machine-learning model

Explicitly *not* a modelling competition. A simple classifier is fine. The model,
evaluation method, data split and all other experimental conditions must stay
**identical across the four feature spaces**, so the comparison isolates the
effect of feature expansion. Metrics: **Accuracy** and **F1** (Precision and
Recall optional).

## 16. Participant tasks

1. Implement the 1D quantum walk
2. Implement the 2D quantum walk
3. Implement the single-qubit initial state with `θ, φ`
4. Implement the 2D initial state with `µ, χ`
5. Choose a feature-selection method
6. Select 4 features for 1D
7. Select 10 features for 2D
8. Define the feature→parameter mapping
9. Find the optimal mapping by permutation search (`4!` / `10!`) and analyse it
10. Map features into appropriate quantum parameter ranges
11. Run the walk for several values of `T`
12. Extract the new walk features
13. Test the four feature spaces
14. Run polynomial expansion as the classical baseline
15. Compare 1D and 2D results
16. Study the effect of `µ, χ`
17. Determine the best `T` or a suitable range
18. Present results as tables and charts
19. Conclude under what conditions the walk does or does not improve performance

## 18. Required results table

| Dimension | Feature Space | T | Selected Features | Final Dimension | Accuracy | F1 | Time |
|---|---|---|---|---|---|---|---|
| 1D | Original | — | — | — | | | |
| 1D | Polynomial-2 | — | — | — | | | |
| 1D | QW | | 4 | | | | |
| 1D | Original + QW | | 4 | | | | |
| 2D | Original | — | — | — | | | |
| 2D | Polynomial-2 | — | — | — | | | |
| 2D | QW | | 10 | | | | |
| 2D | Original + QW | | 10 | | | | |

`µ` and `χ` values must also be recorded for the 2D experiments.

## 20. Acceptance criteria

A submission is complete when:

1. 1D quantum walk correctly implemented
2. 2D quantum walk correctly implemented
3. The coin matches the challenge definition
4. The initial state matches the problem definition
5. The 2D state with `µ, χ` correctly implemented
6. **The final state is normalised: `Σ_x P_T(x) ≈ 1`**
7. Feature selection performed correctly
8. Number of input features matches the number of walk parameters
9. The feature→parameter mapping is explicit
10. All four feature spaces evaluated
11. Results reported for several values of `T`
12. 1D vs 2D comparison performed
13. Effect of `µ, χ` studied
14. Experimental conditions as identical as possible across methods
15. The ML model held fixed across the main comparisons
16. **Test-set information not used for feature selection or parameter choice**
17. Results and method reported reproducibly

## 21–22. Rich feature extraction (mandatory)

Rather than using the position distribution alone, teams must extract **at least
5 features spanning the categories below, with at least one from each**.

**A. Statistical features of the distribution**

```
Mean     = Σ_x x·P_T(x)
Variance = Σ_x (x − Mean)²·P_T(x)
Skewness = Σ_x (x − Mean)³·P_T(x) / Variance^{3/2}
Kurtosis = Σ_x (x − Mean)⁴·P_T(x) / Variance²  − 3
Entropy  = −Σ_x P_T(x)·log P_T(x)
```

**B. Phase- and interference-based features**

```
Interference Pattern = Σ_x |⟨x|ψ_T⟩| · cos(arg⟨x|ψ_T⟩)
Localization Measure = max_x P_T(x)
Spread Measure       = #{x : P_T(x) > ε} / (total positions)
```

**C. Measurement-operator-based features**

```
⟨x̂⟩   = ⟨ψ_T|x̂|ψ_T⟩
⟨x̂²⟩  = ⟨ψ_T|x̂²|ψ_T⟩
⟨p̂⟩   = ⟨ψ_T|p̂|ψ_T⟩
```

Categories B and C require **amplitudes**, not just `P_T(x)`.

The report must explain: why these features were chosen, what quantum dynamics
they represent, and how they help the learning problem.

## 23. Scoring — 100 points

| Section | Points |
|---|---:|
| Correct 1D quantum walk implementation and feature-space generation | 20 |
| Correct generalisation to the 2D quantum walk | 20 |
| Rich feature extraction (≥5 features across categories) | 15 |
| Feature selection and correct mapping to parameters | 10 |
| Running the four feature spaces and a fair comparison | 10 |
| Systematic study of the number of steps `T` | 10 |
| Study of the 1D/2D effect and `µ, χ` | 10 |
| Quality of analysis, report and presentation | 5 |
| **Total** | **100** |

## 24. Questions the report must answer

1. Does the quantum walk improve on the original features? `X_original vs X_QW`
2. Does it beat polynomial expansion? `X_polynomial vs X_QW`
3. Is appending walk features better than walk features alone?
   `X_original vs X_original ⊕ X_QW`
4. Does 2D produce a more useful feature space than 1D? `QW_1D vs QW_2D`
5. Do `µ, χ` and a combined/entangled initial state improve performance?
   `2D without (µ,χ) vs 2D with (µ,χ)`
6. What is the right number of steps, and is performance stable around it?
   `T* = argmax_T A(T)`
7. Can it be confirmed that features **correlated with other features** map, in
   the optimal parameterisation, to parameters contributing more to
   **entanglement**?
8. Are the performance differences between methods **statistically significant**
   at `α = 0.05`?

## 25. References

**Quantum walks**

1. Aharonov, Y., Davidovich, L., Zagury, N., "Quantum Random Walks",
   *Physical Review A*, 48, 1687–1690, 1993.
2. Kempe, J., "Quantum Random Walks: An Introductory Overview",
   *Contemporary Physics*, 44, 307–327, 2003.
3. Venegas-Andraca, S. E., "Quantum Walks: A Comprehensive Review",
   *Quantum Information Processing*, 11, 1015–1106, 2012.

**Feature engineering and polynomial expansion**

1. Hastie, T., Tibshirani, R., Friedman, J., *The Elements of Statistical
   Learning*, 2nd ed., Springer, 2009.
