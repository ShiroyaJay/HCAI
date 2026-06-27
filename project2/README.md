# Project 2 — Explainability

Interactive interface for the **Palmer Penguins** dataset (target: `species` —
Adelie / Gentoo / Chinstrap). One page, mounted at `/project2/`, with shared controls
(model class + regularization λ) driving three linked regions: the selected model, its
counterfactuals, and its feature-effect plots.

## Files
| File | Purpose |
| --- | --- |
| `data.py` | Load/clean the vendored `data/penguins.csv`, feature lists, split, MADs |
| `ml.py` | Tree & logistic-regression pipelines, candidate generation, λ-selection, model plots |
| `counterfactuals.py` | Local-sampling counterfactuals ranked by MAD-weighted L1 distance |
| `feature_effects.py` | Hand-written PDP and ALE (exact for logreg, discretized for tree) |
| `views.py` | Single `index` view reading all UI state from query params |
| `templates/project2/index.html`, `static/project2/style.css` | UI |

The dataset is vendored as a CSV (rows with missing values dropped → 333 penguins), so no
extra dependency is needed beyond the project's existing `scikit-learn / pandas /
matplotlib / numpy`.

## Conceptual notes (answers to the brief's questions)

### Regularization and model selection (Tasks 1–3)
We optimize at the *model-selection* level: among models of varying complexity we pick
the maximizer of `acc_test − λ·Ω(f)`. **λ here is a selection meta-parameter and is
deliberately distinct** from the per-model fitting knob:

| Model | Fitting knob (complexity control) | Complexity measure Ω(f) |
| --- | --- | --- |
| Decision tree | `max_leaf_nodes` (2 … full tree) | number of **leaves** |
| Logistic regression | L1 penalty strength `C` | number of **non-zero coefficients** |

**Why Ω = number of non-zero coefficients for logistic regression?** It is the natural
analogue of "number of leaves": both count the effective pieces of the model a human has
to read. An L1 penalty drives coefficients to exactly zero, so increasing λ yields a
sparser, more interpretable linear model — fewer features actually used.

On this dataset the tree's test accuracy already peaks at 3 leaves, so the slider picks
the 3-leaf tree for small λ and collapses to the 2-leaf tree only at large λ — a clean
illustration that regularization prefers the *simplest model achieving the best accuracy*.

### Counterfactuals — noising categorical data (Task 4)
We sample N points locally around x and keep those the model predicts as the target
class, ranked by MAD-weighted L1 distance.
* **Numeric** features: Gaussian noise with standard deviation scaled by the feature's
  MAD.
* **Categorical / binary** features (`island`, `sex`, `year`): with some probability the
  value is **resampled to a different category** (you cannot add Gaussian noise to a
  category). The MAD-weighted L1 distance uses a 0/1 mismatch term for these features.
* **Iterative fallback**: if fewer than k counterfactuals are found, N and the noise
  level (numeric variance and categorical flip probability) are increased and we resample.

### PDP and ALE — exact vs. discretized derivatives (Task 5)
Both are implemented from scratch (numpy only).

* **PDP(v)** = average over the data of the predicted probability when the chosen feature
  is forced to `v`; one curve per species.
* **ALE** accumulates the *local* effect, i.e. the integral of the partial derivative
  `E[ ∂f/∂x_j ]`, then centers it to mean zero.
  * **Logistic regression → exact derivative.** The predicted probability is
    differentiable, so we use the analytic softmax gradient
    `∂P_c/∂x_j = P_c · (w_{c,j} − Σ_m P_m w_{m,j}) / σ_j`
    (with `σ_j` the standardization scale and `w_{·,j}` the standardized coefficient).
    We bin the feature and, within each bin, average this exact derivative over the
    points that actually fall in the bin (the *conditional* expectation), then accumulate
    bin-width × mean-derivative. Using the conditional distribution — rather than the
    marginal one a global grid would imply — is what keeps ALE distinct from a centered
    PDP when features are correlated.
  * **Decision tree → discretization.** A tree is piecewise-constant, so its derivative is
    zero almost everywhere with jumps at the splits — useless analytically. We instead bin
    the feature and, within each bin, average the prediction *difference* across the bin's
    edges, then accumulate those binned differences (the standard ALE estimator).

You can see the difference directly: the tree's PDP/ALE are step functions that jump at
the tree's split values, while the logistic-regression curves are smooth.
