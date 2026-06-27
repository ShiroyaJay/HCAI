"""Global model-agnostic feature effects (brief, section 3) — hand-written.

No PDP/ALE library is used; everything here is plain numpy/pandas.

PDP(v) for feature j = average over the data of the predicted probability when feature
j is forced to value v. One curve per species.

ALE = accumulated *local* effects, centered to mean zero. ALE is the integral of the
partial derivative E[ df/dx_j ] of the prediction with respect to x_j:

  * Logistic regression -> the predicted probability is differentiable, so we use the
    EXACT analytic derivative (softmax gradient) and accumulate it over a grid.
  * Decision tree -> the prediction is piecewise-constant (derivative is 0 a.e. with
    jumps), so the analytic derivative is useless. We DISCRETIZE: within feature bins we
    take the difference of predictions across the bin edges and accumulate those.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data

GRID_SIZE = 40
N_BINS = 10


def _set_feature(df, feature, value):
    out = df[data.FEATURES].copy()
    out[feature] = value
    return out


# --------------------------------------------------------------------------- #
# Partial Dependence
# --------------------------------------------------------------------------- #
def pdp(pipe, df, feature, classes):
    grid = np.linspace(df[feature].min(), df[feature].max(), GRID_SIZE)
    curves = np.zeros((len(grid), len(classes)))
    for i, v in enumerate(grid):
        proba = pipe.predict_proba(_set_feature(df, feature, v))
        curves[i] = proba.mean(axis=0)
    return grid, {cls: curves[:, j] for j, cls in enumerate(pipe.classes_)}


# --------------------------------------------------------------------------- #
# ALE — exact derivative (logistic regression)
# --------------------------------------------------------------------------- #
def ale_exact_logreg(pipe, df, feature, classes):
    """True (conditional) ALE using the *exact* softmax partial derivative dP_c/dx_j.

    Unlike the PDP, ALE averages the local effect over the *conditional* distribution:
    within each feature bin we average the analytic derivative over the points that
    actually fall in that bin, multiply by the bin width, and accumulate. This is what
    makes ALE differ from a centered PDP when features are correlated, while still
    answering the brief's "which model has an exact derivative?" with logistic
    regression (the softmax is differentiable in closed form).
    """
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    j = data.NUMERIC_FEATURES.index(feature)
    sigma = pre.named_transformers_["num"].scale_[j]  # std used to standardize x_j
    w_j = clf.coef_[:, j]  # per-class coefficient of the standardized feature

    edges = np.unique(np.quantile(df[feature], np.linspace(0, 1, N_BINS + 1)))
    n_classes = len(classes)
    values = df[feature].values
    local = np.zeros((len(edges) - 1, n_classes))
    counts = np.zeros(len(edges) - 1)

    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        mask = (values >= lo) & (values <= hi) if k == 0 else (values > lo) & (values <= hi)
        counts[k] = mask.sum()
        if counts[k] == 0:
            continue
        # exact derivative at the points' real positions (conditional expectation):
        # dP_c/dz = P_c * (w_cj - sum_m P_m w_mj);  dP_c/dx = dP_c/dz / sigma
        P = pipe.predict_proba(df[mask][data.FEATURES])  # (n_bin, C)
        dot = P @ w_j  # (n_bin,)
        d = P * (w_j[None, :] - dot[:, None]) / sigma  # (n_bin, C)
        local[k] = d.mean(axis=0) * (hi - lo)  # mean local effect x bin width

    # accumulate over bins; curve defined at the edges (first edge = 0)
    ale = np.zeros((len(edges), n_classes))
    for k in range(1, len(edges)):
        ale[k] = ale[k - 1] + local[k - 1]

    # center: weight each edge by neighbouring bin counts
    total = counts.sum()
    if total > 0:
        edge_w = np.zeros(len(edges))
        edge_w[:-1] += counts / 2
        edge_w[1:] += counts / 2
        mean = (ale * edge_w[:, None]).sum(axis=0) / total
        ale -= mean[None, :]
    return edges, {cls: ale[:, j2] for j2, cls in enumerate(pipe.classes_)}


# --------------------------------------------------------------------------- #
# ALE — discretization (decision tree)
# --------------------------------------------------------------------------- #
def ale_discretized(pipe, df, feature, classes):
    """Standard binned ALE: mean prediction difference across each bin's edges."""
    edges = np.unique(np.quantile(df[feature], np.linspace(0, 1, N_BINS + 1)))
    n_classes = len(classes)
    local = np.zeros((len(edges) - 1, n_classes))
    counts = np.zeros(len(edges) - 1)

    values = df[feature].values
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if k == 0:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values > lo) & (values <= hi)
        counts[k] = mask.sum()
        if counts[k] == 0:
            continue
        sub = df[mask]
        p_hi = pipe.predict_proba(_set_feature(sub, feature, hi))
        p_lo = pipe.predict_proba(_set_feature(sub, feature, lo))
        local[k] = (p_hi - p_lo).mean(axis=0)

    # accumulate over bins; curve is defined at the edges (first edge = 0)
    ale = np.zeros((len(edges), n_classes))
    for k in range(1, len(edges)):
        ale[k] = ale[k - 1] + local[k - 1]

    # center: weight each edge by neighbouring bin counts
    total = counts.sum()
    if total > 0:
        edge_w = np.zeros(len(edges))
        edge_w[:-1] += counts / 2
        edge_w[1:] += counts / 2
        mean = (ale * edge_w[:, None]).sum(axis=0) / total
        ale -= mean[None, :]
    return edges, {cls: ale[:, j] for j, cls in enumerate(pipe.classes_)}


def ale(pipe, df, feature, classes, model_type):
    if model_type == "logreg":
        return ale_exact_logreg(pipe, df, feature, classes)
    return ale_discretized(pipe, df, feature, classes)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_curves(x, curves, classes, feature, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for cls in classes:
        ax.plot(x, curves[cls], marker="o", markersize=3, label=str(cls))
    ax.set_xlabel(feature)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="#bbb", linewidth=0.6)
    ax.legend(title="species")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
