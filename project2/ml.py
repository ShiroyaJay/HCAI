"""Models for Project 2: decision trees and logistic regression.

Implements equation (1) from the brief at the *model selection* level: for a given
regularization weight ``lambda`` we pick, among models of varying complexity, the one
maximizing ``acc_test - lambda * Omega(f)``.

  * Decision tree:        Omega(f) = number of leaves   (fit knob: max_leaf_nodes)
  * Logistic regression:  Omega(f) = number of non-zero coefficients (fit knob: C, L1)

Note that ``lambda`` here is the selection meta-parameter and is deliberately distinct
from the per-model fitting knob (max_leaf_nodes / C).
"""

from functools import lru_cache

import matplotlib

matplotlib.use("Agg")  # no display on a server — render straight to a file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from . import data

# Slider range for lambda. Chosen so the slider sweeps the full
# accuracy<->simplicity spectrum for this dataset. Tunable.
LAMBDA_MIN = 0.0
LAMBDA_MAX = 0.25
LAMBDA_STEP = 0.005

MAX_TREE_LEAVES = 30  # cap on candidate tree complexity
COEF_TOL = 1e-6  # |coef| below this counts as zero


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def _tree_pipeline(max_leaf_nodes):
    pre = ColumnTransformer(
        [
            ("num", "passthrough", data.NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(), data.CATEGORICAL_FEATURES),
        ]
    )
    clf = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes, random_state=data.RANDOM_STATE
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _logreg_pipeline(C):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), data.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), data.CATEGORICAL_FEATURES),
        ]
    )
    # saga + 3 classes => multinomial (softmax) automatically; softmax is what makes
    # the exact ALE derivatives in feature_effects.py valid.
    clf = LogisticRegression(
        penalty="l1", solver="saga", C=C, max_iter=5000,
        random_state=data.RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def tree_candidates(X_train, y_train, X_test, y_test):
    """Trees of increasing complexity (max_leaf_nodes = 2 .. full tree)."""
    full = _tree_pipeline(None).fit(X_train, y_train)
    max_leaves = min(full.named_steps["clf"].get_n_leaves(), MAX_TREE_LEAVES)

    candidates = []
    for k in range(2, max_leaves + 1):
        pipe = _tree_pipeline(k).fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        omega = int(pipe.named_steps["clf"].get_n_leaves())
        candidates.append({"label": str(omega), "omega": omega, "acc": acc, "pipe": pipe})
    return candidates


def logreg_candidates(X_train, y_train, X_test, y_test):
    """Logistic regressions of increasing complexity via the L1 strength C."""
    candidates = []
    for C in np.logspace(-2, 2, 12):
        pipe = _logreg_pipeline(C).fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        omega = int(np.sum(np.abs(pipe.named_steps["clf"].coef_) > COEF_TOL))
        candidates.append({"label": str(omega), "omega": omega, "acc": acc, "pipe": pipe})
    return candidates


def get_candidates(model_type, X_train, y_train, X_test, y_test):
    if model_type == "logreg":
        return logreg_candidates(X_train, y_train, X_test, y_test)
    return tree_candidates(X_train, y_train, X_test, y_test)


@lru_cache(maxsize=2)
def cached_candidates(model_type):
    """Candidates for ``model_type``, fitted once per process.

    The dataset and split are deterministic (fixed seed), so the candidate set is the
    same on every request — caching avoids refitting every model on each slider move.
    """
    df = data.load_penguins()
    X_train, X_test, y_train, y_test = data.get_split(df)
    return get_candidates(model_type, X_train, y_train, X_test, y_test)


def select_model(candidates, lam):
    """Maximizer of ``acc - lambda * omega``; ties broken toward the simpler model."""
    return max(candidates, key=lambda c: (c["acc"] - lam * c["omega"], -c["omega"]))


# --------------------------------------------------------------------------- #
# Model views (plots / tables)
# --------------------------------------------------------------------------- #
def plot_tree_png(pipe, class_names, save_path):
    """Render the fitted decision tree to a PNG."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    feature_names = list(pre.get_feature_names_out())
    # Strip the ColumnTransformer's "num__"/"cat__" prefixes for readability.
    feature_names = [n.split("__", 1)[-1] for n in feature_names]

    n_leaves = clf.get_n_leaves()
    fig, ax = plt.subplots(figsize=(min(2.2 * n_leaves, 22), 8))
    plot_tree(
        clf, feature_names=feature_names, class_names=list(class_names),
        filled=True, rounded=True, impurity=False, fontsize=9, ax=ax,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def logreg_coef_png(pipe, class_names, save_path):
    """Grouped horizontal bar chart of the logistic-regression coefficients."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    feature_names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    coef = clf.coef_  # (n_classes, n_features)

    n_features = len(feature_names)
    y = np.arange(n_features)
    height = 0.8 / len(class_names)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.5 * n_features + 1.5)))
    for i, cls in enumerate(class_names):
        ax.barh(y + i * height, coef[i], height=height, label=str(cls))
    ax.set_yticks(y + height * (len(class_names) - 1) / 2)
    ax.set_yticklabels(feature_names)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("coefficient (standardized features)")
    ax.legend(title="species")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
