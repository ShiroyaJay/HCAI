"""Task 3: learning-to-defer via a post-hoc rejector.

A joint surrogate loss (Mozannar & Sontag, 2020) needs a custom (K+1)-way
weighted loss that sklearn's estimators cannot express, so we use the staged
approach instead: keep the Task 1 classifier fixed, train a *rejector*
``g(x) = P(expert correct | x)`` on the expert's answers, and defer whenever
the expert looks more likely to be right than the classifier:

    defer(x)  <=>  g(x) - max_y p_clf(y | x) > tau

``tau`` is tuned on the validation split; the test set is scored once.
"""

import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import data, experts, ml

REJECTOR_FILE = "rejector.joblib"


def train_rejector(X, expert_pred, gold):
    """Fit g(x) = P(expert correct | x) on already-vectorized features."""
    correct = (expert_pred == gold).astype(int)
    rejector = LogisticRegression(C=1.0, max_iter=1000, random_state=data.RANDOM_STATE)
    rejector.fit(X, correct)
    return rejector


def deferral_scores(clf_proba, g_prob):
    """Positive score = expert looks better than the classifier."""
    return g_prob - clf_proba.max(axis=1)


def team_predictions(clf_pred, expert_pred, defer_mask):
    return np.where(defer_mask, expert_pred, clf_pred)


def evaluate_team(y, clf_pred, expert_pred, defer_mask):
    """Team accuracy plus deferral-quality metrics.

    Deferral decisions are scored against the oracle should-defer set
    {classifier wrong AND expert right}: precision = fraction of deferrals
    that were needed, recall = fraction of needed deferrals that happened.
    """
    team = team_predictions(clf_pred, expert_pred, defer_mask)
    kept = ~defer_mask
    should_defer = (clf_pred != y) & (expert_pred == y)

    tp = (defer_mask & should_defer).sum()
    out = {
        "team_accuracy": float((team == y).mean()),
        "coverage": float(kept.mean()),
        "n_deferred": int(defer_mask.sum()),
        "accuracy_kept": float((clf_pred[kept] == y[kept]).mean()) if kept.any() else None,
        "expert_accuracy_deferred": (
            float((expert_pred[defer_mask] == y[defer_mask]).mean())
            if defer_mask.any() else None
        ),
        "deferral_precision": float(tp / defer_mask.sum()) if defer_mask.any() else None,
        "deferral_recall": float(tp / should_defer.sum()) if should_defer.any() else None,
        "per_class_deferral_rate": [
            {
                "class": data.CLASS_NAMES[k],
                "rate": float(defer_mask[y == k].mean()),
            }
            for k in range(data.N_CLASSES)
        ],
    }
    return out


def coverage_accuracy_curve(y, clf_pred, expert_pred, scores, n_points=21):
    """Team accuracy when deferring the top-q fraction of scores, q in [0, 1]."""
    order = np.argsort(-scores)
    n = len(y)
    curve = []
    for q in np.linspace(0.0, 1.0, n_points):
        defer_mask = np.zeros(n, dtype=bool)
        defer_mask[order[: int(round(q * n))]] = True
        team = team_predictions(clf_pred, expert_pred, defer_mask)
        curve.append({"deferral_rate": float(q), "team_accuracy": float((team == y).mean())})
    return curve


def baselines(y, clf_pred, expert_pred, deferral_rate, seed=data.RANDOM_STATE):
    """Reference points for the deferral system."""
    rng = np.random.default_rng(seed)
    n = len(y)
    random_mask = np.zeros(n, dtype=bool)
    random_mask[rng.choice(n, int(round(deferral_rate * n)), replace=False)] = True

    # Oracle: defer exactly where it helps and never where it hurts.
    oracle_mask = (clf_pred != y) & (expert_pred == y)
    return {
        "classifier_alone": float((clf_pred == y).mean()),
        "expert_alone": float((expert_pred == y).mean()),
        "random_deferral_same_rate": float(
            (team_predictions(clf_pred, expert_pred, random_mask) == y).mean()
        ),
        "oracle_deferral": float(
            (team_predictions(clf_pred, expert_pred, oracle_mask) == y).mean()
        ),
    }


def _predictions(pipe, df):
    proba = pipe.predict_proba(df["text"])
    return proba, np.asarray(pipe.classes_)[proba.argmax(axis=1)]


def run_task3(models_dir):
    """Train the rejector, tune tau on validation, evaluate once on test."""
    pipe = ml.load_classifier(models_dir)
    vectorizer = pipe.named_steps["tfidf"]
    expert = experts.default_experts()["class_conditional"]

    train_df, val_df = data.load_train_val()
    test_df = data.load_test()

    X_train = vectorizer.transform(train_df["text"])
    rejector = train_rejector(
        X_train, expert.predict(train_df), train_df["label"].values
    )
    joblib.dump(rejector, os.path.join(models_dir, REJECTOR_FILE))

    results = {"expert": expert.describe()}
    # Tune tau on validation: maximize team accuracy over a symmetric grid.
    taus = np.round(np.linspace(-0.3, 0.3, 61), 3)
    split_cache = {}
    for split, df in [("val", val_df), ("test", test_df)]:
        proba, clf_pred = _predictions(pipe, df)
        g_prob = rejector.predict_proba(vectorizer.transform(df["text"]))[:, 1]
        split_cache[split] = (df["label"].values, clf_pred, expert.predict(df),
                              deferral_scores(proba, g_prob))

    y, clf_pred, expert_pred, scores = split_cache["val"]
    val_accs = []
    for tau in taus:
        team = team_predictions(clf_pred, expert_pred, scores > tau)
        val_accs.append(float((team == y).mean()))
    best_tau = float(taus[int(np.argmax(val_accs))])
    results["tau"] = best_tau
    results["tau_sweep"] = [
        {"tau": float(t), "val_team_accuracy": a} for t, a in zip(taus, val_accs)
    ]

    for split in ("val", "test"):
        y, clf_pred, expert_pred, scores = split_cache[split]
        defer_mask = scores > best_tau
        results[split] = evaluate_team(y, clf_pred, expert_pred, defer_mask)
        results[split]["baselines"] = baselines(
            y, clf_pred, expert_pred, defer_mask.mean()
        )
    y, clf_pred, expert_pred, scores = split_cache["test"]
    results["coverage_accuracy_curve"] = coverage_accuracy_curve(
        y, clf_pred, expert_pred, scores
    )
    return results
