"""Task 4: active learning for expert competence discovery.

Setting: the classifier is trained on the full labeled train set, but no
expert labels exist. The learner may query the (simulated) expert on chosen
pool examples; each answer reveals whether the expert was right there. After
every batch the rejector g(x) = P(expert correct | x) is refit on the queried
examples, and the human-AI team is evaluated on the validation split with the
fixed rule "defer iff g(x) > max_y p_clf(y|x)".

Strategies compared (3 seeds each):
- random: uniform queries (baseline)
- clf_uncertainty: query where the classifier is least confident (classic AL
  baseline; finds hard inputs but ignores the expert)
- deferral_boundary: query where |g(x) - max_y p_clf(y|x)| is smallest, i.e.
  where the defer/keep decision itself is most uncertain — the label there is
  exactly the information that improves deferral. First batch random,
  epsilon = 0.1 of each later batch stays random for exploration.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import data, experts, l2d, ml

POOL_SIZE = 30_000
BUDGETS = [25, 50, 100, 250, 500, 1000, 2000, 5000]
BATCH = 100
SEEDS = [0, 1, 2]
EPSILON = 0.1


def _fit_g(X_queried, correct_queried):
    """Refit the rejector on the queried set; None while only one class seen."""
    if len(np.unique(correct_queried)) < 2:
        return None
    g = LogisticRegression(C=1.0, max_iter=1000, random_state=data.RANDOM_STATE)
    g.fit(X_queried, correct_queried)
    return g


def _g_prob(g, X, prior):
    if g is None:
        return np.full(X.shape[0], prior)
    return g.predict_proba(X)[:, 1]


def select_random(rng, candidates, k, **_):
    return rng.choice(candidates, size=k, replace=False)


def select_clf_uncertainty(rng, candidates, k, clf_conf=None, **_):
    return candidates[np.argsort(clf_conf[candidates])[:k]]


def select_deferral_boundary(rng, candidates, k, clf_conf=None, g_pool=None, **_):
    if g_pool is None:
        return select_random(rng, candidates, k)
    n_random = int(round(EPSILON * k))
    boundary = np.abs(g_pool[candidates] - clf_conf[candidates])
    picked = candidates[np.argsort(boundary)[: k - n_random]]
    rest = np.setdiff1d(candidates, picked, assume_unique=True)
    if n_random and len(rest):
        picked = np.concatenate([picked, rng.choice(rest, size=n_random, replace=False)])
    return picked


STRATEGIES = {
    "random": select_random,
    "clf_uncertainty": select_clf_uncertainty,
    "deferral_boundary": select_deferral_boundary,
}


def run_strategy(select_fn, seed, pool, val, expert_prior):
    """One AL run; returns a checkpoint dict per budget in BUDGETS."""
    X_pool, correct_pool, clf_conf_pool = pool
    y_val, clf_pred_val, expert_pred_val, clf_conf_val, X_val, full_info_defer = val

    rng = np.random.default_rng(seed)
    n = X_pool.shape[0]
    queried = np.zeros(n, dtype=bool)
    g = None
    checkpoints = []

    while queried.sum() < max(BUDGETS):
        # Never step past the next checkpoint, so every budget is hit exactly.
        next_budget = min(b for b in BUDGETS if b > queried.sum())
        k = min(BATCH, next_budget - queried.sum())
        candidates = np.flatnonzero(~queried)
        g_pool = None if g is None else _g_prob(g, X_pool, expert_prior)
        picked = select_fn(rng, candidates, k, clf_conf=clf_conf_pool, g_pool=g_pool)
        queried[picked] = True
        g = _fit_g(X_pool[queried], correct_pool[queried])

        n_queried = int(queried.sum())
        if n_queried in BUDGETS:
            g_val = _g_prob(g, X_val, expert_prior)
            defer = g_val > clf_conf_val
            stats = l2d.evaluate_team(y_val, clf_pred_val, expert_pred_val, defer)
            checkpoints.append({
                "n_queries": n_queried,
                "team_accuracy": stats["team_accuracy"],
                "coverage": stats["coverage"],
                "deferral_precision": stats["deferral_precision"],
                "deferral_recall": stats["deferral_recall"],
                "agreement_with_full_info": float((defer == full_info_defer).mean()),
            })
    return checkpoints


def run_task4(models_dir):
    pipe = ml.load_classifier(models_dir)
    vectorizer = pipe.named_steps["tfidf"]
    expert = experts.default_experts()["class_conditional"]

    train_df, val_df = data.load_train_val()
    rng = np.random.default_rng(data.RANDOM_STATE)
    pool_df = train_df.iloc[
        rng.choice(len(train_df), POOL_SIZE, replace=False)
    ].reset_index(drop=True)

    X_pool = vectorizer.transform(pool_df["text"])
    correct_pool = (expert.predict(pool_df) == pool_df["label"].values).astype(int)
    clf_conf_pool = pipe.predict_proba(pool_df["text"]).max(axis=1)
    expert_prior = float(correct_pool.mean())  # only revealed through queries;
    # used solely as the g-prior before the first refit, where any constant works.

    X_val = vectorizer.transform(val_df["text"])
    proba_val = pipe.predict_proba(val_df["text"])
    clf_pred_val = np.asarray(pipe.classes_)[proba_val.argmax(axis=1)]
    clf_conf_val = proba_val.max(axis=1)
    y_val = val_df["label"].values
    expert_pred_val = expert.predict(val_df)

    # Upper bound: the Task 3 rejector (trained on all 110k expert labels),
    # applied with the same tau=0 rule used during active learning.
    import joblib, os
    full_rejector = joblib.load(os.path.join(models_dir, l2d.REJECTOR_FILE))
    g_full = full_rejector.predict_proba(X_val)[:, 1]
    full_info_defer = g_full > clf_conf_val
    full_info_stats = l2d.evaluate_team(
        y_val, clf_pred_val, expert_pred_val, full_info_defer
    )

    pool = (X_pool, correct_pool, clf_conf_pool)
    val = (y_val, clf_pred_val, expert_pred_val, clf_conf_val, X_val, full_info_defer)

    results = {
        "pool_size": POOL_SIZE,
        "budgets": BUDGETS,
        "batch": BATCH,
        "seeds": SEEDS,
        "expert": expert.describe(),
        "references": {
            "classifier_alone_val": float((clf_pred_val == y_val).mean()),
            "full_info_team_accuracy_val": full_info_stats["team_accuracy"],
        },
        "strategies": {},
    }
    for name, fn in STRATEGIES.items():
        runs = [run_strategy(fn, seed, pool, val, expert_prior) for seed in SEEDS]
        results["strategies"][name] = {"runs": runs}
        final = np.mean([r[-1]["team_accuracy"] for r in runs])
        print(f"task4: {name} val team acc @ {max(BUDGETS)} queries: {final:.4f}")
    return results
