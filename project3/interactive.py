"""Task 5: the user plays the expert.

The active-learning loop from Task 4, but the queried "expert" is the person
in the browser: the system picks an article, the user assigns a topic, and the
rejector g(x) — now a model of *this user's* competence — is refit on their
answers. All heavy assets are cached per process; the Django session only
stores queried row indices and the user's labels.
"""

from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import data, ml

MODELS_DIR = None  # set lazily to keep this module import-light for tests

POOL_SIZE = 400          # articles the user can be asked about (stratified)
EVAL_SIZE = 1000         # held-out slice the live team metrics are computed on
WARMUP = 10              # random queries before the boundary strategy kicks in


def available(models_dir):
    return data.dataset_available() and ml.classifier_available(models_dir)


@lru_cache(maxsize=1)
def _cached_assets(models_dir):
    pipe = ml.load_classifier(models_dir)
    vectorizer = pipe.named_steps["tfidf"]

    train_df, _ = data.load_train_val()
    rng = np.random.default_rng(data.RANDOM_STATE)
    # Stratified pool: equal number of articles per topic, so the user's
    # competence profile is probed on every class.
    per_class = POOL_SIZE // data.N_CLASSES
    pool_idx = np.concatenate([
        rng.choice(np.flatnonzero(train_df["label"].values == k), per_class,
                   replace=False)
        for k in range(data.N_CLASSES)
    ])
    rng.shuffle(pool_idx)
    pool_df = train_df.iloc[pool_idx].reset_index(drop=True)

    eval_idx = rng.choice(
        np.setdiff1d(np.arange(len(train_df)), pool_idx), EVAL_SIZE, replace=False
    )
    eval_df = train_df.iloc[eval_idx].reset_index(drop=True)

    X_pool = vectorizer.transform(pool_df["text"])
    X_eval = vectorizer.transform(eval_df["text"])
    proba_pool = pipe.predict_proba(pool_df["text"])
    proba_eval = pipe.predict_proba(eval_df["text"])
    classes = np.asarray(pipe.classes_)
    return {
        "pool_df": pool_df,
        "eval_df": eval_df,
        "X_pool": X_pool,
        "X_eval": X_eval,
        "clf_conf_pool": proba_pool.max(axis=1),
        "clf_pred_eval": classes[proba_eval.argmax(axis=1)],
        "clf_conf_eval": proba_eval.max(axis=1),
    }


def new_state():
    return {"queried": [], "labels": [], "skipped": [], "finished": False}


def _fit_g(assets, state):
    """Rejector over the user's answers so far; None until both outcomes seen."""
    if len(state["queried"]) < 2:
        return None
    idx = np.array(state["queried"])
    gold = assets["pool_df"]["label"].values[idx]
    correct = (np.array(state["labels"]) == gold).astype(int)
    if len(np.unique(correct)) < 2:
        return None
    g = LogisticRegression(C=1.0, max_iter=1000, random_state=data.RANDOM_STATE)
    g.fit(assets["X_pool"][idx], correct)
    return g


def next_query(models_dir, state):
    """Deterministic pick of the next article to show (None when pool is done)."""
    assets = _cached_assets(models_dir)
    excluded = set(state["queried"]) | set(state.get("skipped", []))
    candidates = np.array([i for i in range(len(assets["pool_df"])) if i not in excluded])
    if len(candidates) == 0:
        return None

    g = _fit_g(assets, state) if len(state["queried"]) >= WARMUP else None
    if g is None:
        rng = np.random.default_rng(data.RANDOM_STATE + len(state["queried"]))
        return int(rng.choice(candidates))
    g_pool = g.predict_proba(assets["X_pool"][candidates])[:, 1]
    boundary = np.abs(g_pool - assets["clf_conf_pool"][candidates])
    return int(candidates[int(np.argmin(boundary))])


def article(models_dir, idx):
    row = _cached_assets(models_dir)["pool_df"].iloc[idx]
    return {"index": int(idx), "text": row["text"]}


def record_label(state, idx, label):
    if idx not in state["queried"]:
        state["queried"].append(int(idx))
        state["labels"].append(int(label))
    return state


def record_skip(state, idx):
    """Leave an article unlabeled but take it out of the query pool."""
    idx = int(idx)
    skipped = state.setdefault("skipped", [])
    if idx not in state["queried"] and idx not in skipped:
        skipped.append(idx)
    return state


def record_finish(state):
    """The user chose to stop; live_metrics() already reflects their answers."""
    state["finished"] = True
    return state


def live_metrics(models_dir, state):
    """The sidebar numbers: the user's measured competence and, once the
    rejector can be fit, estimated team metrics on the eval slice."""
    assets = _cached_assets(models_dir)
    n = len(state["queried"])
    out = {"n_queries": n, "pool_size": len(assets["pool_df"]), "warmup": WARMUP}
    if n == 0:
        return out

    idx = np.array(state["queried"])
    gold = assets["pool_df"]["label"].values[idx]
    user = np.array(state["labels"])
    correct = user == gold
    out["user_accuracy"] = float(correct.mean())
    out["per_class"] = [
        {
            "class": data.CLASS_NAMES[k],
            "accuracy": float(correct[gold == k].mean()) if (gold == k).any() else None,
            "n": int((gold == k).sum()),
        }
        for k in range(data.N_CLASSES)
    ]

    g = _fit_g(assets, state)
    if g is not None:
        g_eval = g.predict_proba(assets["X_eval"])[:, 1]
        defer = g_eval > assets["clf_conf_eval"]
        y = assets["eval_df"]["label"].values
        kept_correct = assets["clf_pred_eval"][~defer] == y[~defer]
        # Kept part is exact (gold known); deferred part uses g-hat as the
        # expected chance the user would answer correctly — an estimate.
        est = (kept_correct.sum() + g_eval[defer].sum()) / len(y)
        out["deferral_rate"] = float(defer.mean())
        out["estimated_team_accuracy"] = float(est)
        out["classifier_alone"] = float((assets["clf_pred_eval"] == y).mean())
    return out
