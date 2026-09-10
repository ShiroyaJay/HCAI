"""Simulations backing the report's quantitative claims.

None of this involves a human: synthetic users with known preference vectors
answer under each design, which lets the study protocol (Task 3) be justified
with numbers instead of assertion.

Everything here samples from the same familiarity pool the protocol uses, not
from the raw catalogue, so the numbers describe the study that would actually
be run.
"""

import json
import os
from math import ceil
from statistics import NormalDist

import numpy as np

from .. import data, features, preference

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

# Timing assumptions, stated openly in the report: a pairwise click takes ~4s,
# and one sequential pick inside a rank-10 task about the same, so a full
# ten-film ranking costs ~40s. These two numbers decide which design wins, and
# nothing in the dataset pins them down -- hence sensitivity(), which reports
# the conclusion at four different costs for the ranking task.
SEC_PER_PAIR = 4.0
SEC_PER_RANK_SET = 40.0

ALPHA = 0.05
POWER = 0.80

# The genres features.py deliberately leaves out. Adding them back is the d=34
# representation the report argues against; dimension_study() checks that
# argument against held-out performance instead of against a theorem alone.
EXTRA_GENRES = ["Animation", "Biography", "Music", "Musical", "War", "Western",
                "History", "Sport", "Documentary", "Film-Noir"]

# Number of director indicator columns in the third, deliberately over-wide
# condition -- the regime the separability argument is actually about.
N_DIRECTORS = 40

# Estimates of dz below this are indistinguishable from zero at these trial
# counts, and turning one into a required sample size produces a meaningless
# number. Report the difference and its spread instead.
DZ_FLOOR = 0.10

TRIALS = 300


def _sample_ranking(X, items, w_true, rng, n_ranked=None):
    """Draw a Plackett-Luce ranking from a synthetic user."""
    rem, order = list(items), []
    k = len(items) - 1 if n_ranked is None else n_ranked
    for _ in range(k):
        u = X[rem] @ w_true
        p = np.exp(u - u.max()); p /= p.sum()
        order.append(rem.pop(int(rng.choice(len(rem), p=p))))
    return order + rem


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def n_for_dz(dz, alpha=ALPHA, power=POWER):
    """Participants needed for a paired test, by normal approximation."""
    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)
    return int(ceil((z / abs(dz)) ** 2)) + 1


def _one_participant(X, pool, rng, n_pairs, n_sets, X_fit=None, n_holdout=16):
    """One synthetic participant through both designs. Returns (rho_1, rho_2).

    The same true preference vector answers under both interfaces and is scored
    against the same held-out films -- which is what makes the difference a
    WITHIN-participant quantity, exactly as in the real within-subjects design.
    """
    X_fit = X if X_fit is None else X_fit
    w_true = rng.normal(size=X.shape[1]) * 0.8
    holdout = rng.choice(pool, n_holdout, replace=False)
    stated = X[holdout] @ w_true

    pairs = [preference.ranking(_sample_ranking(X, list(rng.choice(pool, 2, replace=False)), w_true, rng))
             for _ in range(n_pairs)]
    sets = [preference.ranking(_sample_ranking(X, list(rng.choice(pool, 10, replace=False)), w_true, rng))
            for _ in range(n_sets)]
    return (_spearman(X_fit[holdout] @ preference.fit_map(pairs, X_fit), stated),
            _spearman(X_fit[holdout] @ preference.fit_map(sets, X_fit), stated))


def recovery(X, pool, budgets_seconds=(60, 120, 180, 300, 420, 600), trials=TRIALS,
             seed=0, sec_per_rank_set=SEC_PER_RANK_SET):
    """Held-out agreement for each design at matched *time* budgets.

    Reports the PAIRED difference as well as the two condition means. The paired
    standard deviation is the quantity the sample-size calculation needs; using
    the between-participant SD instead -- the obvious mistake -- would inflate
    the required N several-fold in a design built specifically to remove
    between-participant variance.
    """
    rng = np.random.default_rng(seed)
    out = {"budgets": list(budgets_seconds), "pairwise": [], "ranking": [],
           "pairwise_sd": [], "ranking_sd": [], "n_pairs": [], "n_sets": [],
           "diff_mean": [], "diff_sd": [], "dz": [], "n_required": []}
    for secs in budgets_seconds:
        n_pairs = max(1, int(secs / SEC_PER_PAIR))
        n_sets = max(1, int(secs / sec_per_rank_set))
        scores = np.array([_one_participant(X, pool, rng, n_pairs, n_sets)
                           for _ in range(trials)])
        diff = scores[:, 1] - scores[:, 0]
        dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0
        out["pairwise"].append(round(float(scores[:, 0].mean()), 4))
        out["ranking"].append(round(float(scores[:, 1].mean()), 4))
        out["pairwise_sd"].append(round(float(scores[:, 0].std()), 4))
        out["ranking_sd"].append(round(float(scores[:, 1].std()), 4))
        out["n_pairs"].append(n_pairs)
        out["n_sets"].append(n_sets)
        out["diff_mean"].append(round(float(diff.mean()), 4))
        out["diff_sd"].append(round(float(diff.std(ddof=1)), 4))
        out["dz"].append(round(dz, 3))
        out["n_required"].append(n_for_dz(dz) if abs(dz) >= DZ_FLOOR else None)
    return out


def sensitivity(X, pool, budget=300, costs=(30.0, 40.0, 60.0, 90.0), trials=TRIALS, seed=1):
    """Does the conclusion survive being wrong about how long ranking takes?

    Nothing in the dataset says what a ten-film ranking costs a person. It is
    the single assumption the comparison turns on, so it is varied rather than
    asserted, and a pilot would replace it with a measured value.
    """
    rows = []
    for cost in costs:
        rng = np.random.default_rng(seed)
        n_pairs = max(1, int(budget / SEC_PER_PAIR))
        n_sets = max(1, int(budget / cost))
        scores = np.array([_one_participant(X, pool, rng, n_pairs, n_sets)
                           for _ in range(trials)])
        diff = scores[:, 1] - scores[:, 0]
        dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0
        rows.append({"sec_per_rank_set": cost, "n_sets": n_sets,
                     "pairwise": round(float(scores[:, 0].mean()), 4),
                     "ranking": round(float(scores[:, 1].mean()), 4),
                     "diff_mean": round(float(diff.mean()), 4),
                     "diff_sd": round(float(diff.std(ddof=1)), 4),
                     "dz": round(dz, 3),
                     "n_required": n_for_dz(dz) if abs(dz) >= DZ_FLOOR else None})
    return {"budget": budget, "rows": rows}


def _wide_matrix():
    """features.build_matrix() plus the genres it deliberately drops.

    A strict superset, so the narrow representation is nested inside the wide
    one and the comparison is bias against variance rather than two unrelated
    models.
    """
    df = data.catalogue()
    X = features.build_matrix()
    extra, names = [], []
    for g in dict.fromkeys(EXTRA_GENRES):
        col = np.array([1.0 if g in s else 0.0 for s in df["genre_list"]])
        if col.std() > 1e-6:
            extra.append(col - col.mean())
            names.append(f"genre_{g}")
    return np.column_stack([X] + extra), names


def _sparse_matrix():
    """The wide matrix plus indicator columns for the most prolific directors.

    This is the regime the separability argument is really about: many columns
    that are almost always zero, so a handful of films carry each one.
    """
    df = data.catalogue()
    X_wide, names = _wide_matrix()
    top = df["director_name"].value_counts().head(N_DIRECTORS).index
    cols, extra = [], []
    for name in top:
        col = (df["director_name"] == name).to_numpy(dtype=float)
        cols.append(col - col.mean())
        extra.append(f"director_{name}")
    return np.column_stack([X_wide] + cols), names + extra


def dimension_study(pool, budget=300, trials=TRIALS, seed=2):
    """Does the small representation actually predict better out of sample?

    The separability argument is a claim about the estimator; this is the claim
    that matters. Synthetic users' true preferences live in the WIDEST space, so
    every smaller representation is genuinely misspecified and pays a real bias
    cost -- then all three are fitted from the same responses and scored on the
    same held-out films. Anything else would rig the comparison.
    """
    X_full, names = _sparse_matrix()
    widths = {"narrow": features.dimension(),
              "wide": X_full.shape[1] - N_DIRECTORS,
              "sparse": X_full.shape[1]}
    n_pairs = max(1, int(budget / SEC_PER_PAIR))
    n_sets = max(1, int(budget / SEC_PER_RANK_SET))

    out = {"widths": widths, "budget": budget, "n_pairs": n_pairs, "n_sets": n_sets,
           "extra_features": names[:len(names) - N_DIRECTORS],
           "n_directors": N_DIRECTORS}
    for label, d in widths.items():
        rng = np.random.default_rng(seed)
        scores = np.array([_one_participant(X_full, pool, rng, n_pairs, n_sets,
                                            X_fit=X_full[:, :d])
                           for _ in range(trials)])
        out[label] = {"d": d,
                      "pairwise": round(float(scores[:, 0].mean()), 4),
                      "ranking": round(float(scores[:, 1].mean()), 4)}
    return out


def separability(dims=(16, 21, 24, 30, 34), ns=(40, 50, 60, 80)):
    """Cover's counting theorem: P(n random-labelled points in R^d separable).

    Where this probability is near 1 the data cannot pin down a direction, and
    what a regularised fit returns in the unidentified subspace is the prior
    rather than the participant.
    """
    from math import comb
    table = {}
    for d in dims:
        table[d] = [round(min(1.0, sum(comb(n - 1, k) for k in range(d)) / 2 ** (n - 1)), 4)
                    for n in ns]
    return {"dims": list(dims), "ns": list(ns), "table": table}


def power_curve(effects=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8), alpha=ALPHA, target=POWER):
    """Sample size for a paired test, by smallest effect of interest."""
    return {"effects": list(effects), "n": [n_for_dz(e, alpha, target) for e in effects],
            "alpha": alpha, "power": target}


def run(seed=0):
    X = features.build_matrix()
    pool = np.asarray(data.study_pool_index(), dtype=int)
    return {"recovery": recovery(X, pool, seed=seed),
            "sensitivity": sensitivity(X, pool),
            "dimension": dimension_study(pool),
            "separability": separability(),
            "power": power_curve(),
            "assumptions": {"sec_per_pair": SEC_PER_PAIR,
                            "sec_per_rank_set": SEC_PER_RANK_SET,
                            "d": X.shape[1], "n_movies": int(X.shape[0]),
                            "n_pool": int(len(pool)),
                            "sigma": preference.SIGMA,
                            "alpha": ALPHA, "power": POWER}}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    metrics = run()
    with open(os.path.join(OUT_DIR, "sim_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["recovery"], indent=2))
