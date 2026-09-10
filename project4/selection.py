"""Question selection: uniform random (the study protocol) and an adaptive rule.

The brief specifies uniform-random selection for the study itself, which keeps
the two interfaces comparable. Adaptive selection is named in the brief as an
interesting extension; it lives here and is shown on the demo page, outside the
study path, so it cannot contaminate the comparison.

The criterion matters. The plain D-optimal choice, maximising d'(Sigma)d for
d = x_a - x_b, picks the pairs the user finds *easiest*, because a large
|w'd| means a near-certain answer, which carries almost no information. The
expected-information form weights posterior uncertainty by response entropy:

    score(a, b) = log(1 + p(1-p) * d' Sigma d),    p = sigmoid(w'd)

so a good question is one that is both uncertain under the posterior and close
to a coin flip for this user. Candidates are subsampled: the catalogue admits
~11 million pairs, which cannot be enumerated inside a request.
"""

import numpy as np

from . import preference

CANDIDATES = 200


def random_pair(pool, rng):
    return [int(i) for i in rng.sample(list(pool), 2)]


def random_set(pool, rng, k=10):
    return [int(i) for i in rng.sample(list(pool), k)]


def score_pairs(w, cov, X, pairs):
    idx = np.asarray(pairs, dtype=int)
    D = X[idx[:, 0]] - X[idx[:, 1]]
    p = 1.0 / (1.0 + np.exp(-(D @ w)))
    var = np.einsum("ij,jk,ik->i", D, cov, D)
    return np.log1p(p * (1.0 - p) * var)


def adaptive_pair(w, cov, X, pool, rng, candidates=CANDIDATES):
    """The most informative pair among a random subsample of candidates."""
    pool = list(pool)
    pairs = [rng.sample(pool, 2) for _ in range(candidates)]
    return [int(i) for i in pairs[int(np.argmax(score_pairs(w, cov, X, pairs)))]]


def simulate_trace(X, pool, mode="random", n=20, seed=0):
    """One simulated participant, for the demo page.

    Reports how fast the posterior tightens under each selection rule, using a
    synthetic user whose true preferences are known.
    """
    import random as _random
    rng = _random.Random(seed)
    nprng = np.random.default_rng(seed)
    w_true = nprng.normal(size=X.shape[1]) * 0.8

    obs, trace = [], []
    w = np.zeros(X.shape[1])
    cov = np.eye(X.shape[1]) * preference.SIGMA ** 2
    for t in range(n):
        if mode == "adaptive" and obs:
            pair = adaptive_pair(w, cov, X, pool, rng)
        else:
            pair = random_pair(pool, rng)
        p = 1.0 / (1.0 + np.exp(-(X[pair[0]] - X[pair[1]]) @ w_true))
        winner, loser = (pair if nprng.random() < p else pair[::-1])
        obs.append(preference.ranking([winner, loser]))
        w = preference.fit_map(obs, X)
        cov = preference.laplace_cov(w, obs, X)
        trace.append({
            "n": t + 1,
            "uncertainty": round(float(np.sqrt(np.trace(cov))), 3),
            "cosine": round(float(w @ w_true /
                                  (np.linalg.norm(w) * np.linalg.norm(w_true) + 1e-12)), 3),
        })
    return trace
