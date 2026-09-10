"""Task 2 -- Plackett-Luce preference model, MAP estimation, recommendations.

Bradley-Terry gives P(i > j) = sigmoid(w'(x_i - x_j)). The extension to a full
ranking i_1 > i_2 > ... > i_n is Plackett-Luce:

    P(i_1 > ... > i_n | w) = prod_{k=1}^{n-1}  exp(w'x_{i_k})
                                              -------------------------
                                              sum_{l=k}^{n} exp(w'x_{i_l})

i.e. "repeatedly pick the best of what is left". Properties that justify it:
  - it is the unique model satisfying Luce's choice axiom (IIA);
  - it reduces EXACTLY to Bradley-Terry at n = 2;
  - it is the random-utility model with i.i.d. Gumbel noise (McFadden). Note
    this is Gumbel, not Thurstonian: Thurstone Case V uses Gaussian noise and
    has no closed form for n > 2, which is a further argument for Plackett-Luce;
  - its log-likelihood is concave, and strictly concave with a Gaussian prior,
    so the MAP estimate is unique even when the data are linearly separable.

Truncating the product at k = K gives the top-K partial ranking likelihood,
which is what a rank task cut off by the time budget produces -- so partial
responses are used rather than discarded.

The SAME estimator serves both study designs: a pairwise choice is just a
ranking with n = 2. Fitting the two designs with two different optimisers would
confound the headline comparison.
"""

import numpy as np

from . import features

# Prior standard deviation on w. Pre-registered and identical across both
# designs; the report gives a sensitivity analysis over {0.5, 1, 2}.
SIGMA = 1.0


def ranking(items, n_ranked=None):
    """One observation: `items` in preference order (best first).

    n_ranked = how many positions are actually determined (top-K); defaults to
    a complete ranking.
    """
    items = [int(i) for i in items]
    full = len(items) - 1
    return (items, full if n_ranked is None else min(int(n_ranked), full))


def _terms(groups, X):
    """Feature rows for each choice-set in each observation."""
    for items, n_ranked in groups:
        rows = X[items]
        for k in range(n_ranked):
            yield rows[k:]


def neg_log_posterior(w, groups, X, sigma=SIGMA):
    total = 0.5 * (w @ w) / sigma ** 2
    for Z in _terms(groups, X):
        u = Z @ w
        m = u.max()
        total += -(u[0] - m) + np.log(np.exp(u - m).sum())
    return total


def _grad_hess(w, groups, X, sigma=SIGMA):
    d = len(w)
    grad = w / sigma ** 2
    hess = np.eye(d) / sigma ** 2
    for Z in _terms(groups, X):
        u = Z @ w
        p = np.exp(u - u.max())
        p /= p.sum()
        mu = p @ Z
        grad -= Z[0] - mu
        Zc = Z - mu
        hess += (Zc * p[:, None]).T @ Zc
    return grad, hess


def fit_map(groups, X=None, sigma=SIGMA, iters=50, tol=1e-9):
    """MAP estimate of w by damped Newton.

    The negative log-posterior is strictly convex with Hessian >= I/sigma^2, so
    Newton converges in well under 10 iterations and needs no line-search
    heroics -- a simple backtracking guard is enough.
    """
    X = features.build_matrix() if X is None else X
    w = np.zeros(X.shape[1])
    if not groups:
        return w
    fval = neg_log_posterior(w, groups, X, sigma)
    for _ in range(iters):
        grad, hess = _grad_hess(w, groups, X, sigma)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        for _ in range(20):                      # backtracking
            cand = w - t * step
            fcand = neg_log_posterior(cand, groups, X, sigma)
            if fcand <= fval:
                break
            t *= 0.5
        if np.linalg.norm(t * step) < tol:
            w, fval = cand, fcand
            break
        w, fval = cand, fcand
    return w


def laplace_cov(w, groups, X=None, sigma=SIGMA):
    """Posterior covariance under the Laplace approximation, (-H)^-1.

    Used for the uncertainty bars on the taste profile and for the adaptive
    selection rule -- one object, two uses.
    """
    X = features.build_matrix() if X is None else X
    _, hess = _grad_hess(w, groups, X, sigma)
    return np.linalg.inv(hess)


def utilities(w, X=None):
    X = features.build_matrix() if X is None else X
    return X @ w


def prob_prefer(w, i, j, X=None):
    """Bradley-Terry probability that movie i is preferred to movie j."""
    X = features.build_matrix() if X is None else X
    return float(1.0 / (1.0 + np.exp(-(X[i] - X[j]) @ w)))


def rank_movies(w, candidates, X=None, best_first=True):
    """`candidates` (catalogue row positions) sorted by estimated utility."""
    X = features.build_matrix() if X is None else X
    cand = np.asarray(list(candidates), dtype=int)
    u = X[cand] @ w
    order = np.argsort(-u if best_first else u)
    return cand[order]


def recommend(w, candidates, n=5, exclude=(), diversify=True, X=None):
    """Top-n recommendations, excluding seen movies and genre-deduplicated.

    A linear utility maximised over a fixed catalogue lands on feature extremes,
    so without diversification every participant sees a near-identical list.
    """
    from . import data
    excluded = {int(i) for i in exclude}
    ordered = [i for i in rank_movies(w, candidates, X) if int(i) not in excluded]
    if not diversify:
        return [int(i) for i in ordered[:n]]

    df = data.catalogue()
    picked, used_primary = [], set()
    for i in ordered:
        primary = df.iloc[int(i)]["genre_list"][0]
        if primary in used_primary:
            continue
        used_primary.add(primary)
        picked.append(int(i))
        if len(picked) == n:
            return picked
    for i in ordered:                      # top up if genres ran out
        if int(i) not in picked:
            picked.append(int(i))
        if len(picked) == n:
            break
    return picked


def weight_table(w, order_by=None, cov=None):
    """Every weight in plain language, strongest first: {name, label, weight, se}.

    `order_by` decides the ORDER while `w` supplies the values. The reveal page
    needs those separated: the sliders are laid out by what the model inferred,
    but show what the user has since set, so that dragging one control does not
    reshuffle the panel underneath the cursor.
    """
    ref = w if order_by is None else order_by
    se = np.sqrt(np.diag(cov)) if cov is not None else np.zeros_like(w)
    items = [{"name": name,
              "label": features.LABELS[name],
              "weight": float(w[k]),
              "se": float(se[k]),
              "_key": abs(float(ref[k]))}
             for k, name in enumerate(features.FEATURE_NAMES)]
    items.sort(key=lambda it: -it["_key"])
    for it in items:
        del it["_key"]
    return items


def explain(w, cov=None, top=4):
    """w rendered in plain language, most confident weights first.

    Returns (likes, dislikes) lists of {label, weight, se}.
    """
    ranked = weight_table(w, cov=cov)
    likes = [it for it in ranked if it["weight"] > 0][:top]
    dislikes = [it for it in ranked if it["weight"] < 0][:top]
    return likes, dislikes
