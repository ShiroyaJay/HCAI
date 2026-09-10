"""Task 1: the movie feature representation.

Design constraint that drives everything here: the whole point of the study is
to estimate w from ~20-60 interactions. For random +/-1 labels on n difference
vectors in R^d, Cover's counting theorem gives
    P(linearly separable) = 2^-(n-1) * sum_{k<d} C(n-1, k)
which at d=34, n=50 is 0.995. The likelihood is then maximised at infinity and the
"estimated preference vector" would be a pure regularisation artifact. At d=21
it is ~0.1. So the representation is deliberately small and every dimension is
one a person could actually name as a taste.

Two identifiability rules are enforced rather than assumed:
  - genres stay raw 0/1, never row-normalised. Row-normalising makes the
    all-ones genre direction constant across movies, which is exactly
    unidentified: only utility *differences* within a choice set enter the
    Plackett-Luce likelihood, so a constant direction never appears in the
    data, yet its Laplace posterior variance is maximal, so an adaptive selector
    would then chase a provably unlearnable direction.
  - the content-rating one-hot drops a reference level, for the same reason.

Rare binary flags are centered but not divided by their standard deviation:
scaling a 4%-prevalence flag by sd ~ 0.2 turns a handful of films into
high-leverage points that dominate the fit.
"""

from functools import lru_cache

import numpy as np

from . import data

# Top genres by frequency; all appear on >= 460 films. Rare ones (Film-Noir has
# 6) would be pure noise at this sample size.
GENRES = ["Drama", "Comedy", "Thriller", "Action", "Romance", "Adventure",
          "Crime", "Sci-Fi", "Fantasy", "Horror", "Family", "Mystery"]

# Reference level is "adult" (R / NC-17 / X); the other three are explicit.
RATING_GROUPS = ["family", "teen", "unrated"]

FEATURE_NAMES = (
    [f"genre_{g}" for g in GENRES]
    + ["year", "duration", "rating_family", "rating_teen", "rating_unrated",
       "imdb_score", "popularity", "non_english", "black_and_white"]
)

# Plain-language labels for the reveal page, phrased so that a positive weight
# reads as "likes this".
LABELS = {
    **{f"genre_{g}": g.lower() + " films" for g in GENRES},
    "year": "newer films",
    "duration": "longer films",
    "rating_family": "family-friendly (G/PG) films",
    "rating_teen": "PG-13 films",
    "rating_unrated": "unrated films",
    "imdb_score": "critically acclaimed films",
    "popularity": "popular, mainstream films",
    "non_english": "non-English-language films",
    "black_and_white": "black-and-white films",
}


def _rating_group(value):
    if not isinstance(value, str):
        return "unrated"
    v = value.strip()
    if v in ("G", "PG", "TV-G", "TV-PG", "TV-Y", "TV-Y7", "Approved"):
        return "family"
    if v in ("PG-13", "TV-14"):
        return "teen"
    if v in ("R", "NC-17", "X", "M", "GP", "TV-MA"):
        return "adult"
    return "unrated"


@lru_cache(maxsize=1)
def build_matrix():
    """The (n_movies, d) feature matrix for the whole catalogue.

    This is the extractor Task 1 asks for. The standardisation is fitted once,
    here, on the full catalogue and frozen, so a given component of w means
    the same thing for every participant, which is what makes cross-participant
    analysis (and the population baseline) meaningful.
    """
    df = data.catalogue()
    n = len(df)
    cols = []

    genre_sets = [set(gs) for gs in df["genre_list"]]
    for g in GENRES:
        cols.append(np.array([1.0 if g in s else 0.0 for s in genre_sets]))

    # Continuous features: winsorise then standardise.
    def std(values, lo=1, hi=99):
        v = np.asarray(values, dtype=float)
        v = np.clip(v, np.percentile(v, lo), np.percentile(v, hi))
        return (v - v.mean()) / v.std()

    cols.append(std(df["title_year"]))
    cols.append(std(df["duration"]))

    groups = df["content_rating"].map(_rating_group)
    for g in RATING_GROUPS:
        cols.append((groups == g).to_numpy(dtype=float))

    cols.append(std(df["imdb_score"]))
    cols.append(std(np.log10(df["num_voted_users"].to_numpy(dtype=float) + 1)))

    language = df["language"].astype(str).str.strip()
    cols.append((language != "English").to_numpy(dtype=float))
    cols.append((df["color"] == "Black and White").to_numpy(dtype=float))

    X = np.column_stack(cols)
    # Center every column so weights are comparable and no column is constant.
    # (Centering cancels inside the likelihood, since only differences matter, but
    # it makes the rank assertion below meaningful and w readable.)
    X = X - X.mean(axis=0)

    assert X.shape == (n, len(FEATURE_NAMES)), (X.shape, len(FEATURE_NAMES))
    assert np.isfinite(X).all()
    assert X.std(axis=0).min() > 1e-6, "constant feature column"
    rank = np.linalg.matrix_rank(X)
    assert rank == X.shape[1], f"rank deficient: {rank} < {X.shape[1]}"
    return X


def dimension():
    return len(FEATURE_NAMES)
