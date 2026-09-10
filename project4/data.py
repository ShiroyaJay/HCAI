"""IMDB 5000 Movie Dataset: loading, cleaning and the study sampling pool.

The CSV is vendored in project4/data/ and committed, so a fresh clone works
offline and the app deploys to a read-only filesystem without a build step.

The raw file has several well-known defects that every consumer must not have
to think about, so they are fixed exactly once, here:
  - `movie_title` carries a trailing non-breaking space on 4935 of 5043 rows
  - 45 exactly duplicated rows, plus re-releases sharing a title
  - `color` values are 'Color' and ' Black and White' (leading space)
  - `budget` is in unlabelled local currency and `gross` is US-domestic only,
    so neither is usable (both are excluded from the features entirely)
"""

import os
import re
from functools import lru_cache

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "movie_metadata.csv")

# Genres that are not feature films; they distort the genre features and the
# brief asks for a movie recommender.
NON_FILM_GENRES = {"Short", "News", "Reality-TV", "Game-Show"}

# Columns a movie must have to be usable at all.
REQUIRED = ["movie_title", "genres", "title_year", "duration", "imdb_score",
            "num_voted_users", "cast_total_facebook_likes"]

# Minimum IMDB vote count for a movie to be shown to a participant. Uniform
# sampling over the whole catalogue surfaces films almost nobody has heard of,
# which turns "which would you rather watch" into "which title sounds better".
FAMILIARITY_MIN_VOTES = 10_000

# Recommendations are drawn from a wider frame than elicitation: there is no
# point in a recommender that can only suggest films the user was already shown.
# But it cannot be the whole catalogue either. Utility is linear, so maximising
# it over an unfiltered catalogue lands on whichever corner of feature space is
# most extreme -- and because popularity is itself a feature, a user with any
# taste for the niche is handed films with eight IMDB votes. This threshold
# removes that tail (343 films) while still leaving ~950 titles the study never
# showed them.
RECOMMEND_MIN_VOTES = 1_000


def _imdb_id(link):
    """The stable tt-id from a movie_imdb_link, used as the dedup key."""
    if not isinstance(link, str):
        return None
    m = re.search(r"(tt\d+)", link)
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def catalogue():
    """The cleaned movie catalogue, one row per film. Cached per process."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    df["movie_title"] = (df["movie_title"].astype(str)
                         .str.replace("\xa0", "", regex=False).str.strip())
    df["color"] = df["color"].astype(str).str.strip()
    df["imdb_id"] = df["movie_imdb_link"].map(_imdb_id)

    df = df.drop_duplicates()
    df = df.dropna(subset=REQUIRED)
    df = df.drop_duplicates(subset=["imdb_id"])
    df = df.drop_duplicates(subset=["movie_title", "title_year"])

    genres = df["genres"].str.split("|")
    df = df[~genres.map(lambda gs: bool(set(gs) & NON_FILM_GENRES))]

    df["title_year"] = df["title_year"].astype(int)
    df["genre_list"] = df["genres"].str.split("|")
    df = df.reset_index(drop=True)

    # Integrity assertions: these are the defects that silently corrupt
    # downstream features or the UI, so fail loudly instead.
    assert not df["movie_title"].str.contains("\xa0").any()
    assert not df["imdb_id"].duplicated().any()
    assert df["color"].isin(["Color", "Black and White", "nan"]).all()
    assert len(df) > 4000, f"unexpectedly few usable movies: {len(df)}"
    return df


@lru_cache(maxsize=1)
def study_pool_index():
    """Row positions in catalogue() eligible to be shown to a participant.

    A documented sampling frame, not the full catalogue -- see
    FAMILIARITY_MIN_VOTES. Recommendations use catalogue_index() instead.
    """
    df = catalogue()
    return df.index[df["num_voted_users"] >= FAMILIARITY_MIN_VOTES].to_numpy()


@lru_cache(maxsize=1)
def recommend_pool_index():
    """Row positions eligible to be RECOMMENDED -- see RECOMMEND_MIN_VOTES.

    Wider than study_pool_index(), because a recommender that can only propose
    films already shown during elicitation is not recommending anything.
    """
    df = catalogue()
    return df.index[df["num_voted_users"] >= RECOMMEND_MIN_VOTES].to_numpy()


def movies(indices):
    """Display records for the given catalogue row positions, order preserved."""
    df = catalogue()
    out = []
    for i in indices:
        row = df.iloc[int(i)]
        out.append({
            "index": int(i),
            "title": row["movie_title"],
            "year": int(row["title_year"]),
            "genres": ", ".join(row["genre_list"][:3]),
            "duration": int(row["duration"]),
            "score": float(row["imdb_score"]),
            "rating": row["content_rating"] if isinstance(row["content_rating"], str) else "Unrated",
            "director": row["director_name"] if isinstance(row["director_name"], str) else "",
        })
    return out
