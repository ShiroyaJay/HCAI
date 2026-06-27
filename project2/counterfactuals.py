"""Counterfactual explanations by local random sampling (brief, section 2).

For a chosen example ``x`` and a desired class:

  1. Sample N points locally around ``x``. Numeric features get Gaussian noise scaled
     by the feature's MAD; categorical/binary features are occasionally resampled to a
     different category (the brief's "how do you noise categorical data?" requirement).
  2. Keep the samples the model now predicts as the desired class.
  3. Rank them by MAD-weighted L1 distance to ``x`` (numeric: |Δ|/MAD; categorical:
     0/1 mismatch) and return the best k.

If too few are found, N and the noise level grow and we resample (iterative fallback).
"""

import numpy as np
import pandas as pd

from . import data


def _sample_around(x_row, category_options, mads, num_bounds, n, num_scale, p_flip, rng):
    """N samples near ``x_row`` in the original feature space.

    Numeric draws are clipped to each feature's observed [min, max] so counterfactuals
    stay physically plausible (no negative bill lengths / body masses).
    """
    cols_data = {}
    for col in data.NUMERIC_FEATURES:
        sample = x_row[col] + rng.normal(0.0, num_scale * mads[col], size=n)
        lo, hi = num_bounds[col]
        cols_data[col] = np.clip(sample, lo, hi)
    for col in data.CATEGORICAL_FEATURES:
        values = np.full(n, x_row[col], dtype=object)
        flip = rng.random(n) < p_flip
        values[flip] = rng.choice(category_options[col], size=int(flip.sum()))
        cols_data[col] = values
    return pd.DataFrame(cols_data)[data.FEATURES]


def _mad_weighted_l1(x_row, cand, mads):
    dist = np.zeros(len(cand))
    for col in data.NUMERIC_FEATURES:
        dist += np.abs(cand[col].values - x_row[col]) / mads[col]
    for col in data.CATEGORICAL_FEATURES:
        dist += (cand[col].values != x_row[col]).astype(float)
    return dist


def _changed_features(row, x_row, mads):
    """Features that changed *meaningfully* (numeric: > 0.1·MAD), for highlighting."""
    changed = []
    for col in data.NUMERIC_FEATURES:
        if abs(float(row[col]) - float(x_row[col])) > 0.1 * mads[col]:
            changed.append(col)
    for col in data.CATEGORICAL_FEATURES:
        if row[col] != x_row[col]:
            changed.append(col)
    return changed


def generate_counterfactuals(pipe, x_row, target, df, mads, k=3, n=2000, rounds=6):
    """Return up to ``k`` counterfactuals (each a dict) ranked by distance to ``x``."""
    rng = np.random.default_rng(0)
    category_options = {c: df[c].unique() for c in data.CATEGORICAL_FEATURES}
    num_bounds = {c: (df[c].min(), df[c].max()) for c in data.NUMERIC_FEATURES}
    classes = list(pipe.classes_)
    target_idx = classes.index(target)

    num_scale, p_flip = 1.0, 0.15
    found = []  # (distance, row Series, target probability)

    for _ in range(rounds):
        cand = _sample_around(
            x_row, category_options, mads, num_bounds, n, num_scale, p_flip, rng
        )
        proba = pipe.predict_proba(cand)
        mask = proba.argmax(axis=1) == target_idx
        if mask.any():
            sub = cand[mask].reset_index(drop=True)
            dist = _mad_weighted_l1(x_row, sub, mads)
            probs = proba[mask, target_idx]
            for i in range(len(sub)):
                found.append((dist[i], sub.iloc[i], probs[i]))
        if len(found) >= 5 * k:
            break
        # iterative fallback: widen the search
        num_scale *= 1.6
        p_flip = min(0.6, p_flip * 1.5)
        n = int(n * 1.6)

    found.sort(key=lambda t: t[0])

    results, seen = [], set()
    for dist, row, prob in found:
        key = tuple(
            round(float(row[c]), 2) if c in data.NUMERIC_FEATURES else row[c]
            for c in data.FEATURES
        )
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "row": row,
            "changed": set(_changed_features(row, x_row, mads)),
            "distance": float(dist),
            "prob": float(prob),
        })
        if len(results) >= k:
            break
    return results
