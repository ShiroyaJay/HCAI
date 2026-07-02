"""Project 2 — Explainability.

A single page whose UI state lives entirely in the query string, so every control is
a small auto-submitting form:

    ?model=tree&lam=0.02&cf_index=12&cf_target=Gentoo&feature=bill_length_mm

The chosen model class (``model``) and regularization weight (``lam``) select one
active model; the counterfactual and feature-effect regions both explain *that* model.
"""

import hashlib
import os

from django.conf import settings
from django.shortcuts import render

from . import counterfactuals as cf
from . import data, feature_effects as fe, ml


def _fmt(col, value):
    """Display formatting: round numeric features, leave categoricals as-is."""
    if col in data.NUMERIC_FEATURES:
        return round(float(value), 1)
    return value


def _row_cells(row, feature_order, changed):
    return [
        {"col": c, "value": _fmt(c, row[c]), "changed": c in changed}
        for c in feature_order
    ]


def _ensure_session(request):
    if not request.session.session_key:
        request.session.save()
    return request.session.session_key


def _media_paths(name):
    """(absolute path to write, URL to embed) for a media file."""
    return os.path.join(settings.MEDIA_ROOT, name), settings.MEDIA_URL + name


def _cache_bust(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:8]


def index(request):
    # ----- shared controls -------------------------------------------------- #
    model_type = request.GET.get("model", "tree")
    if model_type not in ("tree", "logreg"):
        model_type = "tree"

    try:
        lam = float(request.GET.get("lam", ml.LAMBDA_MIN))
    except (TypeError, ValueError):
        lam = ml.LAMBDA_MIN
    lam = max(ml.LAMBDA_MIN, min(ml.LAMBDA_MAX, lam))

    feature = request.GET.get("feature")
    if feature not in data.NUMERIC_FEATURES:
        feature = data.NUMERIC_FEATURES[0]

    # ----- data + active model ---------------------------------------------- #
    df = data.load_penguins()
    classes = data.species_classes(df)

    candidates = ml.cached_candidates(model_type)
    selected = ml.select_model(candidates, lam)
    pipe = selected["pipe"]
    omega = selected["omega"]
    acc = selected["acc"]

    session_key = _ensure_session(request)
    omega_label = "leaves" if model_type == "tree" else "non-zero coefficients"

    # ----- model view plot -------------------------------------------------- #
    model_plot = f"project2_model_{session_key}.png"
    abs_path, model_plot_url = _media_paths(model_plot)
    if model_type == "tree":
        ml.plot_tree_png(pipe, pipe.classes_, abs_path)
    else:
        ml.logreg_coef_png(pipe, pipe.classes_, abs_path)
    model_plot_url += f"?v={_cache_bust(model_type, lam, omega, acc)}"

    # ----- Region B: counterfactuals --------------------------------------- #
    n_examples = len(df)
    try:
        cf_index = int(request.GET.get("cf_index", 0))
    except (TypeError, ValueError):
        cf_index = 0
    cf_index = max(0, min(n_examples - 1, cf_index))

    x_row = df.loc[cf_index, data.FEATURES]
    cf_true = df.loc[cf_index, data.TARGET]
    cf_pred = pipe.predict(df.loc[[cf_index], data.FEATURES])[0]

    cf_target = request.GET.get("cf_target")
    if cf_target not in classes:
        # default to a class different from the current prediction
        cf_target = next(c for c in classes if c != cf_pred)

    mads = data.numeric_mads(df)
    counterfactuals = cf.generate_counterfactuals(
        pipe, x_row, cf_target, df, mads, k=3
    )
    cf_rows = [
        {
            "cells": _row_cells(r["row"], data.FEATURES, r["changed"]),
            "distance": round(r["distance"], 2),
            "prob": round(r["prob"], 2),
        }
        for r in counterfactuals
    ]
    cf_original = _row_cells(x_row, data.FEATURES, set())

    # ----- Region C: feature effects (PDP + ALE) --------------------------- #
    ordered_classes = list(pipe.classes_)
    bust = _cache_bust(model_type, lam, feature)

    pdp_grid, pdp_curves = fe.pdp(pipe, df, feature, classes)
    pdp_file = f"project2_pdp_{session_key}.png"
    pdp_abs, pdp_url = _media_paths(pdp_file)
    fe.plot_curves(pdp_grid, pdp_curves, ordered_classes, feature,
                   "P(species)", f"PDP — {feature}", pdp_abs)
    pdp_url += f"?v={bust}"

    ale_method = "exact derivative" if model_type == "logreg" else "discretized"
    ale_grid, ale_curves = fe.ale(pipe, df, feature, classes, model_type)
    ale_file = f"project2_ale_{session_key}.png"
    ale_abs, ale_url = _media_paths(ale_file)
    fe.plot_curves(ale_grid, ale_curves, ordered_classes, feature,
                   "ALE (centered)", f"ALE ({ale_method}) — {feature}", ale_abs)
    ale_url += f"?v={bust}"

    context = {
        "model_type": model_type,
        "lam": round(lam, 3),
        "lambda_min": ml.LAMBDA_MIN,
        "lambda_max": ml.LAMBDA_MAX,
        "lambda_step": ml.LAMBDA_STEP,
        "omega": omega,
        "omega_label": omega_label,
        "acc": round(acc, 3),
        "objective": round(acc - lam * omega, 3),
        "model_plot_url": model_plot_url,
        "classes": classes,
        # counterfactual region
        "feature_names": data.FEATURES,
        "n_examples": n_examples,
        "cf_index": cf_index,
        "cf_target": cf_target,
        "cf_true": cf_true,
        "cf_pred": cf_pred,
        "cf_target_is_pred": cf_target == cf_pred,
        "cf_original": cf_original,
        "cf_rows": cf_rows,
        # feature-effect region
        "numeric_features": data.NUMERIC_FEATURES,
        "feature": feature,
        "pdp_url": pdp_url,
        "ale_url": ale_url,
        "ale_method": ale_method,
    }
    return render(request, "project2/index.html", context)
