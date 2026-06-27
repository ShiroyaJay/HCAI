"""The machine-learning engine for Project 1.

Kept separate from the views so the web layer stays simple and this part
can be read/graded on its own. Nothing here talks to the user directly —
the views translate these results into plain language.
"""

import matplotlib
matplotlib.use("Agg")  # no screen on a server — draw straight to a file
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


def detect_problem_type(target):
    """Is the thing to guess a CATEGORY or a NUMBER?

    Returns 'category' (a classification problem, e.g. flower species) or
    'number' (a regression problem, e.g. a price). Plain words on purpose —
    the user never sees the textbook terms.
    """
    if not is_numeric_dtype(target):
        return "category"
    # Numbers with only a handful of distinct values are really labels
    # (e.g. species encoded as 0/1/2), so treat those as categories too.
    if target.nunique() <= 10:
        return "category"
    return "number"


def numeric_feature_columns(df):
    """The clue columns (everything but the last) that hold numbers."""
    feature_cols = df.columns[:-1]
    return [c for c in feature_cols if is_numeric_dtype(df[c])]


def make_picture(df, save_path):
    """Draw a friendly scatter plot of the data and save it.

    Returns (ok, problem_type, caption):
      ok          – True if a picture was drawn, False if we couldn't.
      problem_type – 'category' or 'number'.
      caption      – a plain-language sentence describing the picture.
    """
    target_name = df.columns[-1]
    target = df[target_name]
    problem = detect_problem_type(target)
    features = numeric_feature_columns(df)

    if not features:
        return False, problem, "We couldn't draw a picture for this information."

    fig, ax = plt.subplots(figsize=(7, 5))

    if problem == "category" and len(features) >= 2:
        # Two clues on the axes, colour shows which group each dot belongs to.
        x, y = features[0], features[1]
        codes = target.astype("category").cat.codes
        groups = list(target.astype("category").cat.categories)
        cmap = plt.cm.viridis
        ax.scatter(df[x], df[y], c=codes, cmap=cmap, alpha=0.85,
                   edgecolor="white", s=70)
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        n = max(len(groups) - 1, 1)
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=10,
                              color=cmap(i / n)) for i in range(len(groups))]
        ax.legend(handles, [str(g) for g in groups], title=target_name)
        caption = (f"Each dot is one example. We placed them by “{x}” and "
                   f"“{y}”, and each colour is a different “{target_name}”.")
    else:
        # One clue against the thing we want to guess.
        x = features[0]
        ax.scatter(df[x], target, alpha=0.85, color="#1f6feb",
                   edgecolor="white", s=70)
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(target_name, fontsize=12)
        caption = (f"Each dot is one example, showing how “{x}” relates to "
                   f"“{target_name}”.")

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return True, problem, caption


def make_distribution_picture(df, save_path):
    """Draw how the thing-to-guess is spread out, and save it.

    For a group/category target: a bar chart of how many examples of each group.
    For a number target: a histogram of the values. Returns (ok, caption).
    """
    target_name = df.columns[-1]
    target = df[target_name]
    problem = detect_problem_type(target)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if problem == "category":
        counts = target.astype("category").value_counts().sort_index()
        ax.bar([str(i) for i in counts.index], counts.values,
               color="#1f6feb", edgecolor="white")
        ax.set_ylabel("How many examples")
        caption = (f"How many examples there are of each “{target_name}”. "
                   f"A nice even mix usually makes guessing easier.")
    else:
        bins = min(20, max(5, int(len(target) ** 0.5)))
        ax.hist(target.dropna(), bins=bins, color="#1f6feb", edgecolor="white")
        ax.set_xlabel(target_name)
        ax.set_ylabel("How many examples")
        caption = f"How the values of “{target_name}” are spread out."

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return True, caption


def draw_importances(importances, save_path):
    """Draw a simple bar chart of which clues mattered most, and save it."""
    names = [i["name"] for i in importances][::-1]   # top one at the top
    weights = [i["weight"] for i in importances][::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.55 * len(names) + 1.2)))
    ax.barh(names, weights, color="#1f6feb", edgecolor="white")
    ax.set_xlabel("How much it mattered")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def _quality(score):
    """Turn a 0–1 score into a plain word + a warm sentence."""
    if score >= 0.9:
        return "great", "That's really good! 🎉"
    if score >= 0.75:
        return "good", "That's pretty good!"
    if score >= 0.5:
        return "okay", "That's okay — not bad at all."
    return "tricky", ("It found this one tricky — that's alright, "
                      "some things are just hard to guess.")


def _round_nicely(value):
    """Round a number to something a person would actually say."""
    value = float(value)
    if value >= 100:
        return f"{round(value):,}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _best_over_methods(methods, X_train, y_train, X_test, y_test, scorer):
    """Quietly try every method and every setting; keep the best on the test part.

    Each method that can't run on this data (e.g. nearest-neighbours when there
    are too few examples) is simply skipped, so the app never crashes.

    Returns (best, compared):
      best     – (score, method_name, setting, fitted_model) for the overall winner.
      compared – each method's own best score as a percent, for the details panel.
    """
    best = None
    compared = []
    for name, variants in methods:
        method_best = None
        for setting, model in variants:
            try:
                model.fit(X_train, y_train)
                score = scorer(y_test, model.predict(X_test))
            except Exception:
                continue
            if method_best is None or score > method_best[0]:
                method_best = (score, setting, model)
        if method_best is None:
            continue
        compared.append({"name": name, "setting": method_best[1],
                         "percent": round(max(method_best[0], 0.0) * 100)})
        if best is None or method_best[0] > best[0]:
            best = (method_best[0], name, method_best[1], method_best[2])
    return best, compared


def _classification_methods(n_train):
    """Category-guessing methods we try, each across several settings."""
    ks = [k for k in (1, 3, 5, 7, 9) if k < n_train]
    methods = [
        ("Decision tree", [
            (f"max depth = {d}", DecisionTreeClassifier(max_depth=d, random_state=0))
            for d in range(1, 11)]),
        ("Logistic guess", [
            (f"strength = {c}", make_pipeline(
                StandardScaler(), LogisticRegression(C=c, max_iter=1000)))
            for c in (0.1, 1.0, 10.0)]),
    ]
    if ks:
        methods.append(("Nearest neighbours", [
            (f"{k} closest", make_pipeline(
                StandardScaler(), KNeighborsClassifier(n_neighbors=k)))
            for k in ks]))
    return methods


def _regression_methods(n_train):
    """Number-guessing methods we try, each across several settings."""
    ks = [k for k in (1, 3, 5, 7, 9) if k < n_train]
    methods = [
        ("Decision tree", [
            (f"max depth = {d}", DecisionTreeRegressor(max_depth=d, random_state=0))
            for d in range(1, 11)]),
        ("Straight-line fit", [
            ("simple line", make_pipeline(StandardScaler(), LinearRegression()))]),
    ]
    if ks:
        methods.append(("Nearest neighbours", [
            (f"{k} closest", make_pipeline(
                StandardScaler(), KNeighborsRegressor(n_neighbors=k)))
            for k in ks]))
    return methods


def _explain_methods(problem):
    """A single, deliberately shallow decision tree — one a person can follow.

    Used when the user asks the computer to *explain* its guesses: a shallow
    tree stays readable, and we can name the clues it leaned on the most.
    """
    Tree = DecisionTreeClassifier if problem == "category" else DecisionTreeRegressor
    return [("Decision tree", [
        (f"max depth = {d}", Tree(max_depth=d, random_state=0))
        for d in range(1, 7)])]


def _explain_tree(model, feature_cols, target_name):
    """Turn a fitted decision tree into one plain sentence about *why*."""
    importances = list(getattr(model, "feature_importances_", []))
    ranked = sorted(zip(feature_cols, importances), key=lambda p: p[1], reverse=True)
    top = [name for name, weight in ranked if weight > 0][:2]
    if not top:
        return f"It looked at the information you gave to guess “{target_name}”."
    if len(top) == 1:
        return f"To guess “{target_name}”, it mostly paid attention to “{top[0]}”."
    return (f"To guess “{target_name}”, it mostly paid attention to “{top[0]}”, "
            f"and then to “{top[1]}”.")


def _tree_importances(model, feature_cols):
    """The clues a fitted tree leaned on, ranked, for a bar chart."""
    ranked = sorted(zip(feature_cols, getattr(model, "feature_importances_", [])),
                    key=lambda p: p[1], reverse=True)
    return [{"name": str(n), "weight": float(w)} for n, w in ranked if w > 0]


def teach_computer(df, target_name, problem_override=None, prefer="accurate"):
    """Train end-to-end and return plain-language results.

    Does the whole textbook pipeline automatically — split the data, try
    several methods each over several settings, keep the best, score it —
    then translates the score into everyday words.

    Two optional human choices (both have sensible defaults, so a user who
    just clicks the button never has to touch them):
      problem_override – 'category' or 'number' to overrule the auto-detection,
                         or None to let the app decide.
      prefer           – 'accurate' (try several methods, keep the best) or
                         'explain' (use one readable tree and say *why*).
    """
    target = df[target_name]
    problem = problem_override or detect_problem_type(target)
    # A number to guess only makes sense if the answers are actually numbers.
    if problem == "number" and not is_numeric_dtype(target):
        problem = "category"
    feature_cols = [c for c in df.columns
                    if c != target_name and is_numeric_dtype(df[c])]

    if not feature_cols:
        return {"ok": False,
                "reason": ("We need at least one column of numbers for the "
                           "computer to learn from. Try a different file, or "
                           "use our example.")}

    if len(df) < 5:
        return {"ok": False,
                "reason": ("We need at least 5 examples to learn from. Please "
                           "add a few more rows, or use our example.")}

    X = df[feature_cols]

    if problem == "category":
        y = target.astype("category").cat.codes
        # Keep the class mix balanced across the split when we safely can.
        stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=stratify)

        # Always compare the full set of methods, so the details panel can show
        # every algorithm and the accuracy it reached.
        best_all, compared = _best_over_methods(
            _classification_methods(len(X_train)),
            X_train, y_train, X_test, y_test, accuracy_score)
        if prefer == "explain":
            # The user asked for a readable model, so we keep the best shallow
            # tree even if another method scored higher (accuracy vs. clarity).
            best, _ = _best_over_methods(
                _explain_methods("category"),
                X_train, y_train, X_test, y_test, accuracy_score)
        else:
            best = best_all
        best_score, model_name, setting, best_model = best
        explanation = (_explain_tree(best_model, feature_cols, target_name)
                       if prefer == "explain" else None)
        importances = (_tree_importances(best_model, feature_cols)
                       if prefer == "explain" else None)

        # A few real guesses on the hidden test part, so the user can SEE it work.
        categories = list(target.astype("category").cat.categories)
        samples = []
        for actual_code, pred_code in list(zip(list(y_test), best_model.predict(X_test)))[:4]:
            a, p = int(actual_code), int(pred_code)
            samples.append({
                "guess": str(categories[p]) if 0 <= p < len(categories) else str(p),
                "actual": str(categories[a]) if 0 <= a < len(categories) else str(a),
                "correct": p == a,
            })

        quality, quality_line = _quality(best_score)
        n_correct = round(best_score * len(y_test))
        return {
            "ok": True,
            "problem": "category",
            "headline": (f"It guessed right {n_correct} out of {len(y_test)} times "
                         f"— that's {round(best_score * 100)}%."),
            "detail": f"Tested on {len(y_test)} examples it had never seen before.",
            "quality": quality,
            "quality_line": quality_line,
            "percent": round(best_score * 100),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "model_name": model_name,
            "setting": setting,
            "compared": compared,
            "explanation": explanation,
            "importances": importances,
            "samples": samples,
        }

    # A number to guess → regression.
    y = target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Always compare the full set of methods, so the details panel can show
    # every algorithm and the score it reached.
    best_all, compared = _best_over_methods(
        _regression_methods(len(X_train)),
        X_train, y_train, X_test, y_test, r2_score)
    if prefer == "explain":
        best, _ = _best_over_methods(
            _explain_methods("number"),
            X_train, y_train, X_test, y_test, r2_score)
    else:
        best = best_all
    best_score, model_name, setting, best_model = best
    test_preds = best_model.predict(X_test)
    best_mae = mean_absolute_error(y_test, test_preds)
    explanation = (_explain_tree(best_model, feature_cols, target_name)
                   if prefer == "explain" else None)
    importances = (_tree_importances(best_model, feature_cols)
                   if prefer == "explain" else None)

    # A few real guesses on the hidden test part, with how far off each was.
    samples = []
    for actual_v, pred_v in list(zip(list(y_test), test_preds))[:4]:
        samples.append({
            "guess": _round_nicely(pred_v),
            "actual": _round_nicely(actual_v),
            "off": _round_nicely(abs(float(pred_v) - float(actual_v))),
        })

    closeness = max(best_score, 0.0)  # r² can go negative; floor at 0 for words
    quality, quality_line = _quality(closeness)
    return {
        "ok": True,
        "problem": "number",
        "headline": (f"On average, the computer's guess was off by about "
                     f"{_round_nicely(best_mae)}."),
        "detail": f"Tested on {len(y_test)} examples it had never seen before.",
        "quality": quality,
        "quality_line": quality_line,
        "percent": round(closeness * 100),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "model_name": model_name,
        "setting": setting,
        "compared": compared,
        "explanation": explanation,
        "importances": importances,
        "samples": samples,
    }
