import io
import os

import pandas as pd
from django.conf import settings
from django.shortcuts import redirect, render

from . import ml

# The friendly built-in example: the Iris flowers dataset.
EXAMPLE_CSV = settings.BASE_DIR / "project1" / "data" / "iris.csv"

# How many rows we show as a preview (we don't want to overwhelm anyone).
PREVIEW_ROWS = 4


def index(request):
    """Step 1 — the welcome screen: upload a file or use the example."""
    return render(request, "project1/index.html")


def _remember_table(request, csv_text, source):
    """Check the CSV makes sense and keep it for the next steps.

    Returns a friendly error message (str) if something's wrong, else None.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return ("This file doesn't look like a table we can read. "
                "Try a different file, or use our example below.")

    if df.shape[0] < 1 or df.shape[1] < 2:
        return ("This table needs at least two columns and one row of "
                "information. Try another file, or use our example below.")

    request.session["p1_csv"] = csv_text
    request.session["p1_source"] = source
    return None


def upload(request):
    """Handle a file the user chose, then move to Step 2."""
    if request.method == "POST" and request.FILES.get("file"):
        try:
            csv_text = request.FILES["file"].read().decode("utf-8")
        except Exception:
            return render(request, "project1/index.html", {
                "error": "We couldn't open that file. Please choose a CSV file."
            })

        error = _remember_table(request, csv_text, "your file")
        if error:
            return render(request, "project1/index.html", {"error": error})
        return redirect("project1:data")

    # No file chosen — gently send them back.
    return render(request, "project1/index.html", {
        "error": "Please choose a file first, or use our example below."
    })


def example(request):
    """Load the built-in example flowers, then move to Step 2."""
    csv_text = EXAMPLE_CSV.read_text(encoding="utf-8")
    _remember_table(request, csv_text, "our example flowers")
    return redirect("project1:data")


def data(request):
    """Step 2 — show the information as a friendly table."""
    csv_text = request.session.get("p1_csv")
    if not csv_text:
        # Nothing loaded yet — start at the beginning.
        return redirect("project1:index")

    df = pd.read_csv(io.StringIO(csv_text))
    columns = list(df.columns)

    # Draw the data picture. One file per session so users don't clash.
    if not request.session.session_key:
        request.session.save()
    plot_name = f"project1_plot_{request.session.session_key}.png"
    drawn, problem, caption = ml.make_picture(
        df, os.path.join(settings.MEDIA_ROOT, plot_name))

    # A second picture: how the thing-to-guess is spread out.
    dist_name = f"project1_dist_{request.session.session_key}.png"
    dist_drawn, dist_caption = ml.make_distribution_picture(
        df, os.path.join(settings.MEDIA_ROOT, dist_name))

    context = {
        "source": request.session.get("p1_source", "your file"),
        "columns": columns,
        "rows": df.head(PREVIEW_ROWS).values.tolist(),
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "showing": min(PREVIEW_ROWS, df.shape[0]),
        "more": max(df.shape[0] - PREVIEW_ROWS, 0),
        "target": columns[-1],
        "plot_url": (settings.MEDIA_URL + plot_name) if drawn else None,
        "plot_caption": caption,
        "dist_url": (settings.MEDIA_URL + dist_name) if dist_drawn else None,
        "dist_caption": dist_caption,
        "problem": problem,
    }
    return render(request, "project1/data.html", context)


def choose(request):
    """Step 3 — the one human decision: what should the computer guess?"""
    csv_text = request.session.get("p1_csv")
    if not csv_text:
        return redirect("project1:index")

    df = pd.read_csv(io.StringIO(csv_text))
    columns = list(df.columns)
    return render(request, "project1/choose.html", {
        "columns": columns,
        "default_target": columns[-1],
    })


def train(request):
    """Teach the computer, then show the results."""
    csv_text = request.session.get("p1_csv")
    if not csv_text:
        return redirect("project1:index")

    df = pd.read_csv(io.StringIO(csv_text))
    columns = list(df.columns)

    # The user's choices (all default to a sensible value — one click is enough).
    target = request.POST.get("target") if request.method == "POST" else None
    if target not in columns:
        target = columns[-1]

    override = request.POST.get("problem")
    override = override if override in ("category", "number") else None

    prefer = request.POST.get("prefer")
    prefer = prefer if prefer in ("accurate", "explain") else "accurate"

    result = ml.teach_computer(df, target, problem_override=override, prefer=prefer)
    result["target"] = target

    # In "explain" mode, draw which clues the computer paid attention to.
    if result.get("ok") and result.get("importances"):
        if not request.session.session_key:
            request.session.save()
        imp_name = f"project1_importance_{request.session.session_key}.png"
        ml.draw_importances(
            result["importances"], os.path.join(settings.MEDIA_ROOT, imp_name))
        result["importance_url"] = settings.MEDIA_URL + imp_name

    request.session["p1_result"] = result
    return redirect("project1:results")


def results(request):
    """Show how the computer did, in everyday words."""
    result = request.session.get("p1_result")
    if not result:
        return redirect("project1:index")
    return render(request, "project1/results.html", {"result": result})
