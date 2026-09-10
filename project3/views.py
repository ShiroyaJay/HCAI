"""Views for Project 3: dashboard over offline experiment artifacts,
report download, and the interactive human-as-expert mode (Task 5)."""

import json
import os
import shutil

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "report.pdf")

SESSION_KEY = "p3_interactive"

FIGURE_NAMES = [
    "task1_confusion", "task2_expert_profile",
    "task3_coverage_accuracy", "task3_deferral",
    "task4_learning_curves",
]


def _load_metrics(name):
    """Return the metrics dict for a task stage, or None if not generated yet."""
    path = os.path.join(METRICS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _figure_url(name):
    """URL for one stored figure, publishing it into MEDIA_ROOT on first use.

    The figures are committed under artifacts/figures/, but the page serves them
    from MEDIA_ROOT. run_all's ``publish_to_media`` normally does that copy; doing
    it lazily here too is what lets the page show its plots without the pipeline
    having been run.
    """
    filename = f"project3_{name}.png"
    published = os.path.join(settings.MEDIA_ROOT, filename)
    if not os.path.exists(published):
        source = os.path.join(FIGURES_DIR, f"{name}.png")
        if not os.path.exists(source):
            return None
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        shutil.copy(source, published)
    return settings.MEDIA_URL + filename


def _operating_point(task3, requested):
    """The human-chosen deferral rate, resolved to the nearest point on the
    stored test coverage-accuracy curve. The auto-tuned system is kept as the
    reference. Returns None until the task 3 artifacts exist."""
    curve = (task3 or {}).get("coverage_accuracy_curve")
    if not curve:
        return None
    auto_rate = 1.0 - task3["test"]["coverage"]
    try:
        rate = float(requested)
    except (TypeError, ValueError):
        rate = auto_rate
    rate = max(0.0, min(1.0, rate))
    point = min(curve, key=lambda p: abs(p["deferral_rate"] - rate))
    return {
        "rate": point["deferral_rate"],
        "rate_percent": round(point["deferral_rate"] * 100),
        "team_accuracy": point["team_accuracy"],
        "auto_rate_percent": round(auto_rate * 100),
        "auto_team_accuracy": task3["test"]["team_accuracy"],
    }


def _task4_summary(task4):
    """Mean team accuracy per strategy at each budget, for the results table."""
    if not task4:
        return None
    rows = []
    for name, label in [("random", "Random"),
                        ("clf_uncertainty", "Classifier uncertainty"),
                        ("deferral_boundary", "Deferral boundary")]:
        runs = task4["strategies"][name]["runs"]
        accs = [
            sum(run[i]["team_accuracy"] for run in runs) / len(runs)
            for i in range(len(task4["budgets"]))
        ]
        rows.append({"strategy": label, "accuracies": accs})
    return {"budgets": task4["budgets"], "rows": rows}


def index(request):
    task3 = _load_metrics("task3")
    task4 = _load_metrics("task4")
    figures = {name: _figure_url(name) for name in FIGURE_NAMES}
    context = {
        "task1": _load_metrics("task1"),
        "task2": _load_metrics("task2"),
        "task3": task3,
        "task3_op": _operating_point(task3, request.GET.get("defer_rate")),
        "task4": task4,
        "task4_summary": _task4_summary(task4),
        "figures": figures,
        # The metrics are vendored but the plots are drawn from them, so the
        # tables can be complete while every figure is absent. Say so rather
        # than leaving unexplained gaps between the tables.
        "figures_missing": [n for n, url in figures.items() if url is None],
        "report_available": os.path.exists(REPORT_PATH),
    }
    return render(request, "project3/index.html", context)


def download_report(request):
    if not os.path.exists(REPORT_PATH):
        raise Http404("Report not generated yet.")
    return FileResponse(
        open(REPORT_PATH, "rb"), as_attachment=True, filename="project3_report.pdf"
    )


def interactive(request):
    from . import data
    from . import interactive as loop

    if not loop.available(MODELS_DIR):
        return render(request, "project3/interactive.html", {"ready": False})

    state = request.session.get(SESSION_KEY) or loop.new_state()
    request.session[SESSION_KEY] = state
    finished = state.get("finished", False)
    idx = None if finished else loop.next_query(MODELS_DIR, state)
    context = {
        "ready": True,
        "article": loop.article(MODELS_DIR, idx) if idx is not None else None,
        "finished": finished,
        "class_names": data.CLASS_NAMES,
        "metrics": loop.live_metrics(MODELS_DIR, state),
    }
    return render(request, "project3/interactive.html", context)


def interactive_start(request):
    from . import interactive as loop

    if request.method == "POST":
        request.session[SESSION_KEY] = loop.new_state()
    return redirect("project3:interactive")


def interactive_label(request):
    from . import interactive as loop

    if request.method == "POST" and loop.available(MODELS_DIR):
        state = request.session.get(SESSION_KEY) or loop.new_state()
        try:
            idx = int(request.POST["article_index"])
            label = int(request.POST["label"])
        except (KeyError, ValueError):
            return redirect("project3:interactive")
        if 0 <= label < 4:
            request.session[SESSION_KEY] = loop.record_label(state, idx, label)
    return redirect("project3:interactive")


def interactive_skip(request):
    from . import interactive as loop

    if request.method == "POST" and loop.available(MODELS_DIR):
        state = request.session.get(SESSION_KEY) or loop.new_state()
        try:
            idx = int(request.POST["article_index"])
        except (KeyError, ValueError):
            return redirect("project3:interactive")
        request.session[SESSION_KEY] = loop.record_skip(state, idx)
    return redirect("project3:interactive")


def interactive_finish(request):
    from . import interactive as loop

    if request.method == "POST":
        state = request.session.get(SESSION_KEY) or loop.new_state()
        request.session[SESSION_KEY] = loop.record_finish(state)
    return redirect("project3:interactive")
