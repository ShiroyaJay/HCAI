"""Views for Project 3: dashboard over offline experiment artifacts,
report download, and the interactive human-as-expert mode (Task 5)."""

import json
import os

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "report.pdf")

SESSION_KEY = "p3_interactive"


def _load_metrics(name):
    """Return the metrics dict for a task stage, or None if not generated yet."""
    path = os.path.join(METRICS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _figure_url(name):
    filename = f"project3_{name}.png"
    if os.path.exists(os.path.join(settings.MEDIA_ROOT, filename)):
        return settings.MEDIA_URL + filename
    return None


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
    task4 = _load_metrics("task4")
    context = {
        "task1": _load_metrics("task1"),
        "task2": _load_metrics("task2"),
        "task3": _load_metrics("task3"),
        "task4": task4,
        "task4_summary": _task4_summary(task4),
        "figures": {name: _figure_url(name) for name in [
            "task1_confusion", "task2_expert_profile",
            "task3_coverage_accuracy", "task3_deferral",
            "task4_learning_curves",
        ]},
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
    idx = loop.next_query(MODELS_DIR, state)
    context = {
        "ready": True,
        "article": loop.article(MODELS_DIR, idx) if idx is not None else None,
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
