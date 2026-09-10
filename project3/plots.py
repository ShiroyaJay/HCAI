"""Report/dashboard figures for Project 3, generated from the metrics JSONs.

All figures use a shared quiet style: hairline grid, muted axis ink, series
colors from a CVD-validated palette (blue, aqua, yellow — assigned in fixed
order). Sub-3:1 series (aqua, yellow) get direct labels or a legend, and every
number is also in the dashboard tables / report tables.
"""

import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import data

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#1baf7a", "#eda100"]  # blue, aqua, yellow (fixed order)

_BLUES = LinearSegmentedColormap.from_list("p3blues", ["#fcfcfb", "#cde2fb", "#2a78d6", "#0d366b"])

FIGURE_NAMES = [
    "task1_confusion",
    "task2_expert_profile",
    "task3_coverage_accuracy",
    "task3_deferral",
    "task4_learning_curves",
]


def _style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)


def _new_fig(width, height, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    return fig, axes


def _save(fig, figures_dir, name):
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def _load(metrics_dir, name):
    with open(os.path.join(metrics_dir, f"{name}.json")) as f:
        return json.load(f)


def task1_confusion(metrics_dir, figures_dir):
    m = _load(metrics_dir, "task1")
    cm = np.array(m["test"]["confusion_matrix"], dtype=float)
    row_frac = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = _new_fig(4.6, 4.2)
    ax.imshow(row_frac, cmap=_BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(data.N_CLASSES), data.CLASS_NAMES, color=SECONDARY)
    ax.set_yticks(range(data.N_CLASSES), data.CLASS_NAMES, color=SECONDARY)
    ax.set_xlabel("Predicted", color=SECONDARY)
    ax.set_ylabel("True", color=SECONDARY)
    ax.tick_params(colors=MUTED, labelsize=9)
    for i in range(data.N_CLASSES):
        for j in range(data.N_CLASSES):
            color = "white" if row_frac[i, j] > 0.5 else INK
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title(f"Classifier confusion matrix (test acc {m['test']['accuracy']:.1%})",
                 color=INK, fontsize=11)
    _save(fig, figures_dir, "task1_confusion")


def task2_expert_profile(metrics_dir, figures_dir):
    m1 = _load(metrics_dir, "task1")
    m2 = _load(metrics_dir, "task2")
    clf_recall = [d["recall"] for d in m1["test"]["per_class"]]
    cls_exp = [d["accuracy"] for d in m2["class_conditional"]["per_class"]]
    len_exp = [d["accuracy"] for d in m2["length_conditional"]["per_class"]]

    fig, (ax1, ax2) = _new_fig(9.4, 3.6, ncols=2)

    x = np.arange(data.N_CLASSES)
    w = 0.26
    labels = ["Classifier", "Class-cond. expert", "Length-cond. expert"]
    for k, (vals, label) in enumerate(zip([clf_recall, cls_exp, len_exp], labels)):
        ax1.bar(x + (k - 1) * w, vals, width=w - 0.03, color=SERIES[k], label=label)
    _style(ax1)
    ax1.set_xticks(x, data.CLASS_NAMES)
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_ylabel("Accuracy on true class", color=SECONDARY, fontsize=9)
    ax1.set_title("Accuracy by topic (test set)", color=INK, fontsize=11)
    ax1.legend(fontsize=8, frameon=False, labelcolor=SECONDARY,
               loc="upper center", ncols=3)

    bins = [d["bin"] for d in m2["class_conditional"]["by_length"]]
    x2 = np.arange(len(bins))
    for k, name in enumerate(["class_conditional", "length_conditional"]):
        vals = [d["accuracy"] for d in m2[name]["by_length"]]
        ax2.bar(x2 + (k - 0.5) * w, vals, width=w - 0.03, color=SERIES[k + 1],
                label=labels[k + 1])
    _style(ax2)
    ax2.set_xticks(x2, bins, fontsize=8)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_title("Expert accuracy by article length", color=INK, fontsize=11)
    ax2.legend(fontsize=8, frameon=False, labelcolor=SECONDARY,
               loc="upper center", ncols=2)

    fig.tight_layout()
    _save(fig, figures_dir, "task2_expert_profile")


def task3_coverage_accuracy(metrics_dir, figures_dir):
    m = _load(metrics_dir, "task3")
    curve = m["coverage_accuracy_curve"]
    rates = [p["deferral_rate"] for p in curve]
    accs = [p["team_accuracy"] for p in curve]
    b = m["test"]["baselines"]

    fig, ax = _new_fig(6.4, 4.0)
    _style(ax)
    ax.plot(rates, accs, color=SERIES[0], linewidth=2, solid_capstyle="round",
            label="Team (defer by score rank)")
    ax.axhline(b["classifier_alone"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.99, b["classifier_alone"] - 0.012, "classifier alone", ha="right",
            fontsize=8, color=SECONDARY)
    ax.axhline(b["oracle_deferral"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.99, b["oracle_deferral"] + 0.005, "oracle deferral", ha="right",
            fontsize=8, color=SECONDARY)

    op_rate = 1 - m["test"]["coverage"]
    op_acc = m["test"]["team_accuracy"]
    ax.plot([op_rate], [op_acc], "o", markersize=8, color=SERIES[0],
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"operating point\n({op_rate:.0%} deferred, {op_acc:.1%})",
                (op_rate, op_acc), textcoords="offset points", xytext=(14, 6),
                fontsize=8, color=SECONDARY)

    ax.set_xlabel("Deferral rate (fraction sent to expert)", color=SECONDARY, fontsize=9)
    ax.set_ylabel("Team accuracy (test)", color=SECONDARY, fontsize=9)
    ax.set_title("Coverage-accuracy trade-off", color=INK, fontsize=11)
    _save(fig, figures_dir, "task3_coverage_accuracy")


def task3_deferral(metrics_dir, figures_dir):
    m = _load(metrics_dir, "task3")
    t = m["test"]
    b = t["baselines"]

    fig, (ax1, ax2) = _new_fig(9.4, 3.4, ncols=2)

    systems = [
        ("Expert alone", b["expert_alone"]),
        ("Random deferral", b["random_deferral_same_rate"]),
        ("Classifier alone", b["classifier_alone"]),
        ("Learned deferral", t["team_accuracy"]),
        ("Oracle deferral", b["oracle_deferral"]),
    ]
    names = [s[0] for s in systems]
    vals = [s[1] for s in systems]
    colors = [BASELINE, BASELINE, BASELINE, SERIES[0], BASELINE]
    ax1.barh(range(len(systems)), vals, height=0.55, color=colors)
    _style(ax1, ygrid=False)
    ax1.grid(axis="x", color=GRID, linewidth=1)
    ax1.set_axisbelow(True)
    ax1.set_yticks(range(len(systems)), names, color=SECONDARY)
    ax1.set_xlim(0, 1.08)
    for i, v in enumerate(vals):
        ax1.text(v + 0.006, i, f"{v:.1%}", va="center", fontsize=8, color=SECONDARY)
    ax1.set_title("Test accuracy by system", color=INK, fontsize=11)

    rates = [d["rate"] for d in t["per_class_deferral_rate"]]
    ax2.bar(range(data.N_CLASSES), rates, width=0.5, color=SERIES[0])
    _style(ax2)
    ax2.set_xticks(range(data.N_CLASSES), data.CLASS_NAMES)
    for i, v in enumerate(rates):
        ax2.text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=8, color=SECONDARY)
    ax2.set_ylabel("Deferral rate", color=SECONDARY, fontsize=9)
    ax2.set_title("Where the system defers (true class)", color=INK, fontsize=11)

    fig.tight_layout()
    _save(fig, figures_dir, "task3_deferral")


def task4_learning_curves(metrics_dir, figures_dir):
    m = _load(metrics_dir, "task4")
    budgets = m["budgets"]
    labels = {
        "random": "Random",
        "clf_uncertainty": "Classifier uncertainty",
        "deferral_boundary": "Deferral boundary",
    }

    fig, ax = _new_fig(6.8, 4.2)
    _style(ax)
    for k, (name, label) in enumerate(labels.items()):
        runs = m["strategies"][name]["runs"]
        accs = np.array([[c["team_accuracy"] for c in run] for run in runs])
        mean, std = accs.mean(axis=0), accs.std(axis=0)
        ax.plot(budgets, mean, color=SERIES[k], linewidth=2, marker="o",
                markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label)
        ax.fill_between(budgets, mean - std, mean + std, color=SERIES[k], alpha=0.10)

    refs = m["references"]
    ax.axhline(refs["classifier_alone_val"], color=MUTED, linewidth=1,
               linestyle=(0, (4, 3)))
    ax.text(budgets[0], refs["classifier_alone_val"] + 0.001, "classifier alone",
            ha="left", va="bottom", fontsize=8, color=SECONDARY)
    ax.axhline(refs["full_info_team_accuracy_val"], color=MUTED, linewidth=1,
               linestyle=(0, (4, 3)))
    ax.text(budgets[0], refs["full_info_team_accuracy_val"] - 0.001,
            "full expert data (110k labels)", ha="left", va="top",
            fontsize=8, color=SECONDARY)

    ax.set_xscale("log")
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlabel("Expert queries", color=SECONDARY, fontsize=9)
    ax.set_ylabel("Team accuracy (validation)", color=SECONDARY, fontsize=9)
    ax.set_title("Active learning: team accuracy vs expert queries (3 seeds)",
                 color=INK, fontsize=11)
    ax.legend(fontsize=8, frameon=False, labelcolor=SECONDARY, loc="center left")
    _save(fig, figures_dir, "task4_learning_curves")


def make_all(metrics_dir, figures_dir):
    task1_confusion(metrics_dir, figures_dir)
    task2_expert_profile(metrics_dir, figures_dir)
    task3_coverage_accuracy(metrics_dir, figures_dir)
    task3_deferral(metrics_dir, figures_dir)
    task4_learning_curves(metrics_dir, figures_dir)


def publish_to_media(figures_dir):
    """Copy the figures to media/ so the dashboard can embed them."""
    media_dir = os.path.join(os.path.dirname(__file__), "..", "media")
    os.makedirs(media_dir, exist_ok=True)
    for name in FIGURE_NAMES:
        src = os.path.join(figures_dir, f"{name}.png")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(media_dir, f"project3_{name}.png"))
    print(f"published figures to {media_dir}")
