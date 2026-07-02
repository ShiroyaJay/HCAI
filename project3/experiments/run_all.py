"""Offline experiment pipeline for Project 3.

Run everything:            python -m project3.experiments.run_all
Run selected stages only:  python -m project3.experiments.run_all --stages data,task1

Stages write their outputs under project3/artifacts/ (metrics JSON, models,
figures) and project3/data/ (dataset); the Django app only reads those files.
"""

import argparse
import json
import os
import time

from project3 import data

METRICS_DIR = os.path.join(data.ARTIFACTS_DIR, "metrics")
MODELS_DIR = os.path.join(data.ARTIFACTS_DIR, "models")
FIGURES_DIR = os.path.join(data.ARTIFACTS_DIR, "figures")

STAGES = ["data", "task1", "task2", "task3", "task4", "figures", "report"]


def save_metrics(name, metrics):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {path}")


def stage_data():
    data.download()
    train_df, val_df = data.load_train_val()
    test_df = data.load_test()
    print(f"data: train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")


def stage_task1():
    from project3 import ml

    train_df, val_df = data.load_train_val()
    test_df = data.load_test()
    pipe = ml.train_classifier(train_df)
    ml.save_classifier(pipe, MODELS_DIR)
    metrics = {
        "val": ml.evaluate_classifier(pipe, val_df),
        "test": ml.evaluate_classifier(pipe, test_df),
        "train_size": len(train_df),
    }
    save_metrics("task1", metrics)
    print(f"task1: test acc {metrics['test']['accuracy']:.4f}")


def stage_task2():
    from project3 import experts

    test_df = data.load_test()
    metrics = {}
    for name, expert in experts.default_experts().items():
        metrics[name] = experts.expert_report(expert, test_df)
        print(f"task2: {name} test acc {metrics[name]['accuracy']:.4f}")
    save_metrics("task2", metrics)


def stage_task3():
    from project3 import l2d

    metrics = l2d.run_task3(MODELS_DIR)
    save_metrics("task3", metrics)
    print(f"task3: team test acc {metrics['test']['team_accuracy']:.4f} "
          f"(coverage {metrics['test']['coverage']:.3f})")


def stage_task4():
    from project3 import active

    metrics = active.run_task4(MODELS_DIR)
    save_metrics("task4", metrics)


def stage_figures():
    from project3 import plots

    plots.make_all(METRICS_DIR, FIGURES_DIR)
    plots.publish_to_media(FIGURES_DIR)


def stage_report():
    from project3 import report

    out = report.build_report(METRICS_DIR, FIGURES_DIR,
                              os.path.join(data.ARTIFACTS_DIR, "report.pdf"))
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default=",".join(STAGES),
                        help=f"comma-separated subset of: {','.join(STAGES)}")
    args = parser.parse_args()
    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(requested) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")

    for stage in STAGES:
        if stage not in requested:
            continue
        t0 = time.time()
        print(f"=== stage: {stage} ===")
        globals()[f"stage_{stage}"]()
        print(f"=== {stage} done in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
