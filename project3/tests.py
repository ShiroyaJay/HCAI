"""Tests for Project 3 logic and views.

Logic is tested on small synthetic fixtures; no network access, downloaded
dataset or generated artifacts are required.
"""

import os
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
from django.test import TestCase
from django.urls import reverse

from . import active, experts, l2d


def _synthetic_df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=n)
    texts = ["word " * (5 + int(k) * 3) for k in labels]
    return pd.DataFrame({"text": texts, "label": labels, "key": np.arange(n)})


class ExpertTests(TestCase):
    def test_deterministic(self):
        df = _synthetic_df()
        expert = experts.ClassConditionalExpert()
        first = expert.predict(df)
        second = expert.predict(df)
        np.testing.assert_array_equal(first, second)

    def test_class_conditional_profile(self):
        df = _synthetic_df()
        expert = experts.ClassConditionalExpert(strong_classes=(2, 3),
                                                p_strong=0.95, p_weak=0.60)
        pred = expert.predict(df)
        correct = pred == df["label"].values
        for k in range(4):
            acc = correct[df["label"].values == k].mean()
            target = 0.95 if k in (2, 3) else 0.60
            self.assertAlmostEqual(acc, target, delta=0.05)

    def test_report_shape(self):
        df = _synthetic_df(n=400)
        rep = experts.expert_report(experts.ClassConditionalExpert(), df)
        self.assertIn("accuracy", rep)
        self.assertEqual(len(rep["per_class"]), 4)


class DeferralTests(TestCase):
    y = np.array([0, 1, 2, 3])
    clf_pred = np.array([0, 1, 0, 0])      # right on the first two only
    expert_pred = np.array([1, 1, 2, 0])   # right on examples 1 and 2 only

    def test_team_predictions_and_metrics(self):
        defer = np.array([False, False, True, False])
        team = l2d.team_predictions(self.clf_pred, self.expert_pred, defer)
        np.testing.assert_array_equal(team, [0, 1, 2, 0])
        stats = l2d.evaluate_team(self.y, self.clf_pred, self.expert_pred, defer)
        self.assertAlmostEqual(stats["team_accuracy"], 0.75)
        self.assertAlmostEqual(stats["coverage"], 0.75)
        # The only needed deferral (clf wrong AND expert right) is example 2.
        self.assertAlmostEqual(stats["deferral_precision"], 1.0)
        self.assertAlmostEqual(stats["deferral_recall"], 1.0)

    def test_oracle_at_least_classifier(self):
        base = l2d.baselines(self.y, self.clf_pred, self.expert_pred,
                             deferral_rate=0.25, seed=0)
        self.assertGreaterEqual(base["oracle_deferral"], base["classifier_alone"])
        self.assertGreaterEqual(base["oracle_deferral"], base["expert_alone"])

    def test_curve_endpoints(self):
        scores = np.array([0.1, -0.2, 0.4, -0.3])
        curve = l2d.coverage_accuracy_curve(self.y, self.clf_pred,
                                            self.expert_pred, scores, n_points=3)
        clf_acc = (self.clf_pred == self.y).mean()
        expert_acc = (self.expert_pred == self.y).mean()
        self.assertAlmostEqual(curve[0]["team_accuracy"], clf_acc)
        self.assertAlmostEqual(curve[-1]["team_accuracy"], expert_acc)


class ActiveLearningTests(TestCase):
    def test_selectors_return_fresh_indices(self):
        rng = np.random.default_rng(0)
        candidates = np.arange(50)
        conf = np.linspace(0.5, 1.0, 50)
        g_pool = np.linspace(0.0, 1.0, 50)
        for fn in (active.select_random, active.select_clf_uncertainty,
                   active.select_deferral_boundary):
            picked = fn(rng, candidates, 10, clf_conf=conf, g_pool=g_pool)
            self.assertEqual(len(picked), 10)
            self.assertEqual(len(set(picked.tolist())), 10)
            self.assertTrue(set(picked.tolist()) <= set(candidates.tolist()))

    def test_boundary_falls_back_to_random_without_g(self):
        rng = np.random.default_rng(0)
        picked = active.select_deferral_boundary(
            rng, np.arange(20), 5, clf_conf=np.full(20, 0.9), g_pool=None
        )
        self.assertEqual(len(picked), 5)

    def test_random_reproducible(self):
        a = active.select_random(np.random.default_rng(7), np.arange(100), 10)
        b = active.select_random(np.random.default_rng(7), np.arange(100), 10)
        np.testing.assert_array_equal(a, b)


class ViewTests(TestCase):
    def test_index_degrades_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("project3.views.METRICS_DIR", tmp), \
                 mock.patch("project3.views.REPORT_PATH",
                            os.path.join(tmp, "report.pdf")):
                response = self.client.get(reverse("project3:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "run_all")
        self.assertContains(response, "Report not generated yet")

    def test_report_404_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("project3.views.REPORT_PATH",
                            os.path.join(tmp, "report.pdf")):
                response = self.client.get(reverse("project3:report"))
        self.assertEqual(response.status_code, 404)

    def test_interactive_degrades_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("project3.views.MODELS_DIR", tmp):
                response = self.client.get(reverse("project3:interactive"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "interactive mode needs")
